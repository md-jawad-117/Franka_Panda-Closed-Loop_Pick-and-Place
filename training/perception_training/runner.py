"""Reproducible training/validation selection and separate checkpoint evaluation."""
from collections import defaultdict
import csv
import hashlib
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import Dataset,DataLoader

from .data import MODES,load_arrays,preflight,read_manifest,sha,safe_path
from .metrics import MetricAccumulator
from .model import MODEL_VERSION,TinyUNet
from .objectives import OBJECTIVES,SELECTIONS,loss_for,camera_macro_iou,selection_score


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def seed_worker(_):
    value=torch.initial_seed()%2**32; np.random.seed(value);random.seed(value)


class SegmentationDataset(Dataset):
    def __init__(self,root,split,mode='rgbd',augment=False,limit=None,camera=None):
        self.root=Path(root);self.mode=mode;self.augment=augment
        if camera not in (None,'wrist_cam','top_cam'):raise ValueError('Unknown camera filter')
        _,rows=read_manifest(root);self.rows=[r for r in rows if r['split']==split and (camera is None or r['camera']==camera)]
        if limit is not None: self.rows=self.rows[:limit]
        if not self.rows: raise ValueError(f'Empty split: {split}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        x,y=load_arrays(self.root,self.rows[index],self.mode,self.augment)
        return torch.from_numpy(x),torch.from_numpy(y),index


def make_loader(dataset,batch_size,workers,shuffle,seed,device):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=device.type=='cuda',worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),drop_last=False)


@torch.inference_mode()
def score(model,loader,device,weights,objective='legacy'):
    model.eval();metrics=MetricAccumulator();cameras=defaultdict(MetricAccumulator); loss_sum=0;count=0
    for x,y,indices in loader:
        x=x.to(device);y=y.to(device);logits=model(x)
        loss=loss_for(logits,y,weights,objective)
        loss_sum+=float(loss)*len(x);count+=len(x)
        predictions=logits.argmax(1).cpu().numpy();labels=y.cpu().numpy()
        for index,pred,label in zip(indices.tolist(),predictions,labels):
            metrics.update(pred,label);cameras[loader.dataset.rows[index]['camera']].update(pred,label)
    report=metrics.report();report['loss']=loss_sum/count
    report['by_camera']={key:value.report() for key,value in cameras.items()}
    try:report['camera_macro_iou']=camera_macro_iou(report['by_camera'])
    except ValueError:report['camera_macro_iou']=None
    return report


def train(root,output,epochs=30,batch_size=8,seed=17,mode='rgbd',base=16,workers=2,
          learning_rate=.001,patience=8,device='cuda',amp=True,limit_train=None,limit_val=None,
          objective='legacy',selection='global_foreground_iou',camera=None):
    if epochs<1 or batch_size<1 or workers<0 or patience<1 or learning_rate<=0: raise ValueError('Invalid training settings')
    if objective not in OBJECTIVES or selection not in SELECTIONS:raise ValueError('Unknown objective/selection')
    root=Path(root);out=Path(output)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite training run: {out}')
    target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; enable the Kaggle GPU or explicitly use --device cpu for a smoke test')
    seed_everything(seed)
    # Input validation does not make a prediction or select a model using test scenes.
    audit=preflight(root)
    if camera is not None and selection=='camera_macro_iou':raise ValueError('A single-camera model requires view_foreground_iou selection')
    if camera is None and selection=='view_foreground_iou':raise ValueError('View selection requires a camera filter')
    train_data=SegmentationDataset(root,'train',mode,True,limit_train,camera)
    val_data=SegmentationDataset(root,'validation',mode,False,limit_val,camera)
    if set(r['scene_id'] for r in train_data.rows)&set(r['scene_id'] for r in val_data.rows): raise ValueError('Train/validation overlap')
    train_loader=make_loader(train_data,batch_size,workers,True,seed,target)
    val_loader=make_loader(val_data,batch_size,workers,False,seed+1,target)
    model=TinyUNet(MODES[mode],base).to(target)
    initial_hash=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        initial_hash.update(name.encode());initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    # Weights use training pixels only, never validation/test class frequencies.
    counts=np.asarray(audit['train_class_pixels'],float)
    raw=np.sqrt(counts.sum()/counts);raw=np.minimum(raw/raw.min(),10)
    weights=torch.tensor(raw/raw.mean(),dtype=torch.float32,device=target)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.0001)
    mixed=bool(amp and target.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=mixed)
    source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    config={'model_version':MODEL_VERSION,'input_mode':mode,'input_channels':MODES[mode],'base':base,
            'epochs':epochs,'batch_size':batch_size,'seed':seed,'workers':workers,'learning_rate':learning_rate,
            'patience':patience,'class_weights':weights.cpu().tolist(),'device':str(target),'amp':mixed,
            'python':platform.python_version(),'torch':str(torch.__version__),'numpy':str(np.__version__),
            'gpu':torch.cuda.get_device_name(target) if target.type=='cuda' else None,
            'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_root':str(root),
            'dataset_metadata_sha256':audit['dataset_metadata_sha256'],'manifest_sha256':audit['manifest_sha256'],
            'source_sha256':source,'train_samples':len(train_data),'validation_samples':len(val_data),
            'train_scenes':sorted({r['scene_id'] for r in train_data.rows}),
            'validation_scenes':sorted({r['scene_id'] for r in val_data.rows}),
            'objective':objective,'selection_metric':selection,'camera_filter':camera,
            'class_weight_scope':'all training views; identical fixed weights across shared and camera-specific models',
            'selection':f'highest validation {selection}; test never used for checkpoint selection',
            'balanced_objective_definition':'per-image present-class-normalized CE + per-image/class foreground Dice; absent classes use mean -log(1-p)' if objective!='legacy' else None,
            'class_weights_used':objective=='legacy',
            'depth_scaling':'clip optical-z depth to 0..2.5 metres, divide by 2.5; append validity channel',
            'parameters':sum(p.numel() for p in model.parameters()),
            'initial_state_sha256':initial_hash.hexdigest(),
            'development_smoke':limit_train is not None or limit_val is not None,
            'determinism':'fixed seeds and deterministic algorithms with warnings; bitwise equality across hardware is not promised'}
    out.mkdir(parents=True);write_json(out/'run_config.json',config);write_json(out/'preflight.json',audit)
    best=-float('inf');best_global=-float('inf');best_epoch=None;best_global_epoch=None;stale=0;t0=time.perf_counter();order_hash=hashlib.sha256()
    fields=['epoch','train_loss','val_loss','val_foreground_macro_iou','val_capsule_iou','val_fixture_iou',
            'val_camera_macro_iou','val_wrist_capsule_iou','val_wrist_fixture_iou','val_top_capsule_iou','val_top_fixture_iou',
            'val_selection_score','epoch_seconds']
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,epochs+1):
            model.train();total=0;seen=0;start=time.perf_counter()
            for x,y,indices in train_loader:
                order_hash.update(indices.numpy().astype('<i8').tobytes())
                x=x.to(target,non_blocking=True);y=y.to(target,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=target.type,dtype=torch.float16,enabled=mixed):
                    logits=model(x);loss=loss_for(logits,y,weights,objective)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(optimizer);scaler.update();total+=float(loss.detach())*len(x);seen+=len(x)
            validation=score(model,val_loader,target,weights,objective);value=selection_score(validation,selection)
            if value is None: raise RuntimeError('No foreground validation metric')
            row=dict(epoch=epoch,train_loss=total/seen,val_loss=validation['loss'],val_foreground_macro_iou=validation['foreground_macro_iou'],
                     val_capsule_iou=validation['classes']['capsule']['iou'],val_fixture_iou=validation['classes']['fixture']['iou'],
                     val_camera_macro_iou=validation['camera_macro_iou'],val_selection_score=value,
                     **{f'val_{view}_{cls}_iou':validation['by_camera'].get(camera,{}).get('classes',{}).get(cls,{}).get('iou')
                        for view,camera in (('wrist','wrist_cam'),('top','top_cam')) for cls in ('capsule','fixture')},
                     epoch_seconds=time.perf_counter()-start)
            writer.writerow(row);f.flush();print(json.dumps(row),flush=True)
            if value>best:
                best=value;best_epoch=epoch;stale=0
                torch.save({'model_state':model.state_dict(),'config':config,'epoch':epoch,'validation':validation,
                            'selected_by':selection,'selection_score':value},out/'best.pt')
                write_json(out/'best_validation.json',validation)
            else: stale+=1
            global_score=validation['foreground_macro_iou']
            if global_score>best_global:
                best_global=global_score;best_global_epoch=epoch
                torch.save({'model_state':model.state_dict(),'config':config,'epoch':epoch,'validation':validation,
                            'selected_by':'global_foreground_iou','selection_score':global_score},out/'best_global.pt')
                write_json(out/'best_global_validation.json',validation)
            if stale>=patience: break
    write_json(out/'training_complete.json',{'complete':True,'epochs_run':epoch,'best_validation_foreground_macro_iou':best_global,
               'best_selection_score':best,'selection_metric':selection,'selected_epoch':best_epoch,'best_global_epoch':best_global_epoch,
               'sample_order_sha256':order_hash.hexdigest(),
               'seconds':time.perf_counter()-t0})
    return out/'best.pt'


def visibility_bucket(info):
    if not info['in_frustum']: return 'out_of_view'
    f=info['visible_fraction']
    if f==0: return 'fully_occluded'
    if f<.25: return 'severely_occluded'
    if f<.99: return 'partly_occluded'
    return 'mostly_visible'


def prediction_sheet(root,examples,output):
    palette=np.array([[0,0,0],[255,0,255],[0,255,255]],np.uint8)
    panels=[]
    for row,pred,label in examples:
        rgb=Image.open(safe_path(root,f'observations/{row["sample_id"]}/rgb.png')).convert('RGB')
        w,h=rgb.size;panel=Image.new('RGB',(w*3,h+24),(24,24,24))
        panel.paste(rgb,(0,24));panel.paste(Image.fromarray(palette[label]),(w,24));panel.paste(Image.fromarray(palette[pred]),(w*2,24))
        ImageDraw.Draw(panel).text((4,5),f'{row["sample_id"]}: RGB | truth | prediction',fill='white')
        panels.append(panel)
    sheet=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)))
    y=0
    for panel in panels: sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(output)


def evaluate(root,checkpoint,output,split='test',device='cuda',batch_size=8,workers=2):
    out=Path(output)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite evaluation: {out}')
    target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    audit=preflight(root)
    # Only load checkpoints produced by this project/user. Arbitrary pickle files
    # are not accepted; modern weights_only mode is kept enabled explicitly.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=saved['config']
    if cfg['model_version']!=MODEL_VERSION: raise ValueError('Unsupported model version')
    if cfg['dataset_metadata_sha256']!=audit['dataset_metadata_sha256'] or cfg['manifest_sha256']!=audit['manifest_sha256']:
        raise ValueError('Checkpoint/dataset provenance mismatch')
    model=TinyUNet(cfg['input_channels'],cfg['base']);model.load_state_dict(saved['model_state']);model.to(target).eval()
    data=SegmentationDataset(root,split,cfg['input_mode'],camera=cfg.get('camera_filter'))
    if split=='test' and set(r['scene_id'] for r in data.rows)&set(cfg['train_scenes']+cfg['validation_scenes']): raise ValueError('Test scene leakage')
    loader=make_loader(data,batch_size,workers,False,cfg['seed'],target)
    total=MetricAccumulator();cameras=defaultdict(MetricAccumulator);groups=defaultdict(MetricAccumulator);scenes=defaultdict(MetricAccumulator)
    examples=[];sample_rows=[];start=time.perf_counter()
    # Two fixed, prediction-independent views for each scene/camera instead of
    # showing only the first scene. Cap the sheet at four scenes for readability.
    preview_scenes=sorted({r['scene_id'] for r in data.rows})[:4]
    preview_ids=set()
    for scene in preview_scenes:
        for camera in ('wrist_cam','top_cam'):
            for stage in (2,6):
                candidates=[r for r in data.rows if r['scene_id']==scene and r['camera']==camera and r['frame_index']==stage]
                if candidates:preview_ids.add(candidates[0]['sample_id'])
    with torch.inference_mode():
        for x,y,indices in loader:
            predictions=model(x.to(target)).argmax(1).cpu().numpy();labels=y.numpy()
            for index,pred,label in zip(indices.tolist(),predictions,labels):
                row=data.rows[index];total.update(pred,label);cameras[row['camera']].update(pred,label);scenes[row['scene_id']].update(pred,label)
                # Privileged visibility is used AFTER prediction for evaluation
                # grouping only, never passed into the model or training loader.
                a=json.loads(safe_path(root,f'labels/{row["sample_id"]}/annotation.json').read_text())
                for cls in ('1','2'):
                    key=('capsule' if cls=='1' else 'fixture')+'/'+visibility_bucket(a['visibility'][cls])
                    groups[key].update(pred,label)
                single=MetricAccumulator();single.update(pred,label)
                summary=single.report();sample_rows.append({'sample_id':row['sample_id'],'scene_id':row['scene_id'],
                    'camera':row['camera'],'capsule_iou':summary['classes']['capsule']['iou'],'fixture_iou':summary['classes']['fixture']['iou']})
                if row['sample_id'] in preview_ids:examples.append((row,pred.copy(),label.copy()))
    result={'split':split,'development_pilot':True,'checkpoint_sha256':sha(checkpoint),'selected_epoch':saved['epoch'],
            'dataset_metadata_sha256':audit['dataset_metadata_sha256'],'manifest_sha256':audit['manifest_sha256'],
            'torch':str(torch.__version__),'device':str(target),'gpu':torch.cuda.get_device_name(target) if target.type=='cuda' else None,
            'overall':total.report(),'by_camera':{k:v.report() for k,v in cameras.items()},
            'by_visibility':{k:v.report() for k,v in groups.items()},'by_scene':{k:v.report() for k,v in scenes.items()},'sample_metrics':sample_rows,
            'wall_seconds_including_data_loading':time.perf_counter()-start,
            'objective':cfg.get('objective','legacy'),'selected_by':saved.get('selected_by','global_foreground_iou'),
            'camera_filter':cfg.get('camera_filter'),
            'preview_sample_ids':[row['sample_id'] for row,_,_ in examples],
            'limitation':'small development scene split; segmentation only, not 3D pose accuracy or robot success'}
    out.mkdir(parents=True);write_json(out/'evaluation.json',result)
    prediction_sheet(root,examples,out/'predictions.png')
    return result
