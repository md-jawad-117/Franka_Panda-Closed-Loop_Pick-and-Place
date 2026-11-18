"""Fixed-budget paired objective comparison; validation-only GPU handoff."""
from pathlib import Path
import json
import csv
from PIL import Image,ImageDraw

from .data import sha
from .objectives import camera_macro_iou
from .runner import train,evaluate,write_json

EXPERIMENT_ID='perception_balance_v2'


def run_paired(root,output,epochs=30,batch_size=8,seed=17,workers=2,device='cuda',
               limit_train=None,limit_val=None):
    out=Path(output)
    if out.exists():raise FileExistsError(f'Refusing to overwrite experiment: {out}')
    out.mkdir(parents=True)
    spec={'experiment_id':EXPERIMENT_ID,'epochs':epochs,'batch_size':batch_size,'seed':seed,
          'workers':workers,'device':device,'arms':['legacy','image_class_balanced'],
          'model':'unchanged TinyUNet v1, width 16, RGB-D five channels, full 320x240',
          'optimizer':'AdamW lr=0.001 weight_decay=0.0001, identical augmentations',
          'selection':'both global foreground IoU and equal mean of four camera/class IoUs',
          'patience':epochs+1,'early_stopping':'disabled for equal fixed epoch budgets',
          'evaluation':'validation only; no development test inference in this experiment',
          'dataset_metadata_sha256':sha(Path(root)/'dataset.json'),'manifest_sha256':sha(Path(root)/'manifest.jsonl'),
          'development_smoke':limit_train is not None or limit_val is not None}
    write_json(out/'experiment_spec.json',spec)
    configs={};complete={};entries=[]
    for objective in spec['arms']:
        arm=out/objective
        train(root,arm,epochs=epochs,batch_size=batch_size,seed=seed,workers=workers,
              device=device,objective=objective,selection='camera_macro_iou',patience=epochs+1,
              limit_train=limit_train,limit_val=limit_val)
        configs[objective]=json.loads((arm/'run_config.json').read_text())
        complete[objective]=json.loads((arm/'training_complete.json').read_text())
        history=list(csv.DictReader((arm/'history.csv').open()))
        best_meta=json.loads((arm/'best_validation.json').read_text())
        best_row=next(r for r in history if int(r['epoch'])==complete[objective]['selected_epoch'])
        if abs(float(best_row['val_foreground_macro_iou'])-best_meta['foreground_macro_iou'])>1e-9:
            raise RuntimeError('Logged global score disagrees with selected checkpoint validation')
        if abs(float(best_row['val_camera_macro_iou'])-best_meta['camera_macro_iou'])>1e-9:
            raise RuntimeError('Logged camera score disagrees with selected checkpoint validation')
        for selected_by,filename in (('camera_macro_iou','best.pt'),('global_foreground_iou','best_global.pt')):
            report=evaluate(root,arm/filename,arm/f'validation_{selected_by}',split='validation',
                            device=device,batch_size=batch_size,workers=workers)
            by_camera=report['by_camera']
            entries.append({'objective':objective,'selected_by':selected_by,'selected_epoch':report['selected_epoch'],
                            'global_foreground_iou':report['overall']['foreground_macro_iou'],
                            'camera_macro_iou':camera_macro_iou(by_camera),
                            **{f'{view}_{cls}_iou':by_camera[camera]['classes'][cls]['iou']
                               for view,camera in (('wrist','wrist_cam'),('top','top_cam')) for cls in ('capsule','fixture')},
                            'wrist_fixture_absent':by_camera['wrist_cam']['absent_target_frames']['fixture'],
                            'checkpoint_sha256':report['checkpoint_sha256']})
    a,b=spec['arms']
    controls={'same_initial_weights':configs[a]['initial_state_sha256']==configs[b]['initial_state_sha256'],
              'same_sample_order':complete[a]['sample_order_sha256']==complete[b]['sample_order_sha256'],
              'same_epoch_budget':complete[a]['epochs_run']==complete[b]['epochs_run']==epochs,
              'same_dataset':configs[a]['dataset_metadata_sha256']==configs[b]['dataset_metadata_sha256'],
              'same_manifest':configs[a]['manifest_sha256']==configs[b]['manifest_sha256'],
              'same_training_scenes':configs[a]['train_scenes']==configs[b]['train_scenes'],
              'same_validation_scenes':configs[a]['validation_scenes']==configs[b]['validation_scenes']}
    if not all(controls.values()):raise RuntimeError(f'Paired experimental controls failed: {controls}')
    # Fixed matching validation examples: RGB | truth | legacy | balanced.
    panels=[]
    legacy_dir=out/'legacy'/'validation_camera_macro_iou'
    balanced_dir=out/'image_class_balanced'/'validation_camera_macro_iou'
    left_info=json.loads((legacy_dir/'evaluation.json').read_text())
    right_info=json.loads((balanced_dir/'evaluation.json').read_text())
    ids=left_info['preview_sample_ids']
    if ids!=right_info['preview_sample_ids']:raise RuntimeError('Comparison preview samples disagree')
    left=Image.open(legacy_dir/'predictions.png');right=Image.open(balanced_dir/'predictions.png')
    width=left.width//3;height=left.height//len(ids)
    for index,sid in enumerate(ids):
        panel=Image.new('RGB',(width*4,height),(24,24,24));y=index*height
        panel.paste(left.crop((0,y+24,width*3,y+height)),(0,24))
        panel.paste(right.crop((width*2,y+24,width*3,y+height)),(width*3,24))
        ImageDraw.Draw(panel).text((4,5),f'{sid}: RGB | truth | legacy | balanced',fill='white')
        panels.append(panel)
    sheet=Image.new('RGB',(width*4,height*len(ids)))
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*height))
    sheet.save(out/'validation_comparison.png')
    summary={'experiment_id':EXPERIMENT_ID,'paired_controls':controls,'validation_comparison':entries,
             'development_smoke':spec['development_smoke'],
             'limits':['one seed, 20 training/3 validation scenes; no population-level claim',
                       'GPU arithmetic may be nondeterministic despite matched seeds/order',
                       'checkpoint selection is validation-dependent; validation is not unbiased final evaluation',
                       'no model-quality conclusion from limited CPU smoke training']}
    write_json(out/'comparison.json',summary)
    write_json(out/'experiment_complete.json',{'complete':True,'experiment_id':EXPERIMENT_ID})
    return summary
