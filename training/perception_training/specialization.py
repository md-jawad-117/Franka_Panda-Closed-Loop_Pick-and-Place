"""Expanded-data diagnostic: shared model versus known-camera specialists."""
from pathlib import Path
import json,math
import numpy as np
from PIL import Image,ImageDraw

from .data import sha
from .metrics import MetricAccumulator,NAMES
from .objectives import camera_macro_iou
from .runner import train,evaluate,write_json

EXPERIMENT_ID='view_specialization_v3'


def merge_specialist_reports(reports,image_pixels):
    """Pool counts, never average the two pixel-weighted IoUs."""
    if set(reports)!={'wrist_cam','top_cam'}:raise ValueError('Both known cameras are required')
    total=MetricAccumulator();by_camera={}
    for camera,report in reports.items():
        if report['camera_filter']!=camera or set(report['by_camera'])!={camera}:raise ValueError('Model/view routing mismatch')
        scores=report['overall'];by_camera[camera]=scores
        total.matrix+=np.asarray(scores['confusion']);total.frames+=scores['frames']
        for cls in (1,2):
            absent=scores['absent_target_frames'][NAMES[cls]]
            total.absent[cls]+=absent['frames'];total.absent_fp[cls]+=absent['with_any_false_positive_pixels']
            total.absent_fp_pixels[cls]+=absent['false_positive_pixels']
            total.absent_total_pixels[cls]+=absent['frames']*image_pixels
    return {'overall':total.report(),'by_camera':by_camera,'camera_macro_iou':camera_macro_iou(by_camera)}


def summary_entry(name,report,parameter_count):
    return {'strategy':name,'global_foreground_iou':report['overall']['foreground_macro_iou'],
            'camera_macro_iou':camera_macro_iou(report['by_camera']),
            'stored_parameters':parameter_count,
            **{f'{view}_{cls}_{metric}':report['by_camera'][camera]['classes'][cls][metric]
               for view,camera in (('wrist','wrist_cam'),('top','top_cam'))
               for cls in ('capsule','fixture') for metric in ('iou','precision','recall')},
            'wrist_fixture_absent':report['by_camera']['wrist_cam']['absent_target_frames']['fixture']}


def comparison_sheet(root,shared_report,specialist_reports,output):
    shared_dir=root/'shared'/'validation'
    shared=Image.open(shared_dir/'predictions.png');ids=shared_report['preview_sample_ids']
    width=shared.width//3;height=shared.height//len(ids)
    lookup={}
    for camera,report in specialist_reports.items():
        image=Image.open(root/camera/'validation'/'predictions.png')
        rows=report['preview_sample_ids'];row_height=image.height//len(rows)
        if row_height!=height:raise ValueError('Preview dimensions disagree')
        for i,sid in enumerate(rows):lookup[sid]=image.crop((width*2,i*height+24,width*3,(i+1)*height))
    sheet=Image.new('RGB',(width*4,height*len(ids)),(24,24,24))
    for i,sid in enumerate(ids):
        y=i*height
        sheet.paste(shared.crop((0,y+24,width*3,y+height)),(0,y+24))
        sheet.paste(lookup[sid],(width*3,y+24))
        ImageDraw.Draw(sheet).text((4,y+5),f'{sid}: RGB | truth | shared | camera specialist',fill='white')
    sheet.save(output)


def run_specialization(root,output,epochs=30,batch_size=8,seed=17,workers=2,device='cuda',
                       limit_train=None,limit_val=None):
    out=Path(output)
    if out.exists():raise FileExistsError(f'Refusing to overwrite experiment: {out}')
    out.mkdir(parents=True)
    spec={'experiment_id':EXPERIMENT_ID,'seed':seed,'epochs':epochs,'batch_size':batch_size,
          'objective':'legacy','workers':workers,'device':device,
          'model':'same TinyUNet v1, width 16, RGB-D, full image resolution',
          'class_weights':'identical all-training-view weights for all three models',
          'selection':'shared: equal four-camera/class IoUs; specialists: own-view foreground IoU',
          'routing':'known camera identity, never a target pose or prediction quality',
          'budget':'same aggregate image presentations/optimizer steps; two specialist weight sets versus one shared weight set',
          'early_stopping':'disabled','evaluation':'validation only',
          'dataset_metadata_sha256':sha(Path(root)/'dataset.json'),'manifest_sha256':sha(Path(root)/'manifest.jsonl'),
          'development_smoke':limit_train is not None or limit_val is not None}
    write_json(out/'experiment_spec.json',spec)
    configs={};finished={};reports={}
    for name,camera in (('shared',None),('wrist_cam','wrist_cam'),('top_cam','top_cam')):
        # In a smoke run, the shared subset must contain twice as many images
        # so its combined presentation count matches the two camera subsets.
        train_limit=limit_train*2 if limit_train is not None and camera is None else limit_train
        val_limit=limit_val*2 if limit_val is not None and camera is None else limit_val
        train(root,out/name,epochs=epochs,batch_size=batch_size,seed=seed,workers=workers,device=device,
              objective='legacy',camera=camera,selection='camera_macro_iou' if camera is None else 'view_foreground_iou',
              patience=epochs+1,limit_train=train_limit,limit_val=val_limit)
        configs[name]=json.loads((out/name/'run_config.json').read_text())
        finished[name]=json.loads((out/name/'training_complete.json').read_text())
        reports[name]=evaluate(root,out/name/'best.pt',out/name/'validation',split='validation',
                               device=device,batch_size=batch_size,workers=workers)
    cfg=configs['shared'];specialists={camera:reports[camera] for camera in ('wrist_cam','top_cam')}
    meta=json.loads((Path(root)/'dataset.json').read_text())
    merged=merge_specialist_reports(specialists,meta['width']*meta['height'])
    shared_presentations=cfg['train_samples']*finished['shared']['epochs_run']
    specialist_presentations=sum(configs[c]['train_samples']*finished[c]['epochs_run'] for c in specialists)
    controls={'same_initial_weights':len({c['initial_state_sha256'] for c in configs.values()})==1,
              'same_class_weights':all(c['class_weights']==cfg['class_weights'] for c in configs.values()),
              'same_source':all(c['source_sha256']==cfg['source_sha256'] for c in configs.values()),
              'same_dataset':all(c['dataset_metadata_sha256']==cfg['dataset_metadata_sha256'] for c in configs.values()),
              'same_manifest':all(c['manifest_sha256']==cfg['manifest_sha256'] for c in configs.values()),
              'same_training_scenes':all(c['train_scenes']==cfg['train_scenes'] for c in configs.values()),
              'same_validation_scenes':all(c['validation_scenes']==cfg['validation_scenes'] for c in configs.values()),
              'same_epoch_budget':all(c['epochs_run']==epochs for c in finished.values()),
              'same_total_image_presentations':shared_presentations==specialist_presentations,
              'same_total_optimizer_steps':math.ceil(cfg['train_samples']/batch_size)*epochs==sum(math.ceil(configs[c]['train_samples']/batch_size)*epochs for c in specialists),
              'correct_fixed_camera_routing':all(configs[c]['camera_filter']==c for c in specialists)}
    if not all(controls.values()):raise RuntimeError(f'View comparison controls failed: {controls}')
    summary={'experiment_id':EXPERIMENT_ID,'controls':controls,'validation_comparison':[
                summary_entry('shared',reports['shared'],cfg['parameters']),
                summary_entry('camera_specialists',merged,sum(configs[c]['parameters'] for c in specialists))],
             'shared_image_presentations':shared_presentations,'specialist_image_presentations':specialist_presentations,
             'specialist_selected_epochs':{c:finished[c]['selected_epoch'] for c in specialists},
             'shared_selected_epoch':finished['shared']['selected_epoch'],
             'development_smoke':spec['development_smoke'],
             'limits':['two specialist parameter sets versus one shared set; not a parameter-matched causal proof',
                       'each specialist receives half the joint model updates; aggregate update/image budget is matched',
                       'augmentation distribution/seed matched, not exact per-image transforms or sample ordering',
                       'independent view checkpoint selection has more selection freedom than joint selection',
                       'validation dependent and one training seed; not a final paper result or robot task result']}
    write_json(out/'specialist_validation.json',merged)
    comparison_sheet(out,reports['shared'],specialists,out/'validation_comparison.png')
    write_json(out/'comparison.json',summary);write_json(out/'experiment_complete.json',{'complete':True,'experiment_id':EXPERIMENT_ID})
    return summary
