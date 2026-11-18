"""Audit dataset integrity, independent surface geometry, and saved-state replay."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .dataset_geometry import unproject,capsule_surface_distance,box_surface_distance

SURFACE_TOLERANCE_M=.0005  # 0.5 mm; capsule rasterization uses a tessellated surface.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_file(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('File path escapes dataset root')
    return path


def check_transform(value):
    T=np.asarray(value,dtype=float)
    assert T.shape==(4,4) and np.isfinite(T).all(),'invalid transform'
    assert np.allclose(T[3],[0,0,0,1],atol=1e-9),'invalid homogeneous row'
    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8),'nonorthogonal rotation'
    assert abs(np.linalg.det(T[:3,:3])-1)<1e-8,'improper rotation'
    return T


def check_example(root,row,verify_hashes=True,noise=None):
    sid=row['sample_id']
    expected={f'observations/{sid}/rgb.png',f'observations/{sid}/depth.npz',
              f'labels/{sid}/mask.png',f'labels/{sid}/geometry.npz',f'labels/{sid}/annotation.json'}
    assert set(row['files'])==expected,'unexpected file inventory'
    for name,h in row['files'].items():
        path=relative_file(root,name)
        assert path.is_file(),f'missing {name}'
        if verify_hashes: assert digest(path)==h,f'checksum mismatch: {name}'
    folder=root/'labels'/sid
    a=json.loads((folder/'annotation.json').read_text())
    for key in ('sample_id','scene_id','split','camera','frame_index'):
        assert a[key]==row[key],f'annotation/manifest disagree: {key}'
    shape=(a['height'],a['width'])
    rgb=cv2.imread(str(root/'observations'/sid/'rgb.png'))
    mask=cv2.imread(str(folder/'mask.png'),cv2.IMREAD_UNCHANGED)
    with np.load(root/'observations'/sid/'depth.npz',allow_pickle=False) as d:
        assert set(d.files)=={'depth_m','valid'},'privileged fields in model observation'
        observed=d['depth_m']; valid=d['valid']
    with np.load(folder/'geometry.npz',allow_pickle=False) as d:
        clean=d['ideal_depth_m']; seg=d['raw_segmentation']; amodal=d['amodal_masks']
    assert rgb is not None and rgb.shape==(*shape,3) and rgb.dtype==np.uint8,'invalid RGB'
    assert mask.shape==shape and mask.dtype==np.uint8 and set(np.unique(mask))<={0,1,2},'invalid mask'
    assert observed.shape==shape and observed.dtype==np.float32,'invalid observed depth'
    assert valid.shape==shape and valid.dtype==np.uint8 and set(np.unique(valid))<={0,1},'invalid depth validity'
    assert np.isfinite(observed).all() and np.all(observed[valid==0]==0) and np.all(observed[valid==1]>0),'invalid observed values'
    assert clean.shape==shape and np.isfinite(clean).all() and np.all(clean>0),'invalid ideal depth'
    if noise is not None:
        possible=clean<noise['max_range_m']; retained=valid.astype(bool)
        assert not (retained&~possible).any(),'valid depth outside sensor range'
        n=int(possible.sum()); dropped=int((possible&~retained).sum())
        p=noise['dropout']; sigma=noise['sigma_m']
        assert abs(dropped-n*p)<=6*np.sqrt(n*p*(1-p))+2,'depth dropout inconsistent with noise recipe'
        residual=observed[retained]-clean[retained]
        if len(residual)>500:
            assert abs(float(residual.mean()))<=6*sigma/np.sqrt(len(residual))+1e-6,'observed depth has unexpected bias'
            assert abs(float(residual.std())-sigma)<=.1*sigma,'observed depth noise scale inconsistent with recipe'
    assert seg.shape==(*shape,2) and np.issubdtype(seg.dtype,np.integer),'invalid raw segmentation'
    assert amodal.shape==(*shape,2) and set(np.unique(amodal))<={0,1},'invalid amodal masks'
    K=np.asarray(a['K']); T=check_transform(a['world_from_camera'])
    assert K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0,'invalid intrinsics'
    assert np.allclose(K[2],[0,0,1]),'invalid intrinsic last row'
    for obj in ('object','fixture_floor'):
        world=check_transform(a[f'world_from_{obj}']); camera=check_transform(a[f'camera_from_{obj}'])
        assert np.allclose(T@camera,world,atol=1e-8),'camera/world pose inconsistency'
    expected_mask=np.zeros(shape,np.uint8); residuals={1:[],2:[]}; skipped=0; checked=0
    for g in a['geometries']:
        G=check_transform(g['world_from_geom']); cls=g['class_id']
        assert cls in (1,2),'invalid geometry class'
        m=(seg[:,:,0]==g['geom_id'])&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        expected_mask[m]=cls
        interior=cv2.erode(m.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        if not interior.any():
            skipped+=1
            continue
        p=unproject(clean,interior,K,T); size=np.asarray(g['size'])
        assert g['kind'] in ('capsule','box'),'unsupported geometry'
        e=(capsule_surface_distance(p,G,size[0],size[1]) if g['kind']=='capsule'
           else box_surface_distance(p,G,size))
        e=np.abs(e); checked+=1
        assert e.max()<=SURFACE_TOLERANCE_M,f'{g["kind"]} depth/pose surface mismatch: {e.max():.6f} m'
        residuals[cls].extend(e.tolist())
    assert np.array_equal(mask,expected_mask),'mask disagrees with simulator IDs'
    assert np.allclose(a['world_from_object'],a['geometries'][0]['world_from_geom'],atol=1e-9),'object/geometry pose disagree'
    for cls in (1,2):
        v=mask==cls; full=amodal[:,:,cls-1].astype(bool); info=a['visibility'][str(cls)]
        assert not (v&~full).any(),'visible mask extends outside amodal mask'
        n,f=int(v.sum()),int(full.sum())
        assert (n,f)==(info['visible_pixels'],info['amodal_pixels']),'visibility count mismatch'
        assert info['in_frustum']==bool(f),'frustum flag mismatch'
        assert info['depth_valid_pixels']==int((v&valid.astype(bool)).sum()),'valid depth count mismatch'
        assert info['amodal_touches_border']==bool(full[0].any() or full[-1].any() or full[:,0].any() or full[:,-1].any()),'border flag mismatch'
        assert info['visible_fraction']==(n/f if f else None),'visibility ratio mismatch'
    return a,residuals,checked,skipped


def replay_example(root,row):
    from .dataset_scene import build_scene
    from .dataset_capture import DatasetCapture
    a=json.loads((root/'labels'/row['sample_id']/'annotation.json').read_text())
    recipe=json.loads((root/'scenes'/f'{row["scene_id"]}.json').read_text())
    sim=build_scene(recipe)
    for field in ('qpos','qvel','mocap_pos','mocap_quat'):
        getattr(sim.data,field)[:]=a['state'][field]
    sim.data.time=a['timestamp_s']
    capture=DatasetCapture(sim,a['width'],a['height'],recipe['appearance']['shadow'])
    seed=np.random.SeedSequence([recipe['scene_seed'],row['frame_index'],0 if row['camera']=='wrist_cam' else 1,772])
    try:
        rgb,depth,valid,mask,clean,seg,amodal,b=capture.capture(row['camera'],np.random.default_rng(seed),recipe['noise'])
    finally: capture.close()
    sid=row['sample_id']; folder=root/'labels'/sid
    stored=cv2.cvtColor(cv2.imread(str(root/'observations'/sid/'rgb.png')),cv2.COLOR_BGR2RGB)
    assert np.array_equal(rgb,stored),'replay RGB differs'
    assert np.array_equal(mask,cv2.imread(str(folder/'mask.png'),cv2.IMREAD_UNCHANGED)),'replay mask differs'
    with np.load(folder/'geometry.npz',allow_pickle=False) as d:
        assert np.array_equal(seg,d['raw_segmentation']),'replay simulator IDs differ'
        assert np.array_equal(amodal,d['amodal_masks']),'replay amodal differs'
        assert np.allclose(clean,d['ideal_depth_m'],rtol=0,atol=1e-6),'replay ideal depth differs'
    with np.load(root/'observations'/sid/'depth.npz',allow_pickle=False) as d:
        assert np.allclose(depth,d['depth_m'],rtol=0,atol=1e-6) and np.array_equal(valid,d['valid']),'replay observed depth differs'
    for key in ('K','world_from_camera','world_from_object','world_from_fixture_floor'):
        assert np.allclose(a[key],b[key],atol=1e-9),f'replay {key} differs'
    return sid


def verify_dataset(directory,replay_count=12):
    root=Path(directory); meta=json.loads((root/'dataset.json').read_text())
    rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    errors=[]; residues={1:[],2:[]}; coverage=Counter(); states=Counter(); reasons=Counter()
    def require(condition,message):
        if not condition: errors.append(message)
    require(meta.get('generation_complete') is True,'generation incomplete')
    require(len(rows)==meta['expected_samples']==meta['samples'],'sample count mismatch')
    require(len({r['sample_id'] for r in rows})==len(rows),'duplicate sample IDs')
    scene_splits={}; expected_splits={f'scene_{i:05d}':split for split,ids in meta['splits'].items() for i in ids}
    flattened=[i for ids in meta['splits'].values() for i in ids]
    require(len(flattened)==len(set(flattened))==meta['scene_count'],'scene split overlap/count mismatch')
    require(set(flattened)==set(range(meta['scene_count'])),'scene index inventory mismatch')
    recipes={}; recipe_seeds=[]
    require(set(meta.get('scene_files',{}))=={f'scenes/{s}.json' for s in expected_splits},'scene recipe checksum inventory mismatch')
    for scene in expected_splits:
        try:
            name=f'scenes/{scene}.json'; path=relative_file(root,name)
            require(digest(path)==meta.get('scene_files',{}).get(name),f'scene recipe checksum mismatch: {scene}')
            recipe=json.loads(path.read_text()); recipes[scene]=recipe
            require(scene==f'scene_{recipe["scene_index"]:05d}',f'recipe scene index mismatch: {scene}')
            require(recipe['scene_seed']==meta['dataset_seed']*100000+recipe['scene_index'],f'recipe seed mismatch: {scene}')
            recipe_seeds.append(recipe['scene_seed'])
        except (ValueError,KeyError,OSError,TypeError) as e: errors.append(f'recipe {scene}: {e}')
    require(len(recipe_seeds)==len(set(recipe_seeds)),'repeated recipe seeds across scenes')
    hashes=defaultdict(set); per_scene=defaultdict(list); annotations=[]
    for row in rows:
        sid=row['sample_id']; scene=row['scene_id']; split=row['split']
        if scene in scene_splits: require(scene_splits[scene]==split,f'scene split leakage: {scene}')
        scene_splits[scene]=split
        require(expected_splits.get(scene)==split,f'scene split disagrees with dataset: {scene}')
        per_scene[scene].append((row['frame_index'],row['camera']))
        try:
            a,res,checked,skipped=check_example(root,row,noise=recipes.get(scene,{}).get('noise'))
            require((a['width'],a['height'])==(meta['width'],meta['height']),f'image dimensions disagree with dataset: {sid}')
            require(a['scene_seed']==recipes.get(scene,{}).get('scene_seed'),f'annotation/recipe seed mismatch: {sid}')
            annotations.append((row,a))
            for cls in (1,2):
                info=a['visibility'][str(cls)]
                residues[cls].extend(res[cls]); coverage[f'class_{cls}_visible_frames']+=info['visible_pixels']>0
                coverage[f'class_{cls}_partly_occluded_frames']+=0<info['visible_fraction']<.99 if info['visible_fraction'] is not None else 0
                coverage[f'class_{cls}_fully_occluded_frames']+=info['amodal_pixels']>0 and info['visible_pixels']==0
                coverage[f'class_{cls}_out_of_view_frames']+=not info['in_frustum']
                coverage[f'class_{cls}_border_touching_frames']+=info['amodal_touches_border']
            coverage['checked_geometry_views']+=checked; coverage['skipped_geometry_views_no_interior_pixels']+=skipped
            states[a['actual_state']]+=1; reasons[a['capture_reason']]+=1
            hashes[digest(root/'observations'/sid/'rgb.png')].add(split)
        except (AssertionError,ValueError,KeyError,OSError,TypeError) as e: errors.append(f'{sid}: {e}')
    expected_frames={(i,cam) for i in range(len(meta['stages'])) for cam in ('wrist_cam','top_cam')}
    for scene in expected_splits:
        require(len(per_scene[scene])==len(expected_frames) and set(per_scene[scene])==expected_frames,f'incomplete/duplicate scene frame inventory: {scene}')
        require((root/'scenes'/f'{scene}.json').is_file(),f'missing scene recipe: {scene}')
    require(not any(len(s)>1 for s in hashes.values()),'identical RGB images across splits')
    replayed=[]
    # Cover each split and both cameras, then deterministically sample other scenes/phases.
    candidates=[]
    for split in meta['splits']:
        for cam in ('wrist_cam','top_cam'):
            group=[r for r,a in annotations if r['split']==split and r['camera']==cam]
            if group:
                stage=3 if cam=='wrist_cam' else 6
                candidates.append(next((r for r in group if r['frame_index']==stage),group[0]))
    remaining=[r for r,a in annotations if r not in candidates]
    order=np.random.default_rng(972).permutation(len(remaining))
    candidates += [remaining[i] for i in order]
    for row in candidates[:replay_count]:
        try: replayed.append(replay_example(root,row))
        except (AssertionError,ValueError,KeyError,OSError,TypeError) as e: errors.append(f'replay {row["sample_id"]}: {e}')
    stats={str(cls):{'pixels_checked':len(v),'p95_m':float(np.quantile(v,.95)) if v else None,'max_m':max(v) if v else None} for cls,v in residues.items()}
    for cls,v in residues.items(): require(bool(v),f'no independent surface checks for class {cls}')
    return {'passed':not errors,'errors':errors,'samples':len(rows),'scenes':len(scene_splits),
            'dataset_metadata_sha256':digest(root/'dataset.json'),'manifest_sha256':digest(root/'manifest.jsonl'),
            'split_scene_counts':{k:len(v) for k,v in meta['splits'].items()},
            'surface_tolerance_m':SURFACE_TOLERANCE_M,'surface_residuals':stats,
            'coverage':dict(coverage),'actual_states':dict(states),'capture_reasons':dict(reasons),
            'unique_rgb_images':len(hashes),'replayed_samples':replayed,
            'limits':['geometry checks exclude one-pixel borders and geometry with no interior pixels',
                      'replay equality is checked on this renderer/runtime; cross-hardware RGB equality is not promised',
                      'passing checks establishes label consistency, not real-world realism or model generalization']}


def contact_sheet(directory,output):
    root=Path(directory); rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    chosen=[]
    for split in ('train','validation','test'):
        scene=next(r['scene_id'] for r in rows if r['split']==split)
        for stage,cam in ((0,'top_cam'),(3,'wrist_cam'),(6,'wrist_cam'),(7,'top_cam')):
            chosen.append(next(r for r in rows if r['scene_id']==scene and r['frame_index']==stage and r['camera']==cam))
    panels=[]
    for row in chosen:
        sid=row['sample_id']; a=json.loads((root/'labels'/sid/'annotation.json').read_text())
        rgb=cv2.imread(str(root/'observations'/sid/'rgb.png'))
        mask=cv2.imread(str(root/'labels'/sid/'mask.png'),cv2.IMREAD_UNCHANGED)
        for cls,color in ((1,(255,0,255)),(2,(255,255,0))):
            contours,_=cv2.findContours((mask==cls).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb,contours,-1,color,1)
        with np.load(root/'observations'/sid/'depth.npz') as d:
            depth=d['depth_m']; valid=d['valid']
        scale=np.clip(depth/1.5*255,0,255).astype(np.uint8)
        depth_image=cv2.applyColorMap(255-scale,cv2.COLORMAP_TURBO); depth_image[valid==0]=0
        semantic=np.zeros_like(rgb);semantic[mask==1]=(255,0,255);semantic[mask==2]=(255,255,0)
        panel=np.vstack((np.full((48,rgb.shape[1]*3,3),25,np.uint8),np.hstack((rgb,depth_image,semantic))))
        cv2.putText(panel,f'{row["split"]} {sid}  actual: {a["actual_state"]}',(6,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.putText(panel,'RGB contours | observed depth (black=invalid) | semantic labels',(6,35),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        panels.append(panel)
    sheet=np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(output),sheet)
    return Path(output)
