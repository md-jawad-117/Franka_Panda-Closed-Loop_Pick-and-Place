"""Replayable scene recipes. Appearance/geometry are independent of label colors."""
import copy
import mujoco
import numpy as np

from .sim import Sim,build_spec


def split_scenes(count, seed):
    if count<3:
        raise ValueError('At least three scenes are required for disjoint splits')
    ntrain=max(1,int(count*.8))
    nval=max(1,int(count*.12))
    if ntrain+nval>=count: ntrain=count-nval-1
    order=np.random.default_rng(seed).permutation(count).tolist()
    return {'train':order[:ntrain], 'validation':order[ntrain:ntrain+nval],
            'test':order[ntrain+nval:]}


def scene_recipe(base_cfg, scene_index, dataset_seed, perturb):
    scene_seed=int(dataset_seed)*100000+int(scene_index)
    geometry,appearance,noise=[np.random.default_rng(s) for s in np.random.SeedSequence(scene_seed).spawn(3)]
    cfg=copy.deepcopy(base_cfg)
    cap=cfg['scene']['capsule']
    cap['radius']=float(geometry.uniform(.007,.009))
    cap['half_length']=float(geometry.uniform(.013,.016))
    cap['mass']=float(geometry.uniform(.004,.006))
    cfg['scene']['slot']['cavity']=[float(geometry.uniform(.060,.068)),float(geometry.uniform(.033,.039))]
    cfg['task']['place_height_offset']=cfg['scene']['slot']['wall_height']+.003+cap['radius']
    cfg['perception']['noise']['enabled']=False
    cfg['sim']['wrist_cam']['fovy']=float(geometry.uniform(56,64))
    cfg['sim']['wrist_cam']['tilt_deg']=float(geometry.uniform(25,31))
    for camera in cfg['sim']['cameras'].values():
        camera['pos']=(np.asarray(camera['pos'])+geometry.uniform(-.015,.015,3)).tolist()
        camera['fovy']=float(camera['fovy']+geometry.uniform(-2,2))
    colors={name:appearance.uniform(.15,.95,3).tolist()+[1.] for name in ('capsule','slot','table','pedestal')}
    # Include same-color but different-shape hard negatives. Never add indistinguishable
    # capsule copies whose arbitrary simulator names would make identification impossible.
    props=[]
    for i in range(4):
        kind=('box','sphere','cylinder','box')[i]
        size=appearance.uniform(.009,.024,3)
        if kind=='sphere': size=np.array([size[0],0,0])
        if kind=='cylinder': size=np.array([size[0],size[1],0])
        z=size[0] if kind=='sphere' else size[1] if kind=='cylinder' else size[2]
        x=float(appearance.uniform(.67,.76))
        y=float(appearance.uniform(-.35,.35))
        props.append({'kind':kind,'size':size.tolist(),'pos':[x,y,cfg['scene']['table_top_z']+z],
                      'rgba':colors['capsule'] if i%2==0 else colors['slot']})
    return {'scene_index':scene_index,'scene_seed':scene_seed,'config':cfg,'perturb':perturb,
            'appearance':{'colors':colors,'texture_rgb1':appearance.uniform(.3,.9,3).tolist(),
                          'texture_rgb2':appearance.uniform(.1,.6,3).tolist(),
                          'texture_repeat':appearance.uniform(2,7,2).tolist(),
                          'light_scale':float(appearance.uniform(.65,1.25)),
                          'shadow':bool(appearance.integers(0,2)), 'props':props},
            'noise':{'sigma_m':float(noise.uniform(.0005,.002)),
                     'dropout':float(noise.uniform(0,.03)),'max_range_m':2.5}}


def build_scene(recipe):
    cfg=recipe['config']
    spec=build_spec(cfg)
    app=recipe['appearance']
    for name,rgba in app['colors'].items():
        spec.material(name).rgba=rgba
        spec.material(name).reflectance=0
    spec.add_texture(name='dataset_table',type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,width=128,height=128,
                     rgb1=app['texture_rgb1'],rgb2=app['texture_rgb2'])
    material=spec.material('table')
    textures=list(material.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]='dataset_table'
    material.textures=textures
    material.texrepeat=app['texture_repeat']
    for light in spec.lights:
        light.diffuse=(np.asarray(light.diffuse)*app['light_scale']).tolist()
    for i,prop in enumerate(app['props']):
        spec.worldbody.add_geom(name=f'dataset_prop_{i}',type=getattr(mujoco.mjtGeom,'mjGEOM_'+prop['kind'].upper()),
                                size=prop['size'],pos=prop['pos'],rgba=prop['rgba'],
                                contype=0,conaffinity=0)
    return Sim(cfg,spec=spec)
