"""Observed-surface fixture estimator; no labels, simulator or table-height input.

An interior depth-height mode supplies an observed floor hypothesis. Reject mask
points below that surface before footprint fitting, reducing table-depth leakage
at learned mask edges. This is a heuristic quality gate, not a confidence bound.
The frozen v1 estimator and trained segmentation models remain separate.
"""
import numpy as np
from .geometry import largest_component,erode_cross,backproject,fixture_from_points


SETTINGS={'version':'observed_fixture_surface_v2','seed_erosions':2,'min_seed_points':12,
          'seed_height_band_m':.003,'below_floor_tolerance_m':.0015,'above_rim_tolerance_m':.002}


def estimate_fixture(mask,depth,valid,calibration,protocol,settings=None):
    settings=SETTINGS if settings is None else settings
    def reject(reason,quality=None):
        return {'accepted':False,'reasons':[reason],'position_m':None,'axis_world':None,'quality':quality or {}}
    mask=np.asarray(mask,dtype=bool);component=largest_component(mask);count=int(component.sum())
    if count<protocol['min_component_pixels']:return reject('too_few_mask_pixels')
    fraction=count/max(int(mask.sum()),1)
    if fraction<protocol['min_component_fraction']:return reject('fragmented_mask')
    if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():
        return reject('cropped_at_image_border')
    depth_valid=(np.asarray(valid)>0)&np.isfinite(depth)&(depth>0)&(depth<=protocol['max_depth_m'])
    seed=component.copy()
    for _ in range(settings['seed_erosions']):seed=erode_cross(seed)
    seed_points=backproject(depth,depth_valid,seed,calibration)
    if len(seed_points)<settings['min_seed_points']:return reject('insufficient_interior_surface_evidence')
    z=np.sort(seed_points[:,2]);ends=np.searchsorted(z,z+settings['seed_height_band_m'],side='right')
    start=int(np.argmax(ends-np.arange(len(z))));band=z[start:ends[start]]
    if len(band)<settings['min_seed_points']:return reject('interior_surface_mode_not_supported')
    floor=float(np.median(band))
    points=backproject(depth,depth_valid,component,calibration)
    if len(points)<protocol['min_depth_points']:return reject('too_few_interior_depth_points')
    compact=np.linalg.norm(points-np.median(points,axis=0),axis=1)<.10
    if np.mean(compact)<.9:return reject('depth_component_not_compact')
    points=points[compact];total=len(points)
    height=protocol['nominal_dimensions_m']['fixture_wall_height']
    keep=(points[:,2]>=floor-settings['below_floor_tolerance_m'])&(points[:,2]<=floor+height+settings['above_rim_tolerance_m'])
    points=points[keep]
    quality={'estimator':settings['version'],'observed_seed_floor_m':floor,'seed_mode_points':len(band),
             'seed_mode_fraction':len(band)/len(seed_points),'surface_kept_fraction':len(points)/max(total,1)}
    if len(points)<protocol['min_depth_points']:return reject('too_few_surface_consistent_points',quality)
    n=protocol['max_fit_points']
    if len(points)>n:points=points[np.linspace(0,len(points)-1,n,dtype=int)]
    result=fixture_from_points(points,protocol)
    result['quality'].update(quality)
    result['quality'].update(component_pixels=count,largest_component_fraction=fraction,fit_points=len(points))
    return result
