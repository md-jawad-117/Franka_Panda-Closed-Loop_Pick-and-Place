"""Observed horizontal support planes resolve floor/rim ambiguity.

NumPy, predicted masks, noisy depth and calibration only. There is no simulator,
true table height, label, recipe or object-state input. Known bottom thickness is
a declared CAD prior; missing/incompatible support evidence remains rejected.
"""
import numpy as np
from .geometry import largest_component,backproject,fixture_from_points

SETTINGS={'version':'observed_support_fixture_v17','support_band_m':.003,'support_max_points':5000,
    'support_min_points':200,'support_min_fraction':.12,'support_min_xy_span_m':.12,'support_modes':3,
    'surface_tolerance_m':.0015,'above_rim_tolerance_m':.002,'minimum_surface_fraction':.35,
    'minimum_surface_points':12,'below_floor_tolerance_m':.001}


def observed_planes(depth,valid,background,calibration,settings=SETTINGS):
    points=backproject(depth,valid,background,calibration)
    if len(points)>settings['support_max_points']:points=points[np.linspace(0,len(points)-1,settings['support_max_points'],dtype=int)]
    total=len(points);planes=[]
    for _ in range(settings['support_modes']):
        if len(points)<settings['support_min_points']:break
        order=np.argsort(points[:,2],kind='stable');points=points[order];z=points[:,2]
        ends=np.searchsorted(z,z+settings['support_band_m'],side='right')
        start=int(np.argmax(ends-np.arange(len(z))));end=int(ends[start]);band=points[start:end]
        height=float(np.median(band[:,2]));span=np.quantile(band[:,:2],.95,axis=0)-np.quantile(band[:,:2],.05,axis=0)
        if len(band)>=settings['support_min_points'] and len(band)/max(total,1)>=settings['support_min_fraction'] and np.all(span>=settings['support_min_xy_span_m']):
            planes.append({'height_m':height,'points':len(band),'fraction':len(band)/max(total,1),
                'xy_span_m':span.tolist(),'median_absolute_height_residual_m':float(np.median(abs(band[:,2]-height)))})
        points=points[abs(points[:,2]-height)>settings['support_band_m']]
    return planes


def estimate_fixture(mask,depth,valid,calibration,protocol,planes,settings=SETTINGS):
    def reject(reason,quality=None):return {'accepted':False,'reasons':[reason],'position_m':None,'axis_world':None,'quality':quality or {}}
    component=largest_component(mask);count=int(component.sum());mask=np.asarray(mask,bool)
    if count<protocol['min_component_pixels']:return reject('too_few_mask_pixels')
    fraction=count/max(int(mask.sum()),1)
    if fraction<protocol['min_component_fraction']:return reject('fragmented_mask')
    if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():return reject('cropped_at_image_border')
    points=backproject(depth,valid,component,calibration)
    if len(points)<protocol['min_depth_points']:return reject('too_few_interior_depth_points')
    compact=np.linalg.norm(points-np.median(points,axis=0),axis=1)<.10
    if np.mean(compact)<.9:return reject('depth_component_not_compact')
    points=points[compact];dims=protocol['nominal_dimensions_m'];height=dims['fixture_wall_height'];bottom=dims['fixture_bottom_thickness']
    hypotheses=[]
    for plane in planes:
        floor=plane['height_m']+bottom
        distance=np.minimum(abs(points[:,2]-floor),abs(points[:,2]-floor-height))
        supported=distance<=settings['surface_tolerance_m'];n=int(supported.sum())
        if n>=settings['minimum_surface_points'] and n/len(points)>=settings['minimum_surface_fraction']:
            hypotheses.append((n,floor,plane))
    if not hypotheses:return reject('no_compatible_observed_support',{'support_planes':planes})
    _,floor,plane=max(hypotheses,key=lambda h:(h[0],h[2]['points']))
    # The support plane must account for the visible surface levels, not merely
    # suggest a convenient world height. Table leakage is removed before fitting.
    keep=(points[:,2]>=floor-settings['below_floor_tolerance_m'])&(points[:,2]<=floor+height+settings['above_rim_tolerance_m'])
    selected=points[keep]
    if len(selected)<protocol['min_depth_points']:return reject('too_few_support_consistent_points')
    if len(selected)>protocol['max_fit_points']:selected=selected[np.linspace(0,len(selected)-1,protocol['max_fit_points'],dtype=int)]
    result=fixture_from_points(selected,protocol)
    result['reasons']=[r for r in result['reasons'] if r!='fixture_floor_rim_height_ambiguous']
    result['accepted']=not result['reasons'];result['position_m'][2]=float(floor)
    result['quality'].update({'estimator':settings['version'],'support_plane':plane,'observed_floor_m':floor,
        'support_consistent_points':len(selected),'component_pixels':count,'largest_component_fraction':fraction,
        'floor_height_source':'observed RGB-D horizontal background plane plus declared bottom thickness'})
    return result
