"""Remove learned capsule-mask pixels on the observed support plane before fitting.

The plane is estimated from synchronized fixture observations, never table truth.
The fixed-radius capsule has a visible raised surface; its nominal lower points
may be removed. The unchanged surface fitter must still pass every original gate.
"""
import numpy as np
from .geometry import estimate_pose

SETTINGS={'version':'capsule_raised_surface_fallback_v4','minimum_surface_above_floor_m':.0025,
          'center_floor_slack_m':.007,'fallback_only':True}


def estimate_capsule(mask,depth,valid,calibration,protocol,observed_floor,settings=SETTINGS):
    mask=np.asarray(mask,dtype=bool)
    filtered=mask.copy()
    if observed_floor is not None:
        if not np.isfinite(observed_floor):raise ValueError('Observed floor must be finite')
        v,u=np.nonzero(mask)
        z=np.asarray(depth)[v,u]
        rays=np.column_stack((u,v,np.ones(len(u))))@np.linalg.inv(calibration.K).T
        camera=rays*z[:,None]
        transform=calibration.world_from_camera
        world_z=camera@transform[2,:3]+transform[2,3]
        keep=np.isfinite(world_z)&(world_z>=observed_floor+settings['minimum_surface_above_floor_m'])
        filtered[v[~keep],u[~keep]]=False
    result=estimate_pose(filtered,depth,valid,calibration,'capsule',protocol)
    result['quality'].update({'capsule_estimator':settings['version'],'observed_fixture_floor_m':observed_floor,
                             'surface_mask_kept_fraction':float(filtered.sum()/max(1,mask.sum()))})
    return result


def estimate_capsule_with_fallback(mask,depth,valid,calibration,protocol,observed_floor,settings=SETTINGS):
    original=estimate_pose(mask,depth,valid,calibration,'capsule',protocol)
    below=(observed_floor is not None and original['accepted'] and
           original['position_m'][2]<observed_floor+protocol['nominal_dimensions_m']['capsule_radius']-settings['center_floor_slack_m'])
    if observed_floor is None or (original['accepted'] and not below):
        original['quality'].update({'capsule_estimator':settings['version'],'surface_fallback_used':False,
                                    'observed_fixture_floor_m':observed_floor})
        return original
    result=estimate_capsule(mask,depth,valid,calibration,protocol,observed_floor,settings)
    result['quality'].update(surface_fallback_used=True,original_reasons=original['reasons'],
                             original_below_support_region=bool(below))
    return result
