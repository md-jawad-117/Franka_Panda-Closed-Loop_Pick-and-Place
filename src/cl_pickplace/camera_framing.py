"""Static CAD-derived wrist-camera aiming; no observed or true target input."""
import copy
import numpy as np


def aim_wrist_at_grasp(cfg,nominal_depth_m):
    """Aim at [0,0,TCP offset + nominal depth] in the robot hand frame.

    This uses the fixed mounting convention in build_spec. It changes only camera
    extrinsics, not control/physics or perception acceptance thresholds.
    """
    out=copy.deepcopy(cfg);camera=out['sim']['wrist_cam']
    position=np.asarray(camera['pos'],float)
    depth=float(nominal_depth_m);tcp=float(out['sim']['ee_site_offset'])
    if position.shape!=(3,) or not np.isfinite(position).all() or not np.isfinite(depth):
        raise ValueError('Nonfinite or malformed nominal camera geometry')
    if abs(position[1])>1e-12 or not 0<=depth<=.02:
        raise ValueError('Only the declared sagittal mount and bounded grasp depth are supported')
    dz=tcp+depth-position[2]
    if dz<=0:raise ValueError('Camera must be behind nominal target along the hand axis')
    camera['tilt_deg']=float(np.degrees(np.arctan2(position[0],dz)))
    return out
