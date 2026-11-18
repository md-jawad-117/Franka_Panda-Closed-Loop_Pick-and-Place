"""Choose a wrist FOV from a fixed hand-frame grasp envelope, never target state."""
import copy,itertools
import numpy as np


def design(cfg,settings):
    camera=cfg['sim']['wrist_cam'];capsule=cfg['scene']['capsule']
    position=np.asarray(camera['pos'],float);angle=np.radians(camera['tilt_deg'])
    c,s=np.cos(angle),np.sin(angle)
    # OpenCV camera axes in the hand frame, matching the build_spec mounting.
    rotation=np.array([[c,0,-s],[0,1,0],[s,0,c]])
    radius=float(capsule['radius']);half=float(capsule['half_length'])
    allowance=float(settings['transverse_center_allowance_m'])
    transverse=radius+half*np.sin(np.radians(settings['axis_deviation_deg']))
    extent_x=half+radius+allowance;extent_y=transverse+allowance
    tcp=float(cfg['sim']['ee_site_offset']);depths=settings['center_depth_bounds_m']
    bounds=[[-extent_x,extent_x],[-extent_y,extent_y],[tcp+depths[0]-transverse,tcp+depths[1]+transverse]]
    corners=np.asarray(list(itertools.product(*bounds)));local=(corners-position)@rotation
    if not np.isfinite(local).all() or np.any(local[:,2]<=0):raise ValueError('Declared grasp envelope crosses camera plane')
    width,height=cfg['perception']['camera']['width'],cfg['perception']['camera']['height']
    usable=np.array([(width-1)/2,(height-1)/2])-settings['border_pixels']
    if np.any(usable<=0):raise ValueError('Invalid optical border')
    # focal = height/2 / tan(fovy/2); fit every enclosing-box corner.
    normalized=np.max(np.abs(local[:,:2]/local[:,2:]),axis=0)
    required=float(np.degrees(2*np.arctan(np.max(normalized*(height/2)/usable))))
    step=float(settings['round_fovy_step_deg'])
    fovy=max(float(camera['fovy']),float(np.ceil(required/step)*step))
    if not 0<fovy<settings['maximum_fovy_deg']:raise ValueError('Required FOV outside declared lens envelope')
    return {'fovy_deg':fovy,'minimum_box_fovy_deg':required,'hand_frame_box_m':bounds,
            'scope':'Enclosing nominal grasp-envelope corners; no claim about occlusion, full arbitrary motion or segmentation accuracy.'}


def widen_wrist(cfg,settings):
    out=copy.deepcopy(cfg);out['sim']['wrist_cam']['fovy']=design(cfg,settings)['fovy_deg']
    return out
