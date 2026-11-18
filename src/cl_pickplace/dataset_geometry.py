"""Camera conventions and independent analytic geometry checks for dataset labels."""
import numpy as np


def intrinsics(width, height, fovy_deg):
    focal=.5*height/np.tan(np.radians(fovy_deg)/2)
    # Integer pixel indices refer to centers; OpenGL's half-pixel correction is here.
    return np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])


def transform(rotation, position):
    T=np.eye(4)
    T[:3,:3]=rotation
    T[:3,3]=position
    return T


def camera_transform(mujoco_rotation, position):
    # OpenCV: +x right, +y down, +z forward. MuJoCo camera: +y up, looks along -z.
    return transform(np.asarray(mujoco_rotation)@np.diag([1,-1,-1]),position)


def unproject(depth, mask, K, world_from_camera):
    v,u=np.nonzero(mask)
    z=depth[v,u]
    pts=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
    return pts@world_from_camera[:3,:3].T+world_from_camera[:3,3]


def project(points, K, world_from_camera):
    p=(np.asarray(points)-world_from_camera[:3,3])@world_from_camera[:3,:3]
    return np.column_stack((K[0,0]*p[:,0]/p[:,2]+K[0,2],
                            K[1,1]*p[:,1]/p[:,2]+K[1,2])),p[:,2]


def capsule_surface_distance(points, world_from_geom, radius, half_length):
    p=(np.asarray(points)-world_from_geom[:3,3])@world_from_geom[:3,:3]
    nearest=np.zeros_like(p)
    nearest[:,2]=np.clip(p[:,2],-half_length,half_length)
    return np.linalg.norm(p-nearest,axis=1)-radius


def box_surface_distance(points, world_from_geom, half_size):
    p=(np.asarray(points)-world_from_geom[:3,3])@world_from_geom[:3,:3]
    q=np.abs(p)-half_size
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
