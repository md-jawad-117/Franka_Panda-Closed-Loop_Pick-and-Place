"""Line-of-sight test between a camera and a detected capsule against the hand CAD.

Inputs are calibration (camera centre), proprioception (TCP pose) and a fixed CAD
point envelope of the closed hand and fingers in the TCP frame. No object truth.
"""
import numpy as np


def capsule_samples(center,axis,half_segment,count=5):
    axis=np.asarray(axis,float);axis=axis/max(np.linalg.norm(axis),1e-9)
    return np.asarray(center,float)+np.outer(np.linspace(-half_segment,half_segment,count),axis)


def line_of_sight_clearance(camera,targets,points):
    """Minimum distance (m) from any envelope point to camera->target segments."""
    camera=np.asarray(camera,float);best=np.inf
    for q in np.atleast_2d(targets):
        v=q-camera;L2=float(v@v)
        s=np.clip(((points-camera)@v)/max(L2,1e-12),0.,1.)
        best=min(best,float(np.min(np.linalg.norm(points-(camera+np.outer(s,v)),axis=1))))
    return best


def hand_points_world(points_tcp,ee_pos,ee_mat):
    return np.asarray(ee_pos,float)+np.asarray(points_tcp,float)@np.asarray(ee_mat,float).T
