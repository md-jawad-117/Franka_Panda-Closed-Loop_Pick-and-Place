from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training'))
from pose_evaluation.observed_support_v17 import observed_planes,estimate_fixture
from pose_evaluation.geometry import Calibration


def scene(table_height):
    K=np.array([[400.,0,159.5],[0,400.,119.5],[0,0,1.]])
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    calibration=Calibration(K,T);depth=np.full((240,320),1-table_height)
    mask=np.zeros(depth.shape,bool);distance=1-table_height-.003
    # Nominal 64 x 36 mm cavity from the frozen pose protocol.
    half_x=int(np.floor(.032*400/distance));half_y=int(np.floor(.018*400/distance))
    mask[120-half_y:120+half_y+1,160-half_x:160+half_x+1]=True;depth[mask]=distance
    return depth,np.ones(depth.shape,np.uint8),mask,calibration


def test_floor_only_fixture_uses_measured_support_at_different_world_heights():
    protocol=json.loads((ROOT/'training/pose_protocol_v1.json').read_text())
    for height in (.35,.4,.45):
        depth,valid,mask,calibration=scene(height);planes=observed_planes(depth,valid,~mask,calibration)
        result=estimate_fixture(mask,depth,valid,calibration,protocol,planes)
        assert result['accepted'],result
        assert abs(result['position_m'][2]-(height+.003))<1e-10


def test_no_plane_or_incompatible_plane_cannot_silently_become_a_target():
    depth,valid,mask,calibration=scene(.4);protocol=json.loads((ROOT/'training/pose_protocol_v1.json').read_text())
    assert not estimate_fixture(mask,depth,valid,calibration,protocol,[])['accepted']
    bad=[{'height_m':0.,'points':1000,'fraction':.7}]
    assert not estimate_fixture(mask,depth,valid,calibration,protocol,bad)['accepted']


def test_plane_evidence_requires_spatial_extent_and_does_not_use_fixture_pixels():
    depth,valid,mask,calibration=scene(.4)
    tiny=np.zeros(mask.shape,bool);tiny[100:130,130:170]=True
    assert not observed_planes(depth,valid,tiny,calibration)
    assert not observed_planes(depth,np.zeros_like(valid),~mask,calibration)
