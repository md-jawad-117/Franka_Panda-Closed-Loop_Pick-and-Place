import numpy as np
import pytest

from cl_pickplace.dataset_geometry import (intrinsics,camera_transform,unproject,project,
                                         capsule_surface_distance,box_surface_distance)
from cl_pickplace.dataset_scene import split_scenes,scene_recipe,build_scene
from cl_pickplace.dataset_capture import DatasetCapture,depth_observation


def test_camera_axis_and_pixel_center_conventions_round_trip():
    K=intrinsics(320,240,60)
    T=camera_transform(np.eye(3),np.array([.2,.3,.4]))
    depth=np.ones((240,320),np.float32)
    mask=np.zeros(depth.shape,bool)
    mask[40,70]=True
    p=unproject(depth,mask,K,T)
    uv,z=project(p,K,T)
    np.testing.assert_allclose(uv,[[70,40]],atol=1e-10)
    np.testing.assert_allclose(z,[1])
    assert p[0,2]<.4  # MuJoCo camera looks along -z.


def test_analytic_capsule_and_box_surfaces():
    T=np.eye(4)
    np.testing.assert_allclose(capsule_surface_distance(np.array([[.008,0,0],[0,0,.023]]),T,.008,.015),0,atol=1e-10)
    assert capsule_surface_distance(np.array([[0,0,0]]),T,.008,.015)[0]<0
    np.testing.assert_allclose(box_surface_distance(np.array([[.03,0,0],[.03,.02,.01]]),T,[.03,.02,.01]),0)


def test_split_by_scene_and_recipe_determinism(cfg):
    splits=split_scenes(25,43)
    assert [len(splits[k]) for k in ('train','validation','test')]==[20,3,2]
    assert sorted(sum(splits.values(),[]))==list(range(25))
    assert scene_recipe(cfg,3,43,'all')==scene_recipe(cfg,3,43,'all')
    assert scene_recipe(cfg,3,43,'all')!=scene_recipe(cfg,4,43,'all')


def test_depth_corruption_keeps_labels_separate():
    clean=np.ones((20,20),np.float32)
    a=clean.copy()
    observed,valid=depth_observation(clean,np.random.default_rng(0),{'sigma_m':.002,'dropout':.2,'max_range_m':2.5})
    np.testing.assert_array_equal(clean,a)
    assert (observed[valid==0]==0).all()
    assert 0<int(valid.sum())<400
    assert not np.array_equal(observed,clean)


def test_visible_labels_are_subsets_of_isolated_geometry(cfg):
    recipe=scene_recipe(cfg,2,43,'none')
    s=build_scene(recipe)
    s.reset(np.random.default_rng(recipe['scene_seed']))
    groups=s.model.geom_group.copy()
    capture=DatasetCapture(s)
    try:
        rgb,observed,valid,mask,depth,seg,amodal,ann=capture.capture('top_cam',np.random.default_rng(0),recipe['noise'])
        assert rgb.shape==(240,320,3)
        for cls in (1,2):
            assert ((mask==cls)&~amodal[:,:,cls-1].astype(bool)).sum()==0
            assert (mask==cls).any()
        np.testing.assert_array_equal(s.model.geom_group,groups)
        np.testing.assert_allclose(np.asarray(ann['world_from_camera'])@np.asarray(ann['camera_from_object']),
                                   ann['world_from_object'],atol=1e-8)
    finally:
        capture.close()
