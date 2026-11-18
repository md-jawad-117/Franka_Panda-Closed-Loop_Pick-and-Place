import numpy as np
import pytest

from cl_pickplace.perception import Perception, CameraPerception
from cl_pickplace.sim import Sim
from cl_pickplace.sensors import RGBDSensor


def test_camera_path_works_without_target_pose_getters(cfg,monkeypatch):
    sim=Sim(cfg)
    sim.reset(np.random.default_rng(3))
    def forbidden(*args):
        raise AssertionError('Camera inference attempted a privileged target pose read')
    for name in ('capsule_pos','capsule_yaw','slot_pos','slot_yaw'):
        monkeypatch.setattr(sim,name,forbidden)
    p=Perception(sim,cfg,np.random.default_rng(0),'camera')
    try:
        est=p.update()
        assert est.capsule.visible and est.slot.visible
        assert est.capsule.source in ('wrist_cam','top_cam')
        assert not hasattr(p._camera,'sim')
    finally:
        p.close()


def test_estimator_can_run_from_replayed_frames_without_simulator(cfg):
    sim=Sim(cfg)
    sim.reset(np.random.default_rng(3))
    sensor=RGBDSensor(sim,320,240)
    try:
        frames={name:sensor.capture(name) for name in ('wrist_cam','top_cam')}
    finally:
        sensor.close()
    class Replay:
        def capture(self,name): return frames[name]
        def close(self): pass
    estimator=CameraPerception(Replay(),cfg['perception'],cfg['scene'])
    capsule,slot=estimator.measure()
    assert capsule is not None and slot is not None


def test_unknown_perception_mode_cannot_fall_back_to_oracle(cfg):
    with pytest.raises(ValueError,match='Unknown perception mode'):
        Perception(Sim(cfg),cfg,np.random.default_rng(0),'camrea')


def test_camera_control_tick_uses_no_target_pose_getters(cfg,monkeypatch):
    from cl_pickplace.episode import Episode
    ep=Episode(cfg,perception_mode='camera')
    ep.reset(3)
    # Diagnostics are a separate privileged sink; omit them to audit only policy input.
    monkeypatch.setattr(ep,'_log',lambda obs: None)
    def forbidden(*args):
        raise AssertionError('Controller read a privileged target pose')
    for name in ('capsule_pos','capsule_yaw','slot_pos','slot_yaw'):
        monkeypatch.setattr(ep.sim,name,forbidden)
    try:
        ep.control_tick()
        assert ep.cmd.p_des is not None
    finally:
        ep.perception.close()
