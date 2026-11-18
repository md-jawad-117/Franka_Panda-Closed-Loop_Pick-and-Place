"""The optional geometric worker obeys the existing robot observation boundary."""
from cl_pickplace.config import load_config,ROOT
from cl_pickplace.episode import Episode


def test_surface_physical_control_cannot_read_target_truth(monkeypatch):
    cfg=load_config(ROOT/'config/learned_surface_v2.yaml')
    ep=Episode(cfg,perception_mode='learned',perturb='none')
    try:
        ep.reset(3);monkeypatch.setattr(ep,'_log',lambda obs:None)
        def forbidden(*args):raise AssertionError('Privileged target state entered surface control')
        for name in ('capsule_pos','capsule_yaw','slot_pos','slot_yaw'):monkeypatch.setattr(ep.sim,name,forbidden)
        for _ in range(350):ep.control_tick();ep.physics_period()
        assert any(b=='TRANSPORT' for _,_,b,_ in ep.fsm.stats.transitions)
        assert ep.perception.learned_summary()['metadata']['fixture_estimator']=='observed_fixture_surface_v2'
    finally:ep.perception.close()
