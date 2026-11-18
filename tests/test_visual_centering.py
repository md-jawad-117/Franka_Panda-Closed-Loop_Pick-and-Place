from pathlib import Path
import json,numpy as np
from cl_pickplace.visual_centering import plan_centering
from test_fixture_exit import sample

ROOT=Path(__file__).resolve().parents[1]


def settings():return json.loads((ROOT/'config/centering_assay_protocol_v19.json').read_text())['settings']


def test_inward_pusher_goal_accounts_for_capsule_and_tool_extent_without_mutation():
    cfg,_,obs=sample();before=obs.est.capsule.pos.copy();s=settings()
    result=plan_centering(obs,cfg['task'],s);p=result['selected'];assert p is not None
    n=np.array([np.cos(p['yaw_rad']),np.sin(p['yaw_rad'])])
    contact=.015*abs(obs.est.capsule.axis[:2]@n)+.008+.0085
    np.testing.assert_allclose(np.array(p['finish'])[:2]+contact*n,obs.est.slot.pos[:2],atol=1e-12)
    np.testing.assert_equal(obs.est.capsule.pos,before)


def test_already_centred_or_stale_pair_does_not_start_an_unnecessary_push():
    cfg,_,obs=sample();obs.est.capsule.pos[:2]=obs.est.slot.pos[:2]
    assert plan_centering(obs,cfg['task'],settings())['selected'] is None
    cfg,_,obs=sample();obs.t=1.3
    assert plan_centering(obs,cfg['task'],settings())['selected'] is None


def test_push_abstains_when_orientation_cannot_fit_even_at_the_centre():
    cfg,_,obs=sample();obs.est.capsule.axis=np.array([0.,1.,0.])
    assert plan_centering(obs,cfg['task'],settings())['reason']=='orientation_cannot_fit_centred'
