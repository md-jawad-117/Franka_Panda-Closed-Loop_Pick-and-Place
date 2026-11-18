"""Repeated input, visible end state, and viewer updates during inference."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import threading
import numpy as np

from cl_pickplace.interactive import drain_commands,auto_reset_due


def test_repeat_burst_is_collapsed_but_commands_after_reset_are_preserved():
    keys=Queue()
    for command in ['slot']*100+['push']*4+['reset']+['slot']*10:keys.put(command)
    assert drain_commands(keys)==['slot','push','reset','slot']
    assert keys.empty()


def test_manual_restart_remains_required_even_long_after_timeout():
    for now in (2,10,1000):assert not auto_reset_due(1,now,False,2)
    assert auto_reset_due(1,4,True,2)
    assert not auto_reset_due(None,1000,True,2)


def test_demo_refreshes_outside_lock_and_holds_failed_scene_until_restart(monkeypatch,cfg):
    path=Path(__file__).resolve().parents[1]/'run_demo.py'
    spec=importlib.util.spec_from_file_location('demo_test_under_load',path);demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    lock=threading.Lock();created=[]
    class Clock:
        def __init__(self):self.now=0
        def perf_counter(self):self.now+=.1;return self.now
        def sleep(self,seconds):self.now+=seconds
    class Viewer:
        def __init__(self,callback):
            self.callback=callback;self.syncs=0
            self.cam=SimpleNamespace(lookat=np.zeros(3))
            self.user_scn=SimpleNamespace(ngeom=0)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def lock(self):return lock
        def is_running(self):return self.syncs<20
        def sync(self):
            assert not lock.locked()
            self.syncs+=1
            if self.syncs==1:
                for _ in range(100):self.callback(ord('S'))
    class Perception:
        def __init__(self):self.context=nullcontext;self.refresh=None;self.calls=0;self.closed=False
        def set_live_hooks(self,context,on_wait):self.context=context;self.refresh=on_wait
        def update(self):
            assert not lock.locked();self.calls+=1
            with self.context():assert lock.locked()
            self.refresh();self.refresh()
            return object()
        def close(self):self.closed=True
    class Episode:
        def __init__(self,config,mode=None,perception_mode=None,perturb=None):
            created.append(self);self.mode=mode or 'closed_loop';self.perception_mode=perception_mode or 'learned'
            self.sim=SimpleNamespace(model=object(),data=object());self.perception=Perception();self.t=0
            self.fsm=SimpleNamespace(state=SimpleNamespace(name='TRANSPORT'),guard_action='reobserve_destination')
            self.perturber=SimpleNamespace(rng=np.random.default_rng(0),move_slot=lambda *args:self.moves.append(args))
            self.resets=[];self.moves=[];self.steps=0;self.done=False
        def reset(self,seed,capture=True):self.resets.append(seed);self.done=False;self.est=None
        def step(self,perception_ready=False):
            assert perception_ready and lock.locked();self.steps+=1;self.t+=.02
            if self.steps==2:
                self.done=True;self.result={'success':False,'timeout':True,'final_place_error_m':.1}
    class Drawer:
        def __init__(self,config):pass
        def record(self,ep):pass
        def draw(self,*args):assert lock.locked()
    viewer=Viewer(None)
    def launch(*args,**kwargs):viewer.callback=kwargs['key_callback'];return viewer
    monkeypatch.setattr(demo,'Episode',Episode);monkeypatch.setattr(demo,'MarkerDrawer',Drawer)
    monkeypatch.setattr(demo,'load_config',lambda path:cfg)
    monkeypatch.setattr(demo.mujoco.viewer,'launch_passive',launch)
    monkeypatch.setattr(demo,'time',Clock());monkeypatch.setattr(demo.sys,'argv',['run_demo.py','--perception','learned','--no-auto-reset'])
    demo.main();episode=created[0]
    assert episode.resets==[0] and len(episode.moves)==1
    assert episode.perception.calls==2 and episode.steps==2
    assert episode.perception.closed and viewer.syncs==20
