"""Record the code, configuration, assets and dependency versions of an evaluation."""
import hashlib
import importlib.metadata
import platform

from .config import ROOT, resolve
from .evaluation import PlacementEvaluator


def hashes(files, base):
    return {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def run_metadata(cfg, perception):
    files = list((ROOT/'src').rglob('*.py')) + list(ROOT.glob('run_*.py'))
    robot = resolve(cfg['sim']['panda_file']).parent
    assets = [p for p in robot.rglob('*') if p.is_file()]
    if perception not in ('camera','ground_truth','learned'):raise ValueError('Unknown perception mode')
    assumptions={'camera':'ideal rendered RGB-D, calibrated cameras, known geometry; HSV baseline',
                 'ground_truth':'noisy privileged body poses; oracle baseline',
                 'learned':'RGB plus noisy optical-z/validity, calibrated cameras, fixed nominal geometry; frozen learned masks; synchronous lockstep CPU worker'}
    if perception=='learned':
        files += [ROOT/'training/learned_pose_worker.py',ROOT/'training/pose_evaluation/geometry.py',
                  resolve(cfg['perception']['learned']['candidate']),resolve(cfg['perception']['learned']['protocol'])]
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': {n: importlib.metadata.version(n) for n in
                             ('mujoco', 'numpy', 'opencv-python', 'matplotlib', 'pyyaml')},
            'source_sha256': hashes(files, ROOT), 'robot_asset_sha256': hashes(assets, robot),
            'scene_sha256': hashlib.sha256(resolve(cfg['sim']['scene_file']).read_bytes()).hexdigest(),
            'evaluator_version': PlacementEvaluator.VERSION,
            'effective_perception': perception,
            'sensor_assumptions': assumptions[perception],
            'config': cfg}
