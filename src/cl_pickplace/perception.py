"""Pose estimation of capsule and slot: ground truth (+noise) or wrist camera RGB-D."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np

from .kinematics import wrap_half_pi
from .sim import Sim
from .sensors import RGBDSensor


@dataclass
class PoseEstimate:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0          # long-axis yaw, defined mod pi
    visible: bool = False
    lost_frames: int = 0
    source: str = ""
    measurement_t: float | None = None
    axis: np.ndarray | None = None
    quality: dict = field(default_factory=dict)
    observation: dict = field(default_factory=dict)  # capture evidence, never a replacement pose


@dataclass
class Estimate:
    capsule: PoseEstimate
    slot: PoseEstimate

    def copy(self) -> "Estimate":
        return copy.deepcopy(self)


class PoseFilter:
    """Exponential smoothing with immediate reset on large jumps."""

    def __init__(self, alpha: float, jump_m: float, jump_rad: float):
        self.alpha, self.jump_m, self.jump_rad = alpha, jump_m, jump_rad
        self.est = PoseEstimate()
        self.initialized = False

    def reset(self) -> None:
        self.est = PoseEstimate()
        self.initialized = False

    def update(self, meas: tuple[np.ndarray, float] | None, source: str,
               measurement_t: float | None = None, axis=None, quality=None, alpha=None) -> PoseEstimate:
        e = self.est
        if meas is None:
            e.visible = False
            e.lost_frames += 1
            return e
        pos, yaw = meas
        dyaw = wrap_half_pi(yaw - e.yaw)
        if (not self.initialized or np.linalg.norm(pos - e.pos) > self.jump_m
                or abs(dyaw) > self.jump_rad):
            e.pos, e.yaw = pos.copy(), wrap_half_pi(yaw)
            self.initialized = True
        else:
            a = self.alpha if alpha is None else alpha
            e.pos = (1 - a) * e.pos + a * pos
            e.yaw = wrap_half_pi(e.yaw + a * dyaw)
        e.visible, e.lost_frames, e.source = True, 0, source
        e.measurement_t = measurement_t
        e.axis = None if axis is None else np.array(axis,copy=True)
        e.quality = copy.deepcopy(quality or {})
        return e


class Perception:
    def __init__(self, sim: Sim, cfg: dict[str, Any], rng: np.random.Generator,
                 mode: str | None = None):
        self.sim = sim
        self.cfg = cfg["perception"]
        self.mode = mode or self.cfg["mode"]
        if self.mode not in ('camera', 'ground_truth', 'learned'):
            raise ValueError(f'Unknown perception mode: {self.mode}')
        self.rng = rng
        p = self.cfg
        mk = lambda: PoseFilter(p["filter_alpha"], p["jump_threshold_m"],
                                np.radians(p["jump_threshold_deg"]))
        self.f_capsule, self.f_slot = mk(), mk()
        self._camera: CameraPerception | None = None
        self._learned = None
        self.capture_context = nullcontext
        self.on_wait = None
        self.last_masks: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.f_capsule.reset()
        self.f_slot.reset()
        if self._learned is not None:
            self._learned.rng = self.rng
            self._learned.reset()

    def set_mode(self, mode: str) -> None:
        if mode not in ('camera', 'ground_truth', 'learned'):
            raise ValueError(f'Unknown perception mode: {mode}')
        self.mode = mode
        self.reset()

    def _noisy(self, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, float] | None:
        n = self.cfg["noise"]
        if not n["enabled"]:
            return pos, yaw
        if self.rng.random() < n["dropout_prob"]:
            return None
        return (pos + self.rng.normal(0, n["pos_std_m"], 3),
                yaw + np.radians(self.rng.normal(0, n["yaw_std_deg"])))

    def ground_truth(self) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]]:
        s = self.sim
        return (s.capsule_pos(), s.capsule_yaw()), (s.slot_pos(), s.slot_yaw())

    def update(self) -> Estimate:
        if self.mode == 'learned':
            if self._learned is None:
                from .learned_perception import LearnedPerception
                sensor=RGBDSensor(self.sim,320,240,hide_sites=True,synchronize=True)
                sensor.capture_context=self.capture_context
                try:
                    self._learned=LearnedPerception(sensor,self.cfg['learned'],self.rng,on_wait=self.on_wait)
                except Exception:
                    sensor.close()
                    raise
            t=self.sim.time
            measurements=self._learned.update(t)
            for target,tracker in (('capsule',self.f_capsule),('fixture',self.f_slot)):
                if measurements is not None:
                    measurement=measurements.get(target)
                    if measurement is None:
                        tracker.update(None,'')
                    else:
                        position,yaw,source,axis,quality=measurement
                        tracker.update((position,yaw),source,t,axis,quality,alpha=1.)
                elif not tracker.est.visible or tracker.est.measurement_t is None or t-tracker.est.measurement_t>self.cfg['learned']['max_age_s']:
                    tracker.update(None,'')
            for target,tracker in (('capsule',self.f_capsule),('fixture',self.f_slot)):
                tracker.est.observation = copy.deepcopy(self._learned.latest.get(target,{}))
            return Estimate(self.f_capsule.est,self.f_slot.est)
        if self.mode == "camera":
            if self._camera is None:
                sensor = RGBDSensor(self.sim, self.cfg['camera']['width'], self.cfg['camera']['height'])
                sensor.capture_context=self.capture_context
                self._camera = CameraPerception(sensor, self.cfg, self.sim.cfg['scene'])
            cap, slot = self._camera.measure()
            self.last_masks = self._camera.last_masks
            src = self._camera.last_source
            c = self.f_capsule.update(cap, src.get("capsule", ""))
            s = self.f_slot.update(slot, src.get("slot", ""))
        else:
            with self.capture_context():
                cap, slot = self.ground_truth()
            c = self.f_capsule.update(self._noisy(*cap), "gt")
            s = self.f_slot.update(self._noisy(*slot), "gt")
        return Estimate(c, s)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
        if self._learned is not None:
            self._learned.close()

    def learned_summary(self):
        return self._learned.summary() if self._learned is not None else None

    def set_live_hooks(self,capture_context,on_wait):
        """Lock shared MuJoCo capture briefly; keep CPU inference outside viewer lock."""
        self.capture_context=capture_context;self.on_wait=on_wait
        if self._learned is not None:
            self._learned.sensor.capture_context=capture_context
            self._learned.worker.on_wait=on_wait
        if self._camera is not None:self._camera.sensor.capture_context=capture_context


class CameraPerception:
    """Color segmentation + depth back-projection from wrist_cam, falling back to top_cam."""

    def __init__(self, sensor, pcfg: dict[str, Any], scene_cfg: dict[str, Any]):
        self.sensor = sensor
        c = pcfg["camera"]
        self.w, self.h = c["width"], c["height"]
        self.fallback = c["fallback_camera"]
        self.min_px = c["min_pixels"]
        r = np.array(c["hsv_red"], np.uint8)
        self.red = [(r[0], r[1]), (r[2], r[3])]
        g = np.array(c["hsv_green"], np.uint8)
        self.green = [(g[0], g[1])]
        sc = scene_cfg  # Declared known geometry and table calibration, not live poses.
        self.r = sc["capsule"]["radius"]
        self.half_len = sc["capsule"]["half_length"] + self.r
        sl = sc["slot"]
        self.slot_rim_above_floor = sl["wall_height"]
        t = sl["wall_thickness"]
        self.slot_dims = np.array([sl["cavity"][0] + 2 * t, sl["cavity"][1] + 2 * t])
        self.cap_len = 2 * (sc["capsule"]["half_length"] + self.r)
        self.table_z = sc["table_top_z"]
        self.min_height = c["min_height_above_table"]
        self.size_tol = c["size_tolerance_m"]
        self.size_tol_px = c["size_tolerance_px"]
        self.last_px_size = 0.0
        self.last_masks: dict[str, np.ndarray] = {}
        self.last_source: dict[str, str] = {}
        # Only the scene geoms count: hide nothing, but segmentation is by color.

    def close(self) -> None:
        self.sensor.close()

    def _backproject(self, frame, mask: np.ndarray) -> np.ndarray:
        fovy = frame.fovy_deg
        f = 0.5 * self.h / np.tan(np.radians(fovy) / 2)
        v, u = np.nonzero(mask)
        d = frame.depth[v, u]
        x = (u + 0.5 - self.w / 2) / f * d
        y = -(v + 0.5 - self.h / 2) / f * d
        pts_cam = np.stack([x, y, -d], axis=1)
        R = frame.world_rotation
        self.last_px_size = float(np.median(d)) / f if len(d) else 0.0   # meters per pixel
        return pts_cam @ R.T + frame.world_position

    @staticmethod
    def _mask(hsv: np.ndarray, ranges: list) -> np.ndarray:
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        # Erode: silhouette pixels mix object color with background depth.
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        return m > 0

    @staticmethod
    def _principal_yaw(xy: np.ndarray) -> float:
        c = xy - xy.mean(axis=0)
        w, v = np.linalg.eigh(c.T @ c)
        a = v[:, np.argmax(w)]
        return float(np.arctan2(a[1], a[0]))

    @staticmethod
    def _footprint(xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Center and long-axis yaw of the minimum-area rectangle around xy points."""
        (cx, cy), (w, h), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
        yaw = np.radians(ang) if w >= h else np.radians(ang + 90)
        dims = np.array([max(w, h), min(w, h)]) / 1000
        return np.array([cx, cy]) / 1000, float(yaw), dims

    @staticmethod
    def _touches_border(mask: np.ndarray) -> bool:
        return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())

    def _measure_cam(self, cam: str):
        frame = self.sensor.capture(cam)
        rgb = frame.rgb
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m_red, m_green = self._mask(hsv, self.red), self._mask(hsv, self.green)
        out = {}
        above = self.table_z + self.min_height
        if m_red.sum() >= self.min_px and not self._touches_border(m_red):
            pts = self._backproject(frame, m_red)
            pts = pts[pts[:, 2] > above]
            if len(pts) >= self.min_px:
                yaw = self._principal_yaw(pts[:, :2])
                axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                s_ax = pts @ axis
                length = s_ax.max() - s_ax.min()
                # erosion and anti-aliased rounded ends trim a few pixels
                if abs(length - self.cap_len) < self.size_tol + self.size_tol_px * self.last_px_size:
                    # Across the axis: pixel-uniform samples of the visible half-cylinder
                    # sit r*pi/4 closer to the camera than the axis. Along the axis:
                    # midpoint of the extent (perspective biases the plain mean).
                    pos = pts.mean(axis=0)
                    pos += (0.5 * (s_ax.max() + s_ax.min()) - pos @ axis) * axis
                    w = pos - frame.world_position
                    w -= (w @ axis) * axis
                    pos = pos + self.r * np.pi / 4 * w / (np.linalg.norm(w) + 1e-9)
                    out["capsule"] = (pos, yaw)
        if m_green.sum() >= self.min_px and not self._touches_border(m_green):
            pts = self._backproject(frame, m_green)
            pts = pts[pts[:, 2] > above]
            if len(pts) >= self.min_px:
                # 3D points of walls and floor all project inside the slot footprint.
                xy, yaw, dims = self._footprint(pts[:, :2])
                tol = self.size_tol + self.size_tol_px * self.last_px_size
                if np.all(np.abs(dims - self.slot_dims) < tol):
                    rim = np.percentile(pts[:, 2], 98)
                    out["slot"] = (np.array([xy[0], xy[1], rim - self.slot_rim_above_floor]), yaw)
        return out, rgb, m_red, m_green

    def measure(self):
        res, rgb, mr, mg = self._measure_cam("wrist_cam")
        self.last_masks = {"rgb": rgb, "red": mr, "green": mg}
        self.last_source = {k: "wrist_cam" for k in res}
        if ("capsule" not in res or "slot" not in res) and self.fallback:
            fb, *_ = self._measure_cam(self.fallback)
            for k, v in fb.items():
                if k not in res:
                    res[k] = v
                    self.last_source[k] = self.fallback
        return res.get("capsule"), res.get("slot")
