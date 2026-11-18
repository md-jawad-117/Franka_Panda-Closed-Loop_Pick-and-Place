"""Offscreen rendering to MP4 with text overlays and an optional wrist-cam inset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from .episode import Episode
from .markers import MarkerDrawer

EVENT_TEXT = {"arm_push": ("ARM PUSHED", (255, 80, 80)),
              "slot_move": ("SLOT MOVED", (255, 230, 0)),
              "object_nudge": ("OBJECT NUDGED", (255, 120, 255))}


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
             color: tuple[int, int, int], thick: int = 2) -> None:
    """Text on a translucent dark box (outlines leave ghost glyphs at small scales)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    pad = max(3, int(6 * scale))
    x0, y0 = max(x - pad, 0), max(y - th - pad, 0)
    x1, y1 = min(x + tw + pad, img.shape[1]), min(y + base + pad, img.shape[0])
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * 0.45).astype(img.dtype)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


class FrameRenderer:
    """Renders one annotated frame of an episode."""

    def __init__(self, ep: Episode, cfg: dict[str, Any], width: int, height: int,
                 camera: str, pip_camera: str | None, markers: MarkerDrawer | None = None):
        self.ep, self.cfg = ep, cfg
        self.w, self.h = width, height
        self.camera, self.pip = camera, pip_camera
        self.r = mujoco.Renderer(ep.sim.model, height, width)
        ps = cfg["recording"]["pip_scale"]
        self.pw, self.ph = int(width * ps), int(height * ps)
        self.r_pip = mujoco.Renderer(ep.sim.model, self.ph, self.pw) if pip_camera else None
        self.markers = markers or MarkerDrawer(cfg)
        self.opt = mujoco.MjvOption()

    def close(self) -> None:
        self.r.close()
        if self.r_pip:
            self.r_pip.close()

    def render(self, label: str | None = None) -> np.ndarray:
        ep = self.ep
        self.r.update_scene(ep.sim.data, camera=ep.sim.cams[self.camera], scene_option=self.opt)
        self.markers.draw(self.r.scene, ep)
        img = self.r.render().copy()
        if self.r_pip is not None:
            self.r_pip.update_scene(ep.sim.data, camera=ep.sim.cams[self.pip])
            pip = self.r_pip.render()
            x0, y0 = self.w - self.pw - 12, 12
            img[y0:y0 + self.ph, x0:x0 + self.pw] = pip
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + self.pw, y0 + self.ph), (255, 255, 255), 2)
            put_text(img, self.pip, (x0 + 6, y0 + self.ph - 8), 0.45, (255, 255, 255), 1)
        self._overlay(img, label)
        return img

    def _overlay(self, img: np.ndarray, label: str | None) -> None:
        ep = self.ep
        s = self.h / 720
        mode = "CLOSED-LOOP" if ep.mode == "closed_loop" else "OPEN-LOOP"
        col = (80, 255, 120) if ep.mode == "closed_loop" else (255, 160, 60)
        put_text(img, (label or mode)+' | '+ep.perception_mode, (int(16 * s), int(42 * s)), 1.1 * s, col, max(1, int(2 * s)))
        put_text(img, f"state: {ep.fsm.state.name}", (int(16 * s), int(80 * s)), 0.75 * s,
                 (255, 255, 255), max(1, int(2 * s)))
        put_text(img, f"t = {ep.t:5.2f} s   err = {ep.log.pos_err[-1]*1000 if ep.log.pos_err else 0:5.1f} mm",
                 (int(16 * s), int(112 * s)), 0.65 * s, (220, 220, 220), 1)
        y = int(160 * s)
        for ev in ep.perturber.events:
            if 0 <= ep.t - ev.t_start <= self.cfg["perturbations"]["marker_duration_s"]:
                text, c = EVENT_TEXT[ev.kind]
                put_text(img, text, (int(16 * s), y), 1.0 * s, c, max(1, int(2 * s)))
                y += int(40 * s)
        if ep.result is not None:
            ok = ep.result["success"]
            text = "SUCCESS" if ok else "FAIL"
            c = (80, 255, 120) if ok else (255, 70, 70)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0 * s, 4)
            put_text(img, text, ((self.w - tw) // 2, self.h - int(40 * s)), 2.0 * s, c,
                     max(2, int(4 * s)))


class VideoWriter:
    def __init__(self, path: Path, fps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_log_level="error")

    def add(self, frame: np.ndarray) -> None:
        self.w.append_data(frame)

    def close(self) -> None:
        self.w.close()


def record_episodes(eps: list[Episode], cfg: dict[str, Any], path: Path, width: int,
                    height: int, camera: str, pip_camera: str | None,
                    labels: list[str] | None = None, tail_s: float = 1.5,
                    max_s: float | None = None) -> list[dict]:
    """Run episodes in lockstep (already reset) and write one video, tiled side by side."""
    fps = cfg["recording"]["fps"]
    frs = [FrameRenderer(e, cfg, width, height, camera, pip_camera) for e in eps]
    vw = VideoWriter(path, fps)
    max_s = max_s or cfg["task"]["episode_timeout_s"]
    next_t, end_t = 0.0, None
    try:
        while True:
            for e, fr in zip(eps, frs):
                e.step()
                fr.markers.record(e)
            t = eps[0].t
            if t >= next_t:
                tiles = [fr.render(labels[i] if labels else None) for i, fr in enumerate(frs)]
                vw.add(np.concatenate(tiles, axis=1))
                next_t += 1.0 / fps
            if end_t is None and all(e.done for e in eps):
                end_t = t + tail_s
            if (end_t is not None and t >= end_t) or t > max_s + tail_s:
                break
    finally:
        vw.close()
        for fr in frs:
            fr.close()
        for ep in eps:
            ep.perception.close()
    return [e.result for e in eps]
