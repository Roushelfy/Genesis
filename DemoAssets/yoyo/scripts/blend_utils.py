"""Blend utilities for mixing teleop trajectories with manual IK keyframe corrections.

Provides:
- ``BlendKeyframe`` / ``BlendSchedule``: per-joint keyframe storage + interpolation
- ``bake_trajectory``: produce a new qpos array with blend keyframes baked in
- ``save_blend_keyframes`` / ``load_blend_keyframes``: JSON persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from replay_utils import qpos_to_joint_dict


@dataclass
class BlendKeyframe:
    """A single blend correction keyframe.

    Attributes:
        traj_frame:   Index into the original teleop trajectory.
        blend_weight: 0.0 = pure teleop, 1.0 = pure keyframe override.
        joints:       Only the joints that differ from teleop at this frame.
    """
    traj_frame: int
    blend_weight: float
    joints: dict[str, float] = field(default_factory=dict)


class BlendSchedule:
    """Ordered collection of :class:`BlendKeyframe` with interpolation."""

    def __init__(self) -> None:
        self._keyframes: list[BlendKeyframe] = []

    @property
    def keyframes(self) -> list[BlendKeyframe]:
        return list(self._keyframes)

    def __len__(self) -> int:
        return len(self._keyframes)

    def _sort(self) -> None:
        self._keyframes.sort(key=lambda kf: kf.traj_frame)

    def add_keyframe(self, kf: BlendKeyframe) -> None:
        self._keyframes.append(kf)
        self._sort()

    def remove_keyframe(self, index: int) -> None:
        if 0 <= index < len(self._keyframes):
            self._keyframes.pop(index)

    def update_keyframe(self, index: int, kf: BlendKeyframe) -> None:
        if 0 <= index < len(self._keyframes):
            self._keyframes[index] = kf
            self._sort()

    def clear(self) -> None:
        self._keyframes.clear()

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def evaluate(self, traj_frame: float) -> dict[str, tuple[float, float]]:
        """Interpolate blend keyframes at a (possibly fractional) teleop frame.

        Returns ``{joint_name: (override_value, effective_weight)}`` for every
        joint that has blend influence at this frame.  Joints not listed should
        use the original teleop value (weight 0).

        When a joint appears in only one of two surrounding keyframes, its
        *effective weight* fades to 0 on the side where it is absent, so the
        result smoothly transitions to the original teleop value -- not to 0.0.
        """
        n = len(self._keyframes)
        if n == 0:
            return {}

        if traj_frame <= self._keyframes[0].traj_frame:
            if traj_frame == self._keyframes[0].traj_frame:
                w = self._keyframes[0].blend_weight
                return {k: (v, w) for k, v in self._keyframes[0].joints.items()}
            return {}

        if traj_frame >= self._keyframes[-1].traj_frame:
            if traj_frame == self._keyframes[-1].traj_frame:
                w = self._keyframes[-1].blend_weight
                return {k: (v, w) for k, v in self._keyframes[-1].joints.items()}
            return {}

        # Find surrounding keyframes
        right = 0
        for i, kf in enumerate(self._keyframes):
            if kf.traj_frame >= traj_frame:
                right = i
                break
        left = right - 1

        kf_a = self._keyframes[left]
        kf_b = self._keyframes[right]
        span = kf_b.traj_frame - kf_a.traj_frame
        alpha = (traj_frame - kf_a.traj_frame) / max(span, 1e-12)

        w_a = kf_a.blend_weight
        w_b = kf_b.blend_weight

        all_keys = set(kf_a.joints.keys()) | set(kf_b.joints.keys())
        result: dict[str, tuple[float, float]] = {}
        for key in all_keys:
            va = kf_a.joints.get(key)
            vb = kf_b.joints.get(key)
            if va is not None and vb is not None:
                val = va + alpha * (vb - va)
                weight = w_a + alpha * (w_b - w_a)
            elif va is not None:
                # Joint only in kf_a: keep its override value, fade weight out
                val = va
                weight = w_a * (1.0 - alpha)
            else:
                assert vb is not None
                # Joint only in kf_b: keep its override value, fade weight in
                val = vb
                weight = w_b * alpha
            result[key] = (val, weight)
        return result

    def blend_joints(
        self, teleop_joints: dict[str, float], traj_frame: float
    ) -> dict[str, float]:
        """Return fully blended joint dict for *traj_frame*.

        Each joint gets its own effective weight from :meth:`evaluate`.
        Joints not present in any blend keyframe pass through from
        *teleop_joints* unchanged.
        """
        per_joint = self.evaluate(traj_frame)
        if not per_joint:
            return dict(teleop_joints)

        result = dict(teleop_joints)
        for name, (kf_val, w) in per_joint.items():
            if w < 1e-9:
                continue
            teleop_val = teleop_joints.get(name, 0.0)
            result[name] = teleop_val + w * (kf_val - teleop_val)
        return result


# ------------------------------------------------------------------
# Bake
# ------------------------------------------------------------------


def bake_trajectory(
    original_qpos: np.ndarray,
    genesis_joint_names: list[str],
    blend_schedule: BlendSchedule,
) -> np.ndarray:
    """Produce a new ``(N, D)`` qpos array with blend keyframes applied."""
    n_frames, n_dofs = original_qpos.shape
    out = original_qpos.copy()

    name_to_idx = {name: i for i, name in enumerate(genesis_joint_names) if name}

    for f in range(n_frames):
        teleop_joints = qpos_to_joint_dict(original_qpos[f], genesis_joint_names)
        blended = blend_schedule.blend_joints(teleop_joints, float(f))
        for name, val in blended.items():
            idx = name_to_idx.get(name)
            if idx is not None:
                out[f, idx] = val
    return out


# ------------------------------------------------------------------
# JSON persistence
# ------------------------------------------------------------------


def save_blend_keyframes(path: Path | str, keyframes: list[BlendKeyframe]) -> None:
    data = [
        {
            "traj_frame": kf.traj_frame,
            "blend_weight": kf.blend_weight,
            "joints": kf.joints,
        }
        for kf in keyframes
    ]
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_blend_keyframes(path: Path | str) -> list[BlendKeyframe]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [
        BlendKeyframe(
            traj_frame=int(entry["traj_frame"]),
            blend_weight=float(entry.get("blend_weight", 1.0)),
            joints={str(k): float(v) for k, v in entry.get("joints", {}).items()},
        )
        for entry in raw
    ]
