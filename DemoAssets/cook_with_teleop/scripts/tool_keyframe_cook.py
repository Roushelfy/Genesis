"""Standalone Polyscope GUI for hand-keying pan & spatula cooking sequences.

Creates cooking trajectories directly from manually placed keyframes --
no existing teleop recording needed.  Includes built-in default keyframes
for a wok-tossing (颠锅炒菜) motion.

Output is a trajectory JSON fully compatible with ``replay_cook.py``.

Workflow:
  1. Start with the default wok-toss keyframes (or load a saved set).
  2. Select a keyframe, adjust time / pan / spatula pose via sliders.
  3. Add, duplicate, delete, or reorder keyframes as needed.
  4. Preview the interpolated motion with the playback controls.
  5. Export to a trajectory JSON for physics simulation.

Usage:
    python tool_keyframe_cook.py
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

from usd_mesh_loader import load_usd_mesh

_SCRIPT_DIR = Path(__file__).resolve().parent
_ASSET_ROOT = _SCRIPT_DIR.parent
_COOK_ROOT = _ASSET_ROOT.parent / "cook"

PAN_USD = _COOK_ROOT / "Pan025" / "Pan025.usd"
SPATULA_USD = _COOK_ROOT / "Spatula018" / "Spatula018.usd"
_DEFAULT_PLACEMENT = str(_ASSET_ROOT / "placement.json")
_DEFAULT_OUTPUT = str(_COOK_ROOT / "trajectories" / "cooking_keyframed.json")
_DEFAULT_KF_FILE = str(_SCRIPT_DIR / "cook_keyframes.json")

ENTITIES = ("pan", "spatula")


# ------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------

def euler_deg_to_quat(euler_deg):
    """[rx, ry, rz] degrees (intrinsic XYZ) -> [w, x, y, z] quaternion."""
    rx, ry, rz = (math.radians(a) for a in euler_deg)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


def quat_to_euler_deg(quat):
    """[w, x, y, z] quaternion -> [rx, ry, rz] degrees (intrinsic XYZ)."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    ry = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def quat_pos_to_4x4(quat, pos):
    """[w, x, y, z] quaternion + [x, y, z] position -> 4x4 matrix."""
    w, x, y, z = quat
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 1 - 2 * (y * y + z * z)
    M[0, 1] = 2 * (x * y - w * z)
    M[0, 2] = 2 * (x * z + w * y)
    M[1, 0] = 2 * (x * y + w * z)
    M[1, 1] = 1 - 2 * (x * x + z * z)
    M[1, 2] = 2 * (y * z - w * x)
    M[2, 0] = 2 * (x * z - w * y)
    M[2, 1] = 2 * (y * z + w * x)
    M[2, 2] = 1 - 2 * (x * x + y * y)
    M[0, 3], M[1, 3], M[2, 3] = pos[0], pos[1], pos[2]
    return M


def _apply_4x4(verts, M):
    V4 = np.ones((len(verts), 4), dtype=np.float64)
    V4[:, :3] = verts
    return (V4 @ M.T)[:, :3]


def slerp(q0, q1, t):
    """Spherical linear interpolation for [w, x, y, z] quaternions."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = a + t * (b - a)
        return (r / np.linalg.norm(r)).tolist()
    theta_0 = math.acos(dot)
    sin_t0 = math.sin(theta_0)
    s0 = math.sin((1.0 - t) * theta_0) / sin_t0
    s1 = math.sin(t * theta_0) / sin_t0
    r = s0 * a + s1 * b
    return (r / np.linalg.norm(r)).tolist()


def lerp_vec3(a, b, t):
    return [a[i] + t * (b[i] - a[i]) for i in range(3)]


def quat_conj(q):
    """Conjugate of [w, x, y, z] quaternion."""
    return [q[0], -q[1], -q[2], -q[3]]


def quat_mul(a, b):
    """Hamilton product of two [w, x, y, z] quaternions."""
    return [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ]


def quat_log(q):
    """Log map: unit quaternion -> 3-D rotation vector."""
    w = max(-1.0, min(1.0, q[0]))
    v = np.array(q[1:], dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nv < 1e-10:
        return np.zeros(3, dtype=np.float64)
    theta = math.acos(abs(w))
    sign = 1.0 if w >= 0.0 else -1.0
    return sign * (v / nv) * theta


def quat_exp(v):
    """Exp map: 3-D rotation vector -> unit quaternion."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    axis = v / theta
    s = math.sin(theta)
    return [math.cos(theta), axis[0]*s, axis[1]*s, axis[2]*s]


def _ensure_quat_neighbourhood(q_ref, q):
    """Flip *q* if it is in the opposite hemisphere from *q_ref*."""
    dot = sum(a * b for a, b in zip(q_ref, q))
    if dot < 0.0:
        return [-c for c in q]
    return list(q)


def squad_intermediate(q_prev, q_curr, q_next):
    """Intermediate control quaternion for Squad spline at *q_curr*."""
    qi = quat_conj(q_curr)
    log_next = quat_log(quat_mul(qi, q_next))
    log_prev = quat_log(quat_mul(qi, q_prev))
    exp_arg = -(log_next + log_prev) / 4.0
    return quat_mul(q_curr, quat_exp(exp_arg))


def squad(q_prev, q0, q1, q_next, t):
    """Squad spline interpolation between *q0* and *q1* (t in [0,1]).

    Uses *q_prev* and *q_next* as neighbours to derive tangent control quats.
    """
    s0 = squad_intermediate(q_prev, q0, q1)
    s1 = squad_intermediate(q0, q1, q_next)
    return slerp(slerp(q0, q1, t), slerp(s0, s1, t), 2.0 * t * (1.0 - t))


# ------------------------------------------------------------------
# Catmull-Rom cubic Hermite spline (for vec3 positions)
# ------------------------------------------------------------------

def _catmull_rom_tangent(t_prev, t_curr, t_next, v_prev, v_curr, v_next):
    """Catmull-Rom tangent at *v_curr* with non-uniform time spacing."""
    dt = t_next - t_prev
    if abs(dt) < 1e-12:
        return np.zeros_like(v_curr)
    return (v_next - v_prev) / dt


def cubic_hermite_vec3(times, values, t):
    """Evaluate a Catmull-Rom cubic Hermite spline at time *t*.

    *times*:  1-D array of N knot times (sorted).
    *values*: (N, 3) array of knot positions.
    Returns:  (3,) interpolated position.
    """
    n = len(times)
    if n == 0:
        return np.zeros(3, dtype=np.float64)
    if n == 1 or t <= times[0]:
        return values[0].copy()
    if t >= times[-1]:
        return values[-1].copy()

    seg = 0
    for i in range(n - 1):
        if times[i] <= t <= times[i + 1]:
            seg = i
            break

    dt = times[seg + 1] - times[seg]
    if dt < 1e-12:
        return values[seg].copy()
    s = (t - times[seg]) / dt

    p0, p1 = values[seg], values[seg + 1]

    # Tangents (Catmull-Rom: use neighbours when available, else endpoint)
    if seg > 0:
        m0 = _catmull_rom_tangent(times[seg-1], times[seg], times[seg+1],
                                  values[seg-1], values[seg], values[seg+1]) * dt
    else:
        m0 = (p1 - p0)

    if seg + 2 < n:
        m1 = _catmull_rom_tangent(times[seg], times[seg+1], times[seg+2],
                                  values[seg], values[seg+1], values[seg+2]) * dt
    else:
        m1 = (p1 - p0)

    # Hermite basis functions
    h00 = 2*s*s*s - 3*s*s + 1
    h10 = s*s*s - 2*s*s + s
    h01 = -2*s*s*s + 3*s*s
    h11 = s*s*s - s*s

    return h00*p0 + h10*m0 + h01*p1 + h11*m1


# ------------------------------------------------------------------
# PoseKeyframe
# ------------------------------------------------------------------

@dataclass
class PoseKeyframe:
    time: float
    pan_pos: list
    pan_quat: list
    spatula_pos: list
    spatula_quat: list

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "pan": {"pos": list(self.pan_pos), "quat": list(self.pan_quat)},
            "spatula": {"pos": list(self.spatula_pos), "quat": list(self.spatula_quat)},
        }

    @staticmethod
    def from_dict(d: dict) -> PoseKeyframe:
        return PoseKeyframe(
            time=d["time"],
            pan_pos=list(d["pan"]["pos"]),
            pan_quat=list(d["pan"]["quat"]),
            spatula_pos=list(d["spatula"]["pos"]),
            spatula_quat=list(d["spatula"]["quat"]),
        )

    def clone(self) -> PoseKeyframe:
        return PoseKeyframe(
            self.time,
            list(self.pan_pos), list(self.pan_quat),
            list(self.spatula_pos), list(self.spatula_quat),
        )


def interpolate_kf(k0: PoseKeyframe, k1: PoseKeyframe, t_norm: float) -> PoseKeyframe:
    t = max(0.0, min(1.0, t_norm))
    return PoseKeyframe(
        time=k0.time + t * (k1.time - k0.time),
        pan_pos=lerp_vec3(k0.pan_pos, k1.pan_pos, t),
        pan_quat=slerp(k0.pan_quat, k1.pan_quat, t),
        spatula_pos=lerp_vec3(k0.spatula_pos, k1.spatula_pos, t),
        spatula_quat=slerp(k0.spatula_quat, k1.spatula_quat, t),
    )


def evaluate_at_time(keyframes: list[PoseKeyframe], t: float,
                     spline: bool = False) -> PoseKeyframe | None:
    """Evaluate interpolated pose at time *t*.

    When *spline* is True, uses Catmull-Rom cubic Hermite for positions
    and Squad for quaternions; otherwise plain lerp + slerp.
    """
    if not keyframes:
        return None
    n = len(keyframes)
    if t <= keyframes[0].time:
        return keyframes[0].clone()
    if t >= keyframes[-1].time:
        return keyframes[-1].clone()

    if not spline or n < 3:
        # Piecewise linear / slerp fallback
        for i in range(n - 1):
            if keyframes[i].time <= t <= keyframes[i + 1].time:
                dt = keyframes[i + 1].time - keyframes[i].time
                if dt < 1e-9:
                    return keyframes[i].clone()
                return interpolate_kf(keyframes[i], keyframes[i + 1],
                                      (t - keyframes[i].time) / dt)
        return keyframes[-1].clone()

    # ---- Spline path ----
    times = np.array([k.time for k in keyframes], dtype=np.float64)

    # Positions: Catmull-Rom cubic Hermite per entity
    pan_vals = np.array([k.pan_pos for k in keyframes], dtype=np.float64)
    spat_vals = np.array([k.spatula_pos for k in keyframes], dtype=np.float64)
    pan_pos = cubic_hermite_vec3(times, pan_vals, t).tolist()
    spat_pos = cubic_hermite_vec3(times, spat_vals, t).tolist()

    # Quaternions: Squad (needs segment index + 4 neighbours)
    # Ensure consistent hemisphere first
    pan_quats = [list(keyframes[0].pan_quat)]
    spat_quats = [list(keyframes[0].spatula_quat)]
    for i in range(1, n):
        pan_quats.append(_ensure_quat_neighbourhood(pan_quats[i-1], keyframes[i].pan_quat))
        spat_quats.append(_ensure_quat_neighbourhood(spat_quats[i-1], keyframes[i].spatula_quat))

    # Find segment
    seg = 0
    for i in range(n - 1):
        if times[i] <= t <= times[i + 1]:
            seg = i
            break
    seg_dt = times[seg + 1] - times[seg]
    s = (t - times[seg]) / max(seg_dt, 1e-9)

    def _squad_eval(quats, seg_idx, s_val):
        i0 = max(seg_idx - 1, 0)
        i1 = seg_idx
        i2 = seg_idx + 1
        i3 = min(seg_idx + 2, n - 1)
        return squad(quats[i0], quats[i1], quats[i2], quats[i3], s_val)

    pan_quat = _squad_eval(pan_quats, seg, s)
    spat_quat = _squad_eval(spat_quats, seg, s)

    # Renormalize
    def _norm_q(q):
        nrm = math.sqrt(sum(c*c for c in q))
        return [c / nrm for c in q] if nrm > 1e-12 else [1, 0, 0, 0]

    return PoseKeyframe(t, pan_pos, _norm_q(pan_quat),
                        spat_pos, _norm_q(spat_quat))


# ------------------------------------------------------------------
# Default wok-toss keyframes (颠锅炒菜)
# ------------------------------------------------------------------

def _kf(t, pp, pe, sp, se) -> PoseKeyframe:
    return PoseKeyframe(t, list(pp), euler_deg_to_quat(pe),
                        list(sp), euler_deg_to_quat(se))


def default_wok_toss_keyframes() -> list[PoseKeyframe]:
    """Three-cycle wok-tossing (颠锅) from parametric model.

    Based on "The physics of tossing fried rice" (Ko & Hu, J. R. Soc.
    Interface, 2020).  Professional chef kinematics:

        θ_i(t) = θ̄_i + A_i · cos(2πft + φ_i)

    Measured constants (eq. 4.1):
        θ̄₁ = 0.3 rad   (mean position angle)
        θ̄₂ = 0.0 rad   (mean tilt — horizontal)
        A₁ = A₂ = 0.2 rad ≈ 11.5°  (amplitude)
        f  = 3.0 Hz     (period = 0.33 s)
        φ  = 1.0 rad ≈ 57°  (phase lag: tilt trails translation)

    Key mechanics:
      · Wok pivots on stove rim (see-saw contact).
      · Push forward + far-edge DOWN = catch falling food.
      · Pull back  + far-edge UP   = toss food airborne.
      · Food detaches at ~1.5 g deceleration, airborne ~0.22 s.

    For animation we slow to ~0.55 s/cycle and slightly increase
    amplitudes for visual clarity.

    Axis mapping (after -88.5° yaw):
      rx > rest(-10°)  →  far edge (锅头) UP   (toss)
      rx < rest(-10°)  →  far edge (锅头) DOWN (catch)
    """
    PAN_RX0  = -10.0                  # rest tilt (deg)
    PAN_RZ   = -88.5                  # base yaw
    SPAT_E   = [-44.5, -16.7, -131.7] # spatula rest euler

    # ---- parametric model (scaled for animation) ----
    T      = 0.35    # cycle period (s), fast vigorous toss
    PHI    = 1.0     # phase lag (rad) — from paper
    A_X    = 0.115   # X-position half-amplitude (m)  +15%
    A_Z    = 0.0575  # Z-position half-amplitude (m)  +15%
    A_RX   = 28.75   # tilt half-amplitude (deg)      +15%
    X0, Z0 = 0.50, 0.90
    N_CYC  = 13
    KF_PER = 6       # keyframes per cycle

    omega = 2.0 * math.pi / T

    def _pan_at(t):
        phase = omega * t
        x  = X0 + A_X * math.cos(phase)
        z  = Z0 - A_Z * math.cos(phase + PHI + math.pi / 3)
        rx = PAN_RX0 + A_RX * math.cos(phase + PHI + math.pi)
        return [x, 0.04, z], [rx, 0.0, PAN_RZ]

    def _spat_at(t):
        """Spatula motion with independent per-axis phasing.

        Real technique (助翻勺):
          · Push phase  — spatula leads wok, pushes food forward & presses
                          into the wok (X forward, Z low, rx tilted down)
          · Toss phase  — spatula lifts high to clear airborne food
                          (X retreats, Z peaks, rx tilts back)
          · Catch phase — spatula dips back in, guides food, quick stir
                          (Z drops, rx scoops, slight Y lateral wiggle)
        Each axis has its own phase offset from the wok cycle.
        A 2nd harmonic adds the "quick-push, slow-return" asymmetry.
        """
        phase = omega * t

        # X: track wok proportionally — always stay behind wok center
        #    wok X = X0 + A_X*cos(phase), spatula follows ~48% of that swing
        #    with center 6cm behind wok center → never overshoots
        sx = (X0 - 0.06) + 0.52 * A_X * math.cos(phase)
        # 2nd harmonic: small stir asymmetry
        sx += 0.008 * math.cos(2 * phase)

        # Y: lateral wiggle simulates stirring side-to-side
        sy = 0.14 + 0.02 * math.sin(phase + 0.5)

        # Z: HIGH during toss (clear airborne food), LOW during catch/stir
        sz = 0.99 - 0.015 * math.cos(phase + PHI - 0.3)
        # 2nd harmonic: quick dip into wok for the stir flick
        sz += 0.005 * math.sin(2 * phase + PHI + 1.5)

        # rx: tilt down (scoop/push) during push, tilt back during lift
        srx = SPAT_E[0] + 18.0 * math.cos(phase + PHI + math.pi * 0.8)
        # 2nd harmonic: sharper scoop flick
        srx += 7.0 * math.cos(2 * phase + PHI + 1.0)

        # ry: lateral tilt oscillation (wrist roll during stir)
        sry = SPAT_E[1] + 7.0 * math.sin(phase + 0.3)

        return [sx, sy, sz], [srx, sry, SPAT_E[2]]

    # fixed rest pose (matches the original pre-amplitude-change frame 0)
    PAN_REST_POS  = [0.56, 0.04, 0.911464602411427]
    PAN_REST_E    = [-18.104534588022098, 0.0, PAN_RZ]
    SPAT_REST_POS = [0.5078, 0.14958851077208407, 0.9815197279112524]
    SPAT_REST_E   = [-64.18395154351921, -14.631358553370623, SPAT_E[2]]

    kfs: list[PoseKeyframe] = []

    # rest start (fixed, does not change with amplitude)
    kfs.append(_kf(0.0, PAN_REST_POS, PAN_REST_E, SPAT_REST_POS, SPAT_REST_E))

    # ramp-in: first full cycle, blend from rest toward full amplitude
    RAMP_KFS = KF_PER
    for j in range(RAMP_KFS):
        frac = (j + 1) / RAMP_KFS
        t_param = frac * T
        t_real  = frac * T
        pp, pe = _pan_at(t_param)
        sp, se = _spat_at(t_param)
        blend = frac  # 0→1 over the ramp cycle
        pp[0] = PAN_REST_POS[0] + blend * (pp[0] - PAN_REST_POS[0])
        pp[2] = PAN_REST_POS[2] + blend * (pp[2] - PAN_REST_POS[2])
        pe[0] = PAN_REST_E[0]   + blend * (pe[0] - PAN_REST_E[0])
        sp[0] = SPAT_REST_POS[0] + blend * (sp[0] - SPAT_REST_POS[0])
        sp[1] = SPAT_REST_POS[1] + blend * (sp[1] - SPAT_REST_POS[1])
        sp[2] = SPAT_REST_POS[2] + blend * (sp[2] - SPAT_REST_POS[2])
        se[0] = SPAT_REST_E[0]   + blend * (se[0] - SPAT_REST_E[0])
        se[1] = SPAT_REST_E[1]   + blend * (se[1] - SPAT_REST_E[1])
        kfs.append(_kf(t_real, pp, pe, sp, se))

    # full cycles with small random perturbation for naturalness
    import random
    rng = random.Random(7)
    t_offset = T
    for cyc in range(N_CYC):
        for j in range(KF_PER):
            t_param = t_offset + (cyc + (j + 1) / KF_PER) * T
            t_real  = t_param
            pp, pe = _pan_at(t_param)
            sp, se = _spat_at(t_param)
            pp[0] += rng.gauss(0, 0.004)
            pp[2] += rng.gauss(0, 0.003)
            pe[0] += rng.gauss(0, 1.2)
            sp[0] += rng.gauss(0, 0.004)
            sp[1] += rng.gauss(0, 0.003)
            sp[2] += rng.gauss(0, 0.003)
            se[0] += rng.gauss(0, 1.5)
            se[1] += rng.gauss(0, 1.0)
            kfs.append(_kf(t_real, pp, pe, sp, se))

    # ramp-out: blend back to rest over half cycle
    t_ramp_start = kfs[-1].time
    RAMP_OUT_KFS = KF_PER // 2
    for j in range(RAMP_OUT_KFS):
        frac = (j + 1) / RAMP_OUT_KFS
        t_param_end = t_ramp_start + frac * T / 2
        pp, pe = _pan_at(t_param_end)
        sp, se = _spat_at(t_param_end)
        blend = 1.0 - frac  # 1→0
        pp[0] = PAN_REST_POS[0] + blend * (pp[0] - PAN_REST_POS[0])
        pp[2] = PAN_REST_POS[2] + blend * (pp[2] - PAN_REST_POS[2])
        pe[0] = PAN_REST_E[0]   + blend * (pe[0] - PAN_REST_E[0])
        sp[0] = SPAT_REST_POS[0] + blend * (sp[0] - SPAT_REST_POS[0])
        sp[1] = SPAT_REST_POS[1] + blend * (sp[1] - SPAT_REST_POS[1])
        sp[2] = SPAT_REST_POS[2] + blend * (sp[2] - SPAT_REST_POS[2])
        se[0] = SPAT_REST_E[0]   + blend * (se[0] - SPAT_REST_E[0])
        se[1] = SPAT_REST_E[1]   + blend * (se[1] - SPAT_REST_E[1])
        kfs.append(_kf(t_param_end, pp, pe, sp, se))

    # final rest
    kfs.append(_kf(kfs[-1].time + T / 4, PAN_REST_POS, PAN_REST_E,
                   SPAT_REST_POS, SPAT_REST_E))

    return kfs


# ------------------------------------------------------------------
# Keyframe I/O
# ------------------------------------------------------------------

def save_keyframes_json(path, keyframes):
    Path(path).write_text(
        json.dumps([kf.to_dict() for kf in keyframes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_keyframes_json(path) -> list[PoseKeyframe]:
    with open(path, encoding="utf-8") as f:
        return [PoseKeyframe.from_dict(d) for d in json.load(f)]


def export_trajectory(keyframes: list[PoseKeyframe], fps: float = 60.0,
                      spline: bool = True) -> dict:
    """Bake keyframes into a per-frame trajectory compatible with replay_cook.py."""
    if not keyframes:
        return {"frames": []}
    t0, t1 = keyframes[0].time, keyframes[-1].time
    dt = 1.0 / fps
    frames = []
    t = t0
    while t <= t1 + dt * 0.5:
        kf = evaluate_at_time(keyframes, t, spline=spline)
        frames.append({
            "sim_time": t,
            "pan": {"pos": list(kf.pan_pos), "quat": list(kf.pan_quat)},
            "spatula": {"pos": list(kf.spatula_pos), "quat": list(kf.spatula_quat)},
        })
        t += dt
    return {"frames": frames}


# ------------------------------------------------------------------
# Clipboard for copy-paste poses
# ------------------------------------------------------------------

_clipboard: dict | None = None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    # ---- Load USD meshes ----
    print("[kf-tool] Loading USD meshes ...")
    pan_verts_raw, pan_faces = load_usd_mesh(PAN_USD)
    spatula_verts_raw, spatula_faces = load_usd_mesh(SPATULA_USD)

    pan_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if Path(_DEFAULT_PLACEMENT).exists():
        try:
            with open(_DEFAULT_PLACEMENT, encoding="utf-8") as f:
                placement = json.load(f)
            raw = placement.get("pan", {}).get("scale", [1.0, 1.0, 1.0])
            if isinstance(raw, (int, float)):
                pan_scale[:] = raw
            else:
                pan_scale[:] = raw
            print(f"[kf-tool] Pan scale: {pan_scale.tolist()}")
        except Exception as e:
            print(f"[kf-tool] Placement read failed (non-fatal): {e}")

    pan_verts = pan_verts_raw * pan_scale

    # ---- Polyscope ----
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    pan_mesh = ps.register_surface_mesh("pan", pan_verts, pan_faces)
    pan_mesh.set_transparency(1.0)
    pan_mesh.set_color((0.85, 0.75, 0.55))

    spat_mesh = ps.register_surface_mesh("spatula", spatula_verts_raw, spatula_faces)
    spat_mesh.set_transparency(1.0)
    spat_mesh.set_color((0.60, 0.60, 0.70))

    pan_ghost = ps.register_surface_mesh("pan_ghost", pan_verts, pan_faces)
    pan_ghost.set_transparency(0.35)
    pan_ghost.set_color((1.0, 0.35, 0.35))
    pan_ghost.set_enabled(False)

    spat_ghost = ps.register_surface_mesh("spatula_ghost", spatula_verts_raw, spatula_faces)
    spat_ghost.set_transparency(0.35)
    spat_ghost.set_color((1.0, 0.35, 0.35))
    spat_ghost.set_enabled(False)

    def _pose_mesh(entity, pos, quat, ghost=False):
        M = quat_pos_to_4x4(quat, pos)
        if entity == "pan":
            v = _apply_4x4(pan_verts, M)
            (pan_ghost if ghost else pan_mesh).update_vertex_positions(v)
        else:
            v = _apply_4x4(spatula_verts_raw, M)
            (spat_ghost if ghost else spat_mesh).update_vertex_positions(v)

    # ---- State ----
    keyframes: list[PoseKeyframe] = default_wok_toss_keyframes()

    st = {
        "sel": 0,
        "kf_file": _DEFAULT_KF_FILE,
        "out_file": _DEFAULT_OUTPUT,
        "fps": 60.0,
        "status": f"Default wok-toss loaded ({len(keyframes)} KFs)",
        "kf_time": 0.0,
        "pan_pos": [0.0, 0.0, 0.0],
        "pan_euler": [0.0, 0.0, 0.0],
        "pan_quat": [1.0, 0.0, 0.0, 0.0],
        "spatula_pos": [0.0, 0.0, 0.0],
        "spatula_euler": [0.0, 0.0, 0.0],
        "spatula_quat": [1.0, 0.0, 0.0, 0.0],
    }

    play = {
        "on": False,
        "t": 0.0,
        "speed": 1.0,
        "loop": True,
        "wall": time.monotonic(),
    }

    ghost_cfg = {"on": True}
    interp_cfg = {"spline": True}  # True = Catmull-Rom + Squad, False = lerp + slerp

    # ---- Sync helpers ----

    def _kf_to_sliders(idx):
        if not keyframes:
            return
        idx = max(0, min(idx, len(keyframes) - 1))
        kf = keyframes[idx]
        st["sel"] = idx
        st["kf_time"] = kf.time
        st["pan_pos"][:] = kf.pan_pos
        st["pan_quat"][:] = kf.pan_quat
        st["pan_euler"][:] = quat_to_euler_deg(kf.pan_quat)
        st["spatula_pos"][:] = kf.spatula_pos
        st["spatula_quat"][:] = kf.spatula_quat
        st["spatula_euler"][:] = quat_to_euler_deg(kf.spatula_quat)

    def _sliders_to_kf():
        if not keyframes:
            return
        kf = keyframes[st["sel"]]
        kf.time = st["kf_time"]
        kf.pan_pos[:] = st["pan_pos"]
        kf.pan_quat[:] = st["pan_quat"]
        kf.spatula_pos[:] = st["spatula_pos"]
        kf.spatula_quat[:] = st["spatula_quat"]

    def _refresh_viewport():
        _pose_mesh("pan", st["pan_pos"], st["pan_quat"])
        _pose_mesh("spatula", st["spatula_pos"], st["spatula_quat"])

    def _set_viewport_from_time(t):
        kf = evaluate_at_time(keyframes, t, spline=interp_cfg["spline"])
        if kf is None:
            return
        st["pan_pos"][:] = kf.pan_pos
        st["pan_quat"][:] = kf.pan_quat
        st["pan_euler"][:] = quat_to_euler_deg(kf.pan_quat)
        st["spatula_pos"][:] = kf.spatula_pos
        st["spatula_quat"][:] = kf.spatula_quat
        st["spatula_euler"][:] = quat_to_euler_deg(kf.spatula_quat)
        _refresh_viewport()

    def _refresh_ghosts():
        if not ghost_cfg["on"] or not keyframes or play["on"]:
            pan_ghost.set_enabled(False)
            spat_ghost.set_enabled(False)
            return
        idx = st["sel"]
        nxt = idx + 1 if idx < len(keyframes) - 1 else idx - 1
        if nxt < 0 or nxt >= len(keyframes):
            pan_ghost.set_enabled(False)
            spat_ghost.set_enabled(False)
            return
        gkf = keyframes[nxt]
        _pose_mesh("pan", gkf.pan_pos, gkf.pan_quat, ghost=True)
        _pose_mesh("spatula", gkf.spatula_pos, gkf.spatula_quat, ghost=True)
        pan_ghost.set_enabled(True)
        spat_ghost.set_enabled(True)

    # ---- Init viewport ----
    _kf_to_sliders(0)
    _refresh_viewport()

    # ---- GUI panels ----

    def _panel_file():
        if not imgui.TreeNodeEx("File I/O", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        c, v = imgui.InputText("KF File##fio", st["kf_file"])
        if c:
            st["kf_file"] = v

        if imgui.Button("Save##fio"):
            try:
                save_keyframes_json(st["kf_file"], keyframes)
                st["status"] = f"Saved {len(keyframes)} KFs -> {Path(st['kf_file']).name}"
            except Exception as e:
                st["status"] = f"Save error: {e}"
            print(f"[kf-tool] {st['status']}")
        imgui.SameLine()
        if imgui.Button("Load##fio"):
            try:
                loaded = load_keyframes_json(st["kf_file"])
                keyframes.clear()
                keyframes.extend(loaded)
                st["sel"] = 0
                _kf_to_sliders(0)
                _refresh_viewport()
                st["status"] = f"Loaded {len(keyframes)} KFs <- {Path(st['kf_file']).name}"
            except Exception as e:
                st["status"] = f"Load error: {e}"
            print(f"[kf-tool] {st['status']}")
        imgui.SameLine()
        if imgui.Button("Reset Default##fio"):
            keyframes.clear()
            keyframes.extend(default_wok_toss_keyframes())
            st["sel"] = 0
            _kf_to_sliders(0)
            _refresh_viewport()
            st["status"] = f"Reset to default ({len(keyframes)} KFs)"

        imgui.Separator()
        c, v = imgui.InputText("Output##fio", st["out_file"])
        if c:
            st["out_file"] = v
        c, fps = imgui.SliderFloat("Export FPS##fio", st["fps"], 10.0, 120.0)
        if c:
            st["fps"] = fps

        if imgui.Button("Export Trajectory##fio"):
            try:
                traj = export_trajectory(keyframes, st["fps"],
                                        spline=interp_cfg["spline"])
                out = Path(st["out_file"])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(traj, indent=2, ensure_ascii=False),
                               encoding="utf-8")
                st["status"] = f"Exported {len(traj['frames'])} frames -> {out.name}"
            except Exception as e:
                st["status"] = f"Export error: {e}"
            print(f"[kf-tool] {st['status']}")

        if st["status"]:
            imgui.TextWrapped(st["status"])
        imgui.TreePop()

    def _panel_kf_list():
        if not imgui.TreeNodeEx("Keyframes", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        n = len(keyframes)
        imgui.Text(f"Total: {n}")

        # --- Management buttons ---
        if imgui.Button("+ Add After##kfl"):
            new = PoseKeyframe(
                st["kf_time"] + 0.15,
                list(st["pan_pos"]), list(st["pan_quat"]),
                list(st["spatula_pos"]), list(st["spatula_quat"]),
            )
            keyframes.insert(st["sel"] + 1, new)
            st["sel"] += 1
            _kf_to_sliders(st["sel"])
        imgui.SameLine()
        if imgui.Button("Dup##kfl") and n > 0:
            dup = keyframes[st["sel"]].clone()
            dup.time += 0.10
            keyframes.insert(st["sel"] + 1, dup)
            st["sel"] += 1
            _kf_to_sliders(st["sel"])
        imgui.SameLine()
        if imgui.Button("Del##kfl") and n > 1:
            keyframes.pop(st["sel"])
            st["sel"] = min(st["sel"], len(keyframes) - 1)
            _kf_to_sliders(st["sel"])
            _refresh_viewport()

        if imgui.Button("Sort Time##kfl"):
            _sliders_to_kf()
            keyframes.sort(key=lambda k: k.time)
            _kf_to_sliders(st["sel"])
        imgui.SameLine()
        if imgui.Button("Up##kfl") and st["sel"] > 0:
            _sliders_to_kf()
            i = st["sel"]
            keyframes[i - 1], keyframes[i] = keyframes[i], keyframes[i - 1]
            st["sel"] = i - 1
        imgui.SameLine()
        if imgui.Button("Down##kfl") and st["sel"] < n - 1:
            _sliders_to_kf()
            i = st["sel"]
            keyframes[i], keyframes[i + 1] = keyframes[i + 1], keyframes[i]
            st["sel"] = i + 1

        # --- Copy / Paste ---
        global _clipboard
        if imgui.Button("Copy Pose##kfl") and n > 0:
            _clipboard = {
                "pan_pos": list(st["pan_pos"]),
                "pan_quat": list(st["pan_quat"]),
                "spatula_pos": list(st["spatula_pos"]),
                "spatula_quat": list(st["spatula_quat"]),
            }
        imgui.SameLine()
        if imgui.Button("Paste Pose##kfl") and _clipboard is not None and n > 0:
            st["pan_pos"][:] = _clipboard["pan_pos"]
            st["pan_quat"][:] = _clipboard["pan_quat"]
            st["pan_euler"][:] = quat_to_euler_deg(_clipboard["pan_quat"])
            st["spatula_pos"][:] = _clipboard["spatula_pos"]
            st["spatula_quat"][:] = _clipboard["spatula_quat"]
            st["spatula_euler"][:] = quat_to_euler_deg(_clipboard["spatula_quat"])
            _sliders_to_kf()
            _refresh_viewport()

        # --- List combo ---
        labels = [f"KF {i}: t={kf.time:.3f}s" for i, kf in enumerate(keyframes)]
        if labels:
            c, sel = imgui.Combo("##kf_combo", st["sel"], labels)
            if c:
                _sliders_to_kf()
                st["sel"] = sel
                _kf_to_sliders(sel)
                _refresh_viewport()
                _refresh_ghosts()

        imgui.TreePop()

    def _panel_time():
        if not imgui.TreeNodeEx("Keyframe Time", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return
        c, t = imgui.InputFloat("Time (s)##t", st["kf_time"], 0.01, 0.1, "%.3f")
        if c:
            st["kf_time"] = max(0.0, t)
            _sliders_to_kf()
        if keyframes:
            dur = keyframes[-1].time - keyframes[0].time
            imgui.Text(f"Range: {keyframes[0].time:.3f}s .. {keyframes[-1].time:.3f}s  "
                       f"Duration: {dur:.3f}s")
        imgui.TreePop()

    def _panel_entity(entity):
        label = entity.capitalize()
        if not imgui.TreeNodeEx(f"{label} Pose##ent", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        pos = st[f"{entity}_pos"]
        euler = st[f"{entity}_euler"]
        quat = st[f"{entity}_quat"]

        pose_ch = False
        c, pos[0] = imgui.SliderFloat(f"X##{entity}", pos[0], -0.5, 1.5)
        pose_ch |= c
        c, pos[1] = imgui.SliderFloat(f"Y##{entity}", pos[1], -0.5, 1.5)
        pose_ch |= c
        c, pos[2] = imgui.SliderFloat(f"Z##{entity}", pos[2], 0.0, 1.5)
        pose_ch |= c

        imgui.Separator()
        euler_ch = False
        c, euler[0] = imgui.SliderFloat(f"Rx##{entity}", euler[0], -180.0, 180.0)
        euler_ch |= c
        c, euler[1] = imgui.SliderFloat(f"Ry##{entity}", euler[1], -180.0, 180.0)
        euler_ch |= c
        c, euler[2] = imgui.SliderFloat(f"Rz##{entity}", euler[2], -180.0, 180.0)
        euler_ch |= c

        if euler_ch:
            quat[:] = euler_deg_to_quat(euler)

        imgui.Text(f"q: w={quat[0]:+.4f} x={quat[1]:+.4f} "
                   f"y={quat[2]:+.4f} z={quat[3]:+.4f}")

        if pose_ch or euler_ch:
            _pose_mesh(entity, pos, quat)
            _sliders_to_kf()

        imgui.TreePop()

    def _panel_playback():
        if not imgui.TreeNodeEx("Preview Playback", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return
        if not keyframes:
            imgui.Text("No keyframes.")
            imgui.TreePop()
            return

        t0 = keyframes[0].time
        t1 = keyframes[-1].time
        imgui.Text(f"Time: {play['t']:.3f}s / {t1:.3f}s")

        if imgui.Button("Play" if not play["on"] else "Pause"):
            play["on"] = not play["on"]
            play["wall"] = time.monotonic()
            if play["on"]:
                _sliders_to_kf()
        imgui.SameLine()
        if imgui.Button("Reset##pb"):
            play["on"] = False
            play["t"] = t0
            _set_viewport_from_time(t0)

        c, t = imgui.SliderFloat("Scrub##pb", play["t"], t0, max(t0 + 0.001, t1))
        if c:
            play["on"] = False
            play["t"] = t
            _set_viewport_from_time(t)

        c, spd = imgui.SliderFloat("Speed##pb", play["speed"], 0.1, 5.0)
        if c:
            play["speed"] = spd
        c, lp = imgui.Checkbox("Loop##pb", play["loop"])
        if c:
            play["loop"] = lp

        imgui.Separator()
        c, sp = imgui.Checkbox("Spline Interp (Catmull-Rom + Squad)##pb",
                               interp_cfg["spline"])
        if c:
            interp_cfg["spline"] = sp

        # Snap-to-keyframe buttons
        if imgui.Button("<< Prev KF##pb"):
            play["on"] = False
            for i in range(len(keyframes) - 1, -1, -1):
                if keyframes[i].time < play["t"] - 1e-4:
                    play["t"] = keyframes[i].time
                    st["sel"] = i
                    _kf_to_sliders(i)
                    _refresh_viewport()
                    break
        imgui.SameLine()
        if imgui.Button("Next KF >>##pb"):
            play["on"] = False
            for i in range(len(keyframes)):
                if keyframes[i].time > play["t"] + 1e-4:
                    play["t"] = keyframes[i].time
                    st["sel"] = i
                    _kf_to_sliders(i)
                    _refresh_viewport()
                    break

        imgui.TreePop()

    def _panel_ghost():
        if not imgui.TreeNode("Ghost Display##gh"):
            return
        c, gv = imgui.Checkbox("Show adjacent KF ghost##gh", ghost_cfg["on"])
        if c:
            ghost_cfg["on"] = gv
            _refresh_ghosts()
        imgui.TreePop()

    # ---- Main loop ----

    def on_update():
        imgui.Text("=== Cook Keyframe Editor ===")
        imgui.TextColored((0.6, 1.0, 0.6, 1.0), "Hand-key wok-toss sequences")
        imgui.Separator()

        _panel_file()
        imgui.Separator()
        _panel_kf_list()
        imgui.Separator()
        _panel_time()
        imgui.Separator()
        _panel_entity("pan")
        imgui.Separator()
        _panel_entity("spatula")
        imgui.Separator()
        _panel_playback()
        imgui.Separator()
        _panel_ghost()

        # Playback tick
        if play["on"] and keyframes:
            now = time.monotonic()
            dt_wall = now - play["wall"]
            play["wall"] = now
            play["t"] += dt_wall * play["speed"]

            t0, t1 = keyframes[0].time, keyframes[-1].time
            if play["t"] > t1:
                if play["loop"]:
                    dur = max(t1 - t0, 1e-6)
                    play["t"] = t0 + (play["t"] - t0) % dur
                else:
                    play["t"] = t1
                    play["on"] = False
            _set_viewport_from_time(play["t"])

        if not play["on"]:
            _refresh_ghosts()

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
