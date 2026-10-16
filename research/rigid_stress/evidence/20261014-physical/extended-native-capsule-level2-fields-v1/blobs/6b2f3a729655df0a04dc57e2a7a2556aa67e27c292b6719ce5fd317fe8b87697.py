"""Reconstruct the authored macro-step control inputs from an immutable condition bank."""

import numpy as np


def control_at_tick(conditions, tick, env=0):
    delay = int(conditions["delays"][env])
    phase = ((tick - delay) % 600) / 599 if tick >= delay else 0.0

    def smooth(value):
        x = np.clip(value, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    approach, lift, slide = smooth(phase / 0.15), smooth((phase - 0.3) / 0.2), smooth((phase - 0.6) / 0.15)
    initial, pick, raised, shifted = [
        np.asarray(conditions[key][env], dtype=float)
        for key in ["initial_joints", "pick_joints", "lift_joints", "slide_joints"]
    ]
    target = initial * (1.0 - approach) + pick * approach
    target += (raised - pick) * lift + (shifted - raised) * slide
    # Original Torch scalar where outputs and native controller buffers are float32.
    grip = float(np.float32(0.04 if phase < 0.15 or phase > 0.9 else 0.018))
    limit = float(np.float32(0.055 if 0.7 <= phase < 0.76 else 4.0))
    target[7:] = grip
    radius = float(conditions["radius"][env] * (1.0 + 0.15 * np.sin(2.0 * np.pi * phase)))
    return {"phase": phase, "target": target, "finger_cap_N": limit, "native_radius_m": radius}
