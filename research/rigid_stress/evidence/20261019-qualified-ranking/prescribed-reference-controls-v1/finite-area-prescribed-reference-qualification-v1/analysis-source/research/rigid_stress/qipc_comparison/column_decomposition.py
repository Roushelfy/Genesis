"""Integrate a three-field tensor identity without subtracting scalar error norms."""

import numpy as np


def products(values, weights):
    """Return weighted energies for (candidate, same-mesh control, fine reference)."""
    first = values[0] - values[1]
    second = values[1] - values[2]
    total = values[0] - values[2]
    return np.array(
        [
            np.einsum("qij,qij,q->", first, first, weights),
            np.einsum("qij,qij,q->", second, second, weights),
            np.einsum("qij,qij,q->", first, second, weights),
            np.einsum("qij,qij,q->", total, total, weights),
            np.einsum("qij,qij,q->", values[2], values[2], weights),
        ]
    )


def summarize(energy, measure, unit):
    first, second, cross, total, reference = energy
    assert np.isfinite(energy).all() and min(first, second, total, reference) >= 0 and measure > 0
    scale = first + second + 2 * abs(cross) + total
    defect = abs(total - first - second - 2 * cross)
    assert defect <= 128 * np.finfo(float).eps * scale, "Integrated tensor identity failed"
    return {
        "unit": unit,
        "first_difference_rms": float(np.sqrt(first / measure)),
        "second_difference_rms": float(np.sqrt(second / measure)),
        "total_difference_rms": float(np.sqrt(total / measure)),
        "reference_rms": float(np.sqrt(reference / measure)),
        "first_relative_l2": float(np.sqrt(first / reference)) if reference > 0 else None,
        "second_relative_l2": float(np.sqrt(second / reference)) if reference > 0 else None,
        "total_relative_l2": float(np.sqrt(total / reference)) if reference > 0 else None,
        "twice_cross_mean_square": float(2 * cross / measure),
        "cross_correlation": float(cross / np.sqrt(first * second)) if first > 0 and second > 0 else None,
        "identity_relative_defect": float(defect / scale) if scale > 0 else 0.0,
        "zero_reference_norm": bool(reference == 0),
    }


def integrate(fields, basis, area, angular_weights, depth_nodes, depth_weights):
    """Use one union of all three partitions and the fine reference's physical volume."""
    assert len(fields) == 3 and fields[-1].volumes is not None
    breaks = np.sort(np.concatenate([field.breaks() for field in fields]), axis=0)
    np.testing.assert_allclose(np.diff(breaks, axis=0).sum(axis=0), 1, rtol=0, atol=1e-13)
    energy = np.zeros(5)
    measure = 0.0
    membrane = np.zeros((3, len(area), 2, 2))
    bending = np.zeros_like(membrane)
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        width = upper - lower
        for node, weight in zip(depth_nodes, depth_weights):
            depth = (lower + upper + node * width) / 2
            values = np.stack([field.sample(depth, basis) for field in fields])
            volume_weight = width / 2 * angular_weights * fields[-1].volume_density(depth) * weight
            energy += products(values, volume_weight)
            measure += volume_weight.sum()
            membrane += values * (width / 2 * 0.0005 * weight)[None, :, None, None]
            bending += values * (width / 2 * (0.5 - depth) * 0.0005**2 * weight)[None, :, None, None]
    np.testing.assert_allclose(measure, fields[-1].volumes.sum(), rtol=1e-9, atol=1e-18)
    metrics = {"tangent": summarize(energy, measure, "Pa")}
    metrics["membrane"] = summarize(products(membrane, area), area.sum(), "N/m")
    metrics["bending"] = summarize(products(bending, area), area.sum(), "N")
    for name, depth in (("outer", 0.0), ("inner", 1.0)):
        values = np.stack([field.sample(np.full(len(area), depth), basis) for field in fields])
        metrics[name + "_tangent"] = summarize(products(values, area), area.sum(), "Pa")
    return {
        "metrics": metrics,
        "passed_tensor_identity": True,
        "scope": "Pointwise signed tensors satisfy (field0-field2)=(field0-field1)+(field1-field2). "
        "All three differences and their signed cross product are integrated directly on one union "
        "of material partitions and the full reference volume/area. The squared total equals the "
        "two squared contributions plus twice their cross product; their RMS magnitudes are not "
        "additive error budgets. All relative values use the same fine-reference norm, with null "
        "relative values at an exactly zero reference and no arbitrary floor. Assigning physical "
        "meaning to each difference requires the separately verified field sources and admissions.",
    }
