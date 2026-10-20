"""Reintegrate complete retained fields and form explicit conditional error intervals."""

import math

import numpy as np


def conditional_interval(error, reference_envelope, measurement_change=0.0):
    """Triangle-inequality interval, conditional on the declared uncertainty bounds."""
    if not all(math.isfinite(value) for value in (error, reference_envelope, measurement_change)):
        raise ValueError("Finite error and uncertainty are required")
    if error < 0 or not 0 <= reference_envelope < 1 or measurement_change < 0:
        raise ValueError("Nonnegative error/measurement change and reference uncertainty below one are required")
    return (
        max(error - measurement_change - reference_envelope, 0) / (1 + reference_envelope),
        (error + measurement_change + reference_envelope) / (1 - reference_envelope),
    )


def uncertainty_product(components):
    values = list(components)
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Finite nonnegative observed changes are required")
    return math.prod(1 + value for value in values) - 1


def pair_relation(first, second):
    if first[1] < second[0]:
        return "first_lower"
    if second[1] < first[0]:
        return "second_lower"
    return "unresolved"


def measure_case(case, column_path, surface_path=None, peak_case=None, *, require_surfaces=True):
    """Use full physical-volume and area weights without a near-zero normalization floor.

    Legacy controls may explicitly omit surface arrays; the final ranking
    requires them and independently recomputes both fiber norms.
    """
    assert len(case["raw"]) == 2 and len(case["adjacent_comparisons"]) == 1
    comparison = case["adjacent_comparisons"][0]
    assert comparison["integration_measure_unit"] == "m3" and comparison["integration_measure"] > 0
    rows, checks = [], {}

    def add(metric, error, reference, unit, reported):
        assert np.isfinite([error, reference, reported]).all() and error >= 0 and reference > 0 and reported >= 0
        relative = error / reference
        np.testing.assert_allclose(relative, reported, rtol=1e-10, atol=1e-14)
        rows.append(
            {
                "metric": metric,
                "absolute_error": float(error),
                "reference_rms_or_peak": float(reference),
                "unit": unit,
                "relative_error": float(relative),
            }
        )
        checks[metric + "/absolute_relative_identity"] = True

    with np.load(column_path) as saved:
        area, direction, basis = (saved[key] for key in ("area_m2", "direction", "basis"))
        assert area.ndim == 1 and np.isfinite(area).all() and (area > 0).all()
        for metric, key, unit in (("membrane", "membrane_n_per_m", "N/m"), ("bending", "bending_n", "N")):
            values = saved[key]
            assert values.shape == (2, len(area), 2, 2) and np.isfinite(values).all()
            difference = values[0] - values[1]
            error = np.sqrt(np.einsum("qij,qij,q->", difference, difference, area) / area.sum())
            reference = np.sqrt(np.einsum("qij,qij,q->", values[1], values[1], area) / area.sum())
            add(metric, error, reference, unit, comparison[metric + "_relative_l2"])
            if metric + "_rms_error_" + ("n_per_m" if metric == "membrane" else "n") in comparison:
                np.testing.assert_allclose(
                    error,
                    comparison[metric + "_rms_error_" + ("n_per_m" if metric == "membrane" else "n")],
                    rtol=1e-10,
                    atol=0,
                )
            checks[metric + "/complete_resultant_reintegration"] = True
        error2, reference2 = (saved[key] for key in ("error_square_pa2_m3", "reference_square_pa2_m3"))
        assert error2.shape == reference2.shape == area.shape
        assert (
            np.isfinite(error2).all()
            and np.isfinite(reference2).all()
            and (error2 >= 0).all()
            and (reference2 >= 0).all()
        )
        volume = comparison["integration_measure"]
        error, reference = np.sqrt(error2.sum() / volume), np.sqrt(reference2.sum() / volume)
        add("tangent", error, reference, "Pa", comparison["tangent_relative_l2"])
        np.testing.assert_allclose(error, comparison["tangent_rms_error_pa"], rtol=1e-10, atol=0)
        checks["tangent/all_volume_columns_reintegrated"] = True
    if require_surfaces and surface_path is None:
        raise ValueError("Complete outer/inner arrays are mandatory for final errors")
    if surface_path is not None:
        with np.load(surface_path) as saved:
            np.testing.assert_array_equal(saved["direction"], direction)
            np.testing.assert_array_equal(saved["basis"], basis)
            for side in ("outer", "inner"):
                metric = side + "_tangent"
                values = saved[side + "_tangent_pa"]
                assert values.shape == (2, len(area), 2, 2) and np.isfinite(values).all()
                difference = values[0] - values[1]
                error = np.sqrt(np.einsum("qij,qij,q->", difference, difference, area) / area.sum())
                reference = np.sqrt(np.einsum("qij,qij,q->", values[1], values[1], area) / area.sum())
                add(metric, error, reference, "Pa", comparison[metric + "_relative_l2"])
                np.testing.assert_allclose(error, comparison[metric + "_rms_error_pa"], rtol=1e-10, atol=0)
                checks[metric + "/complete_surface_reintegration"] = True
    else:
        for side in ("outer", "inner"):
            metric = side + "_tangent"
            error, relative = comparison[metric + "_rms_error_pa"], comparison[metric + "_relative_l2"]
            assert relative > 0
            add(metric, error, error / relative, "Pa", relative)
    for metric in ("mises", "tension"):
        actual, reference = [row["raw_maximum_" + metric + "_pa"] for row in case["raw"]]
        add("raw_" + metric, abs(actual - reference), reference, "Pa", comparison["raw_" + metric + "_relative_change"])
    peak_case = case if peak_case is None else peak_case
    if "common_spatial_measures" in peak_case["adjacent_comparisons"][0]:
        assert [(r["mesh"], r["field"]) for r in case["raw"]] == [(r["mesh"], r["field"]) for r in peak_case["raw"]]
        peaks = peak_case["adjacent_comparisons"][0]["common_spatial_measures"]["tangent_peaks"]
        assert len(peaks) == 2 and {p["metric"] for p in peaks} == {"mises", "tension"}
        for peak in peaks:
            actual, reference = peak["actual"]["stress_pa"], peak["reference"]["stress_pa"]
            error = abs(actual - reference)
            np.testing.assert_allclose(error, peak["absolute_difference_pa"], rtol=1e-10, atol=1e-7)
            add("common_" + peak["metric"], error, reference, "Pa", peak["relative_difference"])
    return {
        "rows": rows,
        "checks": checks,
        "surface_arrays_reintegrated": surface_path is not None,
        "area_m2": float(area.sum()),
        "volume_m3": float(comparison["integration_measure"]),
        "column_count": len(area),
    }


def raw_stress_peaks(mesh_path, field_path, *, shell=False):
    """Recompute unsmoothed full3D invariants and retain actual maximizing sample locations."""
    with np.load(mesh_path) as mesh, np.load(field_path) as field:
        stress = field["cauchy_world_pa"] if shell else field["stress_pa"]
        assert stress.shape[-2:] == (3, 3) and np.isfinite(stress).all()
        np.testing.assert_allclose(stress, stress.swapaxes(-1, -2), rtol=1e-10, atol=1e-7)
        xx, yy, zz = (stress[..., i, i] for i in range(3))
        xy, xz, yz = stress[..., 0, 1], stress[..., 0, 2], stress[..., 1, 2]
        mises = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + xz**2 + yz**2))
        tension = np.linalg.eigvalsh(stress)[..., -1]
        vertices, tets = mesh["vertices"], mesh["tetrahedra"]
        if shell:
            mid, faces = mesh["mid_vertices"], mesh["mid_faces"]

        def points(indices, shape):
            if shell:
                face, fiber = np.unravel_index(indices, shape)
                tri = mid[faces[face]]
                normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                normal /= np.linalg.norm(normal, axis=1)[:, None]
                return tri.mean(axis=1) + (fiber - 1)[:, None] * 0.00025 * normal
            if stress.ndim == 3:
                return vertices[tets[indices]].mean(axis=1)
            element, corner = np.unravel_index(indices, shape)
            return vertices[tets[element, corner]]

        result = {}
        for name, magnitudes in (("mises", mises), ("tension", tension)):
            maximum = float(magnitudes.max())
            assert math.isfinite(maximum) and maximum > 0
            # Near ties affect which actual sample is reported, never the raw maximum.
            selected = np.flatnonzero(np.abs(magnitudes.reshape(-1) - maximum) <= 1e-7 + 1e-10 * abs(maximum))
            candidates = np.unique(points(selected, magnitudes.shape), axis=0)
            result[name] = {
                "stress_pa": maximum,
                "independent_peak_position_m": points(np.array([magnitudes.argmax()]), magnitudes.shape)[0].tolist(),
                "roundoff_tie_sample_positions_m": candidates.tolist(),
            }
        return result


def check_raw_receipt(raw, independent, *, require_positions=True):
    for metric in ("mises", "tension"):
        peak = independent[metric]
        np.testing.assert_allclose(raw["raw_maximum_" + metric + "_pa"], peak["stress_pa"], rtol=1e-10, atol=1e-7)
        key = "raw_" + metric + "_peak_position_m"
        if key in raw:
            point = np.asarray(raw[key])
        elif metric == "mises" and "raw_peak_position_m" in raw:
            point = np.asarray(raw["raw_peak_position_m"])
        elif require_positions:
            raise ValueError("Both raw peak locations are required")
        else:
            continue
        candidates = np.asarray(peak["roundoff_tie_sample_positions_m"])
        assert point.shape == (3,) and np.isfinite(point).all()
        assert np.linalg.norm(candidates - point, axis=1).min() <= 1e-12
