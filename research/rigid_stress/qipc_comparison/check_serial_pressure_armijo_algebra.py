"""Check the serial pressure energy difference against exact-binary-input Decimal oracles.

Scientific execution requires Slurm rtx-mid/image 1_26. This independent scalar
algebra audit uses no Genesis, NumPy, pressure-fit oracle or production output
as an input. It does not change the objective, gradient, line-search threshold
or final force/moment gates.
"""

import argparse
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    weight: float
    coordinates: tuple[float, float, float]


@dataclass(frozen=True)
class Case:
    name: str
    coefficient: tuple[float, float, float]
    step: tuple[float, float, float]
    fraction: float
    samples: tuple[Sample, ...]


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def _decimal_dot(first, second):
    return sum((Decimal.from_float(a) * Decimal.from_float(b) for a, b in zip(first, second)), Decimal(0))


def _gamma(count):
    relative = Decimal(count) * (Decimal(2) ** -53)
    if relative >= 1:
        raise ArithmeticError("The finite-operation floating error bound is not applicable.")
    return relative / (1 - relative)


def _fixtures():
    for count in (128, 1024, 4096, 16384):
        raw = [1.0 + (index * 17 % 97) / 101.0 for index in range(count // 4)]
        normalizer = 4.0 * math.fsum(raw)
        samples = []
        for index, weight in enumerate(raw):
            x, y = (1 + index % 31) / 37.0, (1 + index * 7 % 29) / 41.0
            for first, second in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                samples.append(Sample(weight / normalizer, (1.0, first * x, second * y)))
        random.Random(731 + count).shuffle(samples)
        samples = tuple(samples)
        intercept = 1.0 / math.fsum(sample.weight for sample in samples)
        for exponent in (20, 22, 24, 26):
            size = 2.0**-exponent
            for direction in ((1, 0, 0), (0, 1, -1), (1, 1, -1), (-1, 1, 1)):
                step = tuple(size * value for value in direction)
                yield Case(
                    f"near-stationary-n{count}-e{exponent}-d{direction}",
                    (intercept + step[0], step[1], step[2]),
                    step,
                    1.0,
                    samples,
                )
    corners = tuple(
        Sample(weight, (1.0, x, y))
        for weight, (x, y) in zip((0.1, 0.2, 0.3, 0.4), ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)))
    )
    yield Case("all-four-active-transitions", (0.0, 0.75, 0.25), (0.0, 0.5, 1.0), 1.0, corners)
    yield Case("reverse-active-transitions", (0.0, 0.25, -0.75), (0.0, -0.5, -1.0), 1.0, corners)
    yield Case("fractional-crossing", (0.0, 0.75, 0.25), (0.0, 0.5, 1.0), 0.25, corners)
    yield Case("rounded-intercept-stays", (1.0, 2.0**-20, -(2.0**-21)), (2.0**-56, 2.0**-60, -(2.0**-61)), 1.0, corners)
    yield Case("rounded-candidate-identical", (1.0, 0.25, -0.125), (2.0**-56, 2.0**-57, 2.0**-58), 1.0, corners)
    zeros = (Sample(0.5, (1.0, 0.0, 0.0)), Sample(0.5, (1.0, -1.0, 0.0)))
    yield Case("zero-to-positive-boundary", (0.0, 1.0, 0.0), (-(2.0**-50), 0.0, 0.0), 1.0, zeros)
    yield Case("positive-to-zero-boundary", (2.0**-50, 1.0, 0.0), (2.0**-50, 0.0, 0.0), 1.0, zeros)


def _evaluate(case):
    coefficient = case.coefficient
    candidate = tuple(value - case.fraction * step for value, step in zip(coefficient, case.step))
    difference = tuple(new - old for new, old in zip(candidate, coefficient))
    exact_difference = tuple(
        Decimal.from_float(new) - Decimal.from_float(old) for new, old in zip(candidate, coefficient)
    )
    gradient = [-1.0, 0.0, 0.0]
    old_objective, new_objective = coefficient[0], candidate[0]
    stable_gain = difference[0]
    exact_old, exact_new = Decimal.from_float(coefficient[0]), Decimal.from_float(candidate[0])
    identity = exact_difference[0]
    bound = abs(Decimal.from_float(difference[0]) - exact_difference[0])
    absolute_sum = abs(Decimal.from_float(difference[0]))
    transitions = {name: 0 for name in ("both_active", "both_inactive", "activation", "deactivation")}
    for sample in case.samples:
        coordinates, weight = sample.coordinates, sample.weight
        old = max(0.0, _dot(coefficient, coordinates))
        new = max(0.0, _dot(candidate, coordinates))
        exact_old_raw, exact_new_raw = _decimal_dot(coefficient, coordinates), _decimal_dot(candidate, coordinates)
        exact_old_profile, exact_new_profile = max(Decimal(0), exact_old_raw), max(Decimal(0), exact_new_raw)
        delta_profile = new - old
        old_error = _gamma(6) * sum(
            abs(Decimal.from_float(a) * Decimal.from_float(b)) for a, b in zip(coefficient, coordinates)
        )
        new_error = _gamma(6) * sum(
            abs(Decimal.from_float(a) * Decimal.from_float(b)) for a, b in zip(candidate, coordinates)
        )
        if old > 0.0 and new > 0.0:
            transitions["both_active"] += 1
            delta_profile = _dot(difference, coordinates)
            delta_error = sum(
                abs(Decimal.from_float(value) - exact_value) * abs(Decimal.from_float(coordinate))
                for value, exact_value, coordinate in zip(difference, exact_difference, coordinates)
            )
            delta_error += _gamma(6) * sum(
                abs(Decimal.from_float(a) * Decimal.from_float(b)) for a, b in zip(difference, coordinates)
            )
            # A rounded positive dot can lie across an exact zero. Account for
            # that clipping discrepancy instead of assuming the active sets agree.
            delta_error += abs((exact_new_raw - exact_old_raw) - (exact_new_profile - exact_old_profile))
        else:
            transitions["both_inactive" if new == old == 0.0 else "activation" if new > 0 else "deactivation"] += 1
            delta_error = old_error + new_error
        profile_sum = new + old
        sum_error = old_error + new_error + _gamma(1) * (abs(Decimal.from_float(new)) + abs(Decimal.from_float(old)))
        dp, total, w = Decimal.from_float(delta_profile), Decimal.from_float(profile_sum), Decimal.from_float(weight)
        term = -0.5 * weight * delta_profile * profile_sum
        stable_gain += term
        bound += abs(w) / 2 * (abs(dp) * sum_error + abs(total) * delta_error + delta_error * sum_error)
        bound += _gamma(3) * abs(w * dp * total) / 2
        absolute_sum += abs(Decimal.from_float(term))
        old_objective -= 0.5 * weight * old * old
        new_objective -= 0.5 * weight * new * new
        for index in range(3):
            gradient[index] += weight * old * coordinates[index]
        exact_old -= w * exact_old_profile * exact_old_profile / 2
        exact_new -= w * exact_new_profile * exact_new_profile / 2
        identity -= w * (exact_new_profile - exact_old_profile) * (exact_new_profile + exact_old_profile) / 2
    bound += _gamma(len(case.samples)) * absolute_sum
    exact_gain = exact_new - exact_old
    old_gain = new_objective - old_objective
    slope = _dot(gradient, case.step)
    threshold = 1e-4 * case.fraction * slope - 1e-15
    reference_accept = exact_gain >= Decimal.from_float(threshold)
    new_accept = stable_gain >= threshold
    old_accept = new_objective >= old_objective + 1e-4 * case.fraction * slope - 1e-15
    new_error = abs(Decimal.from_float(stable_gain) - exact_gain)
    certified = abs(exact_gain - Decimal.from_float(threshold)) > bound
    identity_passed = exact_gain == identity
    bound_passed = new_error <= bound
    decision_passed = not certified or new_accept == reference_accept
    return {
        "name": case.name,
        "passed": identity_passed and bound_passed and decision_passed,
        "samples": len(case.samples),
        "coefficient_hex": [value.hex() for value in coefficient],
        "candidate_hex": [value.hex() for value in candidate],
        "actual_delta_hex": [value.hex() for value in difference],
        "nominal_delta_hex": [(-case.fraction * value).hex() for value in case.step],
        "actual_delta_differs_from_nominal": any(
            value != -case.fraction * step for value, step in zip(difference, case.step)
        ),
        "active_transitions": transitions,
        "exact_binary_input_gain": str(exact_gain),
        "exact_identity_gain": str(identity),
        "decimal_identity_exact": identity_passed,
        "stable_float_gain": stable_gain,
        "old_float_gain": old_gain,
        "stable_absolute_error": str(new_error),
        "old_absolute_error": str(abs(Decimal.from_float(old_gain) - exact_gain)),
        "stable_error_bound": str(bound),
        "stable_within_derived_bound": bound_passed,
        "float_gradient_norm": math.sqrt(_dot(gradient, gradient)),
        "unchanged_nominal_armijo_slope": slope,
        "unchanged_float_armijo_threshold": threshold,
        "reference_armijo_accept": reference_accept,
        "stable_armijo_accept": new_accept,
        "old_armijo_accept": old_accept,
        "stable_decision_certified": certified,
        "stable_decision_correct": decision_passed,
        "old_wrong_reject": reference_accept and not old_accept,
        "old_wrong_accept": not reference_accept and old_accept,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-source", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    if os.environ.get("GENESIS_IMAGE_VER") != "1_26":
        parser.error("Use the declared image 1_26.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh output under the project data directory.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with localcontext() as context:
        context.prec = 400
        rows = [_evaluate(case) for case in _fixtures()]
    source = args.engine_source.resolve()
    contact = source / "genesis/engine/solvers/rigid/stress/contact.py"
    wrong_rejects = [row["name"] for row in rows if row["old_wrong_reject"]]
    result = {
        "complete": True,
        "passed": all(row["passed"] for row in rows),
        "checks": len(rows),
        "failed_checks": [row["name"] for row in rows if not row["passed"]],
        "decimal_precision": 400,
        "old_wrong_reject_count": len(wrong_rejects),
        "old_wrong_reject_witnesses": wrong_rejects,
        "old_wrong_accept_count": sum(row["old_wrong_accept"] for row in rows),
        "rounded_endpoint_controls": [row["name"] for row in rows if row["actual_delta_differs_from_nominal"]],
        "rows": rows,
        "identity": {
            "host": socket.gethostname(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "image": os.environ.get("GENESIS_IMAGE_VER"),
        },
        "engine_source": str(source),
        "engine_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "engine_contact_sha256": hashlib.sha256(contact.read_bytes()).hexdigest(),
        "algebra_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formula": "Q(c)=c0-.5*sum(w*max(c.x,0)^2); actualdelta=candidate-c; DeltaQ=actualdelta0-.5*sum(w*(pnew-pold)*(pnew+pold)). Both active uses actualdelta.x; crossing/inactive uses clipped-profile difference.",
        "error_bound": "u=2^-53, gamma(n)=nu/(1-nu). Bound actual coefficient subtraction, dot/positive-part errors, active-classification discrepancy, three product roundings and N-term final accumulation by triangle inequality. Normal finite nonunderflowing fixtures; no empirical multiplier or production tolerance change.",
        "scope": "Independent scalar algebra and cancellation control for exact real Q at actual binary64 coefficient/candidate/input endpoints. Correcting evaluation does not seek bitwise equality to the noisy old objectives. No native production fit/wrench/K qualification is inferred; unchanged gates and actual-source replay are separate.",
        "seconds": time.perf_counter() - started,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise ArithmeticError("The independent pressure energy-difference algebra or derived error bound failed.")


if __name__ == "__main__":
    main()
