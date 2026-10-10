"""Generate a reproducible Quadrants graph fusion trial from native pass bodies."""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy


def generate(directory):
    root = Path("genesis/engine/solvers/rigid/stress")
    selection = {
        "solve.py": ("kernel_balance", "kernel_direct_init", "kernel_full_residual", "kernel_peak_impl"),
        "surface_inverse.py": ("kernel_surface_apply",),
        "inverse.py": ("kernel_apply_inverse",),
        "factor.py": ("kernel_solve_cooperative",),
    }
    header = """import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressState, StressInfo
from genesis.engine.solvers.rigid.stress.factor import StressFactorInfo
from genesis.engine.solvers.rigid.stress.inverse import StressInverseInfo
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.engine.solvers.rigid.stress.operators import func_shape_gradient
"""
    blocks = [header]
    hashes = {}
    for filename, names in selection.items():
        content = (root / filename).read_text()
        hashes[filename] = hashlib.sha256(content.encode()).hexdigest()
        found = {node.name: node for node in ast.parse(content).body if isinstance(node, ast.FunctionDef)}
        for name in names:
            node = found[name]
            node.name = name.replace("kernel_", "func_").replace("peak_impl", "peak")
            node.decorator_list = [ast.Attribute(value=ast.Name(id="qd", ctx=ast.Load()), attr="func", ctx=ast.Load())]
            blocks.append(ast.unparse(node))
    blocks.append("""
@qd.kernel(graph=True)
def kernel_fused(young: float, poisson: float, tolerance: float, absolute_tolerance: float,
                 omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo,
                 surface_inverse_info: StressSurfaceInverseInfo, inverse_info: StressInverseInfo,
                 factor_info: StressFactorInfo, n_nodes: qd.template()):
    func_balance(omega, stress_state, stress_info)
    func_direct_init(stress_state)
    func_surface_apply(young, omega, stress_state, surface_inverse_info)
    func_full_residual(young, tolerance, absolute_tolerance, stress_state, stress_info, False)
    for _ in qd.static(range(2)):
        func_apply_inverse(young, stress_state, inverse_info, True, False)
        func_full_residual(young, tolerance, absolute_tolerance, stress_state, stress_info, True)
    func_solve_cooperative(young, stress_state, stress_info, factor_info, n_nodes, True)
    func_full_residual(young, tolerance, absolute_tolerance, stress_state, stress_info, True)
    func_peak(young, poisson, stress_state, stress_info, True)
""")
    parallel_source = (
        blocks[-1]
        .replace("def kernel_fused", "def kernel_parallel")
        .replace(
            "    func_balance(omega, stress_state, stress_info)\n    func_direct_init(stress_state)\n    func_surface_apply(young, omega, stress_state, surface_inverse_info)",
            "    with qd.graph.parallel_context():\n        with qd.graph.parallel():\n            func_balance(omega, stress_state, stress_info)\n        with qd.graph.parallel():\n            func_surface_apply(young, omega, stress_state, surface_inverse_info)\n    func_direct_init(stress_state)",
        )
    )
    blocks.append(parallel_source)
    destination = directory / "fused_elastic_generated.py"
    destination.write_text("\n\n".join(blocks) + "\n")
    spec = importlib.util.spec_from_file_location("fused_elastic_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_fused, module.kernel_parallel, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused, parallel, hashes = generate(args.output.parent)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions(mesh=Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")))
    state = model.create_state(args.envs)
    random = np.random.default_rng(10917)
    force = np.zeros((model.info.vertices.shape[0], args.envs, 3))
    nodes = qd_to_numpy(model.surface_inverse.info.nodes)
    force[nodes] = random.normal(size=(len(nodes), args.envs, 3)) * 0.01
    state.force.from_numpy(force)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    omega.from_numpy(random.normal(size=(args.envs, 3)) * 2)
    model.recover(omega, state, surface_load=True)
    expected = qd_to_numpy(state.displacement, copy=True)
    expected_peak = qd_to_numpy(state.peak, copy=True)
    separate = lambda: model.recover(omega, state, surface_load=True)
    combined = lambda: fused(
        model.options.young,
        model.options.poisson,
        model.options.tolerance,
        model.options.absolute_tolerance,
        omega,
        state,
        model.info,
        model.surface_inverse.info,
        model.inverse.info,
        model.factor.info,
        model.info.vertices.shape[0],
    )
    concurrent = lambda: parallel(
        model.options.young,
        model.options.poisson,
        model.options.tolerance,
        model.options.absolute_tolerance,
        omega,
        state,
        model.info,
        model.surface_inverse.info,
        model.inverse.info,
        model.factor.info,
        model.info.vertices.shape[0],
    )
    reports = []
    for name, operation in (("separate", separate), ("combined", combined), ("combined-parallel", concurrent)):
        operation()
        actual = qd_to_numpy(state.displacement, copy=True)
        peak = qd_to_numpy(state.peak, copy=True)
        assert qd_to_numpy(state.valid).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-15)
        np.testing.assert_allclose(peak, expected_peak, rtol=1e-4, atol=1e-3)
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        reports.append(
            {
                "name": name,
                "milliseconds": 1e3 * (time.perf_counter() - started) / args.repetitions,
                "peak_error_Pa": float(abs(peak - expected_peak).max()),
                "complete_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    args.output.write_text(
        json.dumps({"envs": args.envs, "source_hashes": hashes, "reports": reports}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
