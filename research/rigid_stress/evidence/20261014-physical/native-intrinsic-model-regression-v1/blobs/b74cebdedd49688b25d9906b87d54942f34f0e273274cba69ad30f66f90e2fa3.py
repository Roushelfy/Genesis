"""Trial one serial native graph for the complete rigid stress observation."""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import genesis as gs
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from research.rigid_stress.probe_fused_contacts import generate as contacts_generate
from research.rigid_stress.probe_fused_elastic import generate as elastic_generate


def generate(directory):
    contacts_generate(directory)
    elastic_generate(directory)
    root = Path("genesis/engine/solvers/rigid/stress")
    blocks = [
        """import quadrants as qd
import genesis as gs
from genesis.utils import array_class, geom
from genesis.utils.array_class import ErrorCode
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressState, StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.engine.solvers.rigid.stress.inverse import StressInverseInfo
from genesis.engine.solvers.rigid.stress.factor import StressFactorInfo
from fused_contacts_generated import (func_anchor, func_pack_contacts, func_pressure_warp,
    func_pressure_correct_warp, func_refine_contacts, func_scatter_warp)
from fused_elastic_generated import (func_balance, func_direct_init, func_full_residual,
    func_apply_inverse, func_solve_cooperative, func_peak)
"""
    ]
    hashes = {}
    for filename, names in (
        ("association.py", ("kernel_associate",)),
        ("recovery.py", ("kernel_begin_step", "kernel_accept")),
        ("surface_inverse.py", ("kernel_surface_pack", "kernel_surface_apply_packed")),
    ):
        source = (root / filename).read_text()
        hashes[filename] = hashlib.sha256(source.encode()).hexdigest()
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        for name in names:
            node = functions[name]
            node.name = name.replace("kernel_", "func_")
            node.decorator_list = [ast.Attribute(value=ast.Name(id="qd", ctx=ast.Load()), attr="func", ctx=ast.Load())]
            blocks.append(ast.unparse(node))
    blocks.append("""
@qd.kernel(graph=True)
def kernel_pipeline(young: float, poisson: float, tolerance: float, absolute: float, epsilon: float,
                    link: int, offsets_pos: qd.types.ndarray(), offsets_quat: qd.types.ndarray(),
                    omega: qd.Tensor, dyn: array_class.DynState, collider: array_class.ColliderState,
                    state: StressState, contacts: StressContactState, info: StressInfo,
                    surface: StressSurfaceInfo, boundary: StressSurfaceInverseInfo,
                    inverse: StressInverseInfo, factor: StressFactorInfo, errno: qd.Tensor,
                    batch_offsets: qd.template(), first: qd.template(), enable: bool, n_nodes: qd.template()):
    if qd.static(first):
        func_begin_step(state)
    func_associate(link, offsets_pos, offsets_quat, omega, dyn, contacts, collider,
                   batch_offsets, enable, state.step_valid, errno)
    func_anchor(epsilon, contacts, surface)
    func_pack_contacts(contacts)
    func_pressure_warp(contacts, surface, False)
    func_pressure_correct_warp(contacts, surface)
    func_refine_contacts(contacts)
    func_pressure_warp(contacts, surface, True)
    func_pressure_correct_warp(contacts, surface)
    func_scatter_warp(contacts, state, info, surface, True)
    func_balance(omega, state, info)
    func_direct_init(state)
    func_surface_pack(state, boundary)
    func_surface_apply_packed(young, omega, state, boundary)
    func_full_residual(young, tolerance, absolute, state, info, False)
    for _ in qd.static(range(2)):
        func_apply_inverse(young, state, inverse, True, False)
        func_full_residual(young, tolerance, absolute, state, info, True)
    func_solve_cooperative(young, state, info, factor, n_nodes, True)
    func_full_residual(young, tolerance, absolute, state, info, True)
    func_peak(young, poisson, state, info, True)
    func_accept(state, contacts, errno)
""")
    destination = directory / "pipeline_graph_generated.py"
    destination.write_text("\n\n".join(blocks) + "\n")
    spec = importlib.util.spec_from_file_location("pipeline_graph_generated", destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kernel_pipeline, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trial-mode", choices=("native", "fused"), required=True)
    args, remaining = parser.parse_known_args()
    original_command = [sys.executable, *sys.argv]
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    if args.trial_mode == "fused":
        kernel, hashes = generate(output.parent)
        original = RigidStressRecovery.recover

        def recover(self, i_substep):
            entry = self.links[0]
            model, options, solver = entry.model, entry.link.stress_options, self.solver
            if (
                gs.backend == gs.cuda
                and len(self.links) == 1
                and model.surface_inverse is not None
                and options.inverse_corrections == 2
                and options.cooperative_pressure
                and options.cooperative_scatter
                and options.cooperative_solve
                and options.cached_peak
                and options.cached_face_bounds
                and options.packed_surface_loads
                and entry.history is None
            ):
                kernel(
                    options.young,
                    options.poisson,
                    options.tolerance,
                    options.absolute_tolerance,
                    self.source_epsilon,
                    entry.link.idx,
                    solver._links_offset_pos,
                    solver._links_offset_quat,
                    entry.omega,
                    solver.dyn_state,
                    solver.collider.collider_state,
                    entry.state,
                    entry.contacts,
                    model.info,
                    entry.surface.info,
                    model.surface_inverse.info,
                    model.inverse.info,
                    model.factor.info,
                    solver._errno,
                    solver._links_offset_quat.ndim == 3,
                    i_substep == 0,
                    not solver._disable_constraint,
                    model.info.vertices.shape[0],
                )
            else:
                original(self, i_substep)

        RigidStressRecovery.recover = recover
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()
    result = json.loads(output.read_text())
    result["development_pipeline_graph_trial"] = args.trial_mode
    result["trial_native_source_sha256"] = hashes
    result["trial_command"] = original_command
    result["trial_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
