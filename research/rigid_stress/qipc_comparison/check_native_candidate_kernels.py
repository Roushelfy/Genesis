"""Small native module control before complete Franka candidate validation."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.atlas import StressAtlas, StressAtlasState
from genesis.engine.solvers.rigid.stress.atlas_output import StressAtlasOutput
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_pack_contacts
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.history import StressHistory
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.response import StressResponse, StressResponseInventory
from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace, kernel_scatter_faces
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .audit_gpu_candidates import per_environment
from .check_native_atlas import cpu_control, cpu_operator, relative


@dataclass
class Control:
    model: StressModel
    state: StressState
    omega: qd.Tensor
    options: gs.options.RigidStressOptions


@qd.kernel
def kernel_load(
    step: int, omega: qd.Tensor, state: StressState, atlas: StressAtlasState, boundary: StressSurfaceInverseInfo
):
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.active.shape[0]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_n, i_g, i_b in qd.ndrange(atlas.force.shape[0], atlas.force.shape[1], state.active.shape[0]):
        atlas.force[i_n, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_b in range(state.active.shape[0]):
        scale = qd.select(step == 3, 0.0, 0.001 * (step + 1) * (i_b + 1))
        first = scale * qd.Vector([1.0, 2.0, -3.0])
        second = scale * qd.Vector([-0.4, 0.3, 0.1])
        first_node, second_node = boundary.nodes[0], boundary.nodes[1]
        state.force[first_node, i_b] = first
        state.force[second_node, i_b] = second
        for i_g in range(1, 4):
            fraction = qd.select(i_g == 1, 0.2, qd.select(i_g == 2, 0.3, 0.5))
            atlas.force[0, i_g, i_b] = fraction * first
            atlas.force[1, i_g, i_b] = fraction * second
        omega[i_b] = qd.Vector([0.2, 0.3, 0.4]) * qd.select(step == 3, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--atlas-output", choices=("merged", "independent"), default="merged")
    parser.add_argument("--atlas-partition", choices=("direct", "coarse_local"), default="direct")
    parser.add_argument("--response-precision", choices=("64", "32"), default="64")
    parser.add_argument("--response-accumulation", choices=("64", "32"), default="64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    result = dict(identity=runtime_identity(), complete=False, rows=[])
    write_json(args.output / "result.json", result)
    asset = Path(gs.__file__).parent.parent / "examples/rigid/assets/hollow_egg/level1"
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset / ("shell.npz" if args.stress_backend == "shell" else "elastic.npz"),
        backend=args.stress_backend,
        output_mode="full",
        quadrature=16,
        atlas_regions=6,
        stress_response="none" if args.atlas_output == "independent" else "sparse",
        response_precision=args.response_precision,
        response_accumulation=args.response_accumulation,
    )
    model = ShellModel(options) if args.stress_backend == "shell" else StressModel(options)
    state = model.create_state(2, "full")
    contacts = create_contacts(1, 2, 0.001, hertz_profile=True)
    contacts.valid.fill(False)
    contacts.force.fill(0)
    surface = StressSurface(16, model.load_info)
    atlas = StressAtlas(model, contacts, 6, 0.003, args.atlas_partition)
    if args.atlas_output == "independent":
        atlas.output = StressAtlasOutput(model, 2, 7)
    history = StressHistory(model.info.vertices.shape[0], 2, 2)
    response = StressResponse(model.output_response, 2) if model.output_response is not None else None
    omega = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    workspace = create_scatter_workspace(2, 32)
    kernel_pack_contacts(contacts)
    atlas.prepare(contacts, model.load_info, surface.info, 0.003)
    kernel_scatter_faces(contacts, state, model.load_info, surface.info, workspace, True, atlas=atlas.scatter)
    reference = options.model_copy(update={"atlas_regions": 0, "stress_response": "none"})
    fields = (("displacement", state.displacement), ("stress_tensor", state.stress_tensor), ("peak", state.peak))
    if args.stress_backend == "shell":
        fields += (
            ("membrane", state.membrane_resultant),
            ("bending", state.bending_resultant),
            ("shear", state.transverse_resultant),
        )
    for step in range(4):
        kernel_load(step, omega, state, atlas.state, model.surface_inverse.info)
        model.recover(omega, state, options, history, True, atlas, response)
        assert qd_to_numpy(state.valid).all()
        history.append(state)
        candidate = {name: qd_to_numpy(array, transpose=True) for name, array in fields}
        hits = qd_to_numpy(history.state.hit)
        model.recover(omega, state, reference, surface_load=True)
        errors = {
            name: float(per_environment(candidate[name], qd_to_numpy(array, transpose=True)).max())
            for name, array in fields
        }
        result["rows"].append(dict(step=step, history_hits=hits.tolist(), errors=errors))
        write_json(args.output / "result.json", result)
        exact_output = args.response_precision == args.response_accumulation == "64"
        assert errors["displacement"] < 1e-8
        assert max(errors.values()) < (1e-8 if exact_output else 1e-4)
        if step == 0:
            result["cpu_control"] = cpu_control(Control(model, state, omega, options))
        if step == 2:
            assert hits.all()
    if atlas.coarse is not None:
        matrix = cpu_operator(model.info).matrix
        basis = qd_to_numpy(atlas.coarse.info.basis).transpose(0, 2, 1).reshape((matrix.shape[0], -1))
        projection = qd_to_numpy(atlas.coarse.info.projection)
        native = (
            qd_to_numpy(model.surface_inverse.info.force_blocks).transpose(0, 2, 1, 3).reshape((matrix.shape[0], -1))
        )
        residual = qd_to_numpy(atlas.response.force_blocks).transpose(0, 2, 1, 3).reshape(native.shape)
        result["coarse_reconstruction_error"] = relative(basis @ projection + residual, native)
        result["coarse_galerkin_error"] = relative(basis.T @ matrix @ (native - residual), basis.T @ matrix @ native)
        assert result["coarse_reconstruction_error"] < 1e-12
        assert result["coarse_galerkin_error"] < 1e-8
    if response is None:
        response = StressResponse(StressResponseInventory(model, options), 2)
    if exact_output:
        maximum = model.create_state(2)
        kernel_load(0, omega, maximum, atlas.state, model.surface_inverse.info)
        model.recover(
            omega,
            maximum,
            reference.model_copy(update={"stress_response": "screen"}),
            surface_load=True,
            response=response,
        )
        candidate_peak = qd_to_numpy(maximum.peak)
        model.recover(omega, maximum, reference, surface_load=True)
        result["screen_error"] = float(per_environment(candidate_peak, qd_to_numpy(maximum.peak)).max())
        assert result["screen_error"] < 1e-8
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
