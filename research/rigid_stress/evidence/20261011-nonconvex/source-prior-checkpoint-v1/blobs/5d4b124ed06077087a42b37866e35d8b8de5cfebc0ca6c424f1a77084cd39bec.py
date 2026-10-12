"""Create an isolated source-prior dual experiment from the qualified native scalar numerical path."""

import ast
from pathlib import Path

root = Path(__file__).parent
source = Path("genesis/engine/solvers/rigid/stress/traction.py").read_text()
tree = ast.parse(source)


def expression(text):
    return ast.parse(text, mode="eval").body


class Transform(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node.name = {"func_traction_dual": "func_prior_dual", "kernel_fit_tractions_impl": "kernel_fit_source_priors", "func_source_traction": "func_prior_traction"}[node.name]
        node.args.args.append(ast.arg(arg="prior", annotation=ast.Name(id="StressSourcePriors", ctx=ast.Load())))
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "func_traction_dual":
            node.func.id = "func_prior_dual"
            node.args.append(ast.Name(id="prior", ctx=ast.Load()))
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if self.name == "func_traction_dual" and target == "value":
                node.value = expression("prior.force[i_s, i_b] * (traction_state.weight_sum[i_c, i_b] / prior.weight_sum[i_s, i_b]) / traction_state.scale[i_c, i_b] - dual[0:3] - dual[3:6].cross(relative)")
            if self.name == "func_source_traction" and target == "force":
                node.value = expression("prior.force[i_s, i_b] * (traction_state.weight_sum[i_c, i_b] / prior.weight_sum[i_s, i_b]) / traction_state.scale[i_c, i_b]")
            if self.name == "kernel_fit_tractions_impl" and target == "scale":
                node.value = expression("qd.max(prior.scale[i_b], force.norm(), moment.norm() / radius)")
            if self.name == "kernel_fit_tractions_impl" and target == "budget_scale" and ast.unparse(node.value) == "force.norm()":
                node.value = expression("prior.scale[i_b]")
        return node


header = '''"""Unselected explicit aggregate model retaining every original source-force prior.

Same positive native profiles, radius, friction and topology; aggregate wrench
conservation replaces individual point-component conservation in this experiment.
No production module is patched or changed.
"""

from dataclasses import dataclass
from typing import ClassVar
import quadrants as qd
import genesis as gs
from genesis.utils.array_class import DataKind, V, V_VEC
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, StressSourceStatus
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionState, StressTractionInfo, StressTractionStatus,
    func_traction_geometry, func_traction_source_friction,
    func_traction_support_range, func_coulomb_projection, func_traction_direction,
)

@dataclass(frozen=True)
class StressSourcePriors:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    force: qd.Tensor
    weight_sum: qd.Tensor
    scale: qd.Tensor

def create_source_priors(n_sources, n_envs):
    return StressSourcePriors(
        force=V_VEC(3, dtype=gs.qd_float, shape=(n_sources,n_envs)),
        weight_sum=V(dtype=gs.qd_float, shape=(n_sources,n_envs)),
        scale=V(dtype=gs.qd_float, shape=(n_envs,)),
    )

@qd.kernel
def kernel_snapshot_source_priors(contacts: StressContactState, sources: StressSourceState, tractions: StressTractionState, prior: StressSourcePriors):
    for i_s, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        prior.force[i_s,i_b] = contacts.force[i_s,i_b]
        total = gs.qd_float(0.0)
        for i_q in range(tractions.sample_count[i_s,i_b]):
            total += tractions.profile[i_q,i_s,i_b]
        prior.weight_sum[i_s,i_b] = total
    for i_b in range(contacts.valid.shape[1]):
        scale = gs.qd_float(0.0)
        for i_s in range(contacts.valid.shape[0]):
            scale += qd.max(contacts.force[i_s,i_b].norm(),sources.anchor_moment[i_s,i_b].norm()/contacts.radius[i_s,i_b])
        prior.scale[i_b] = scale

@qd.kernel
def kernel_aggregate_pair(contacts: StressContactState, sources: StressSourceState):
    for i_b in range(contacts.valid.shape[1]):
        anchor = contacts.center[0,i_b]
        point = contacts.position[0,i_b]
        force = contacts.force[0,i_b] + contacts.force[1,i_b]
        moment = sources.anchor_moment[0,i_b] + sources.anchor_moment[1,i_b]
        moment += (contacts.center[1,i_b]-anchor).cross(contacts.force[1,i_b])
        contacts.force[0,i_b] = force
        sources.anchor_moment[0,i_b] = moment
        sources.free_moment[0,i_b] = moment - (point-anchor).cross(force)
        contacts.valid[1,i_b] = False
        contacts.force[1,i_b] = qd.Vector.zero(gs.qd_float,3)
        sources.anchor_moment[1,i_b] = qd.Vector.zero(gs.qd_float,3)
        sources.free_moment[1,i_b] = qd.Vector.zero(gs.qd_float,3)

'''
parts = [header]
for name in ("func_traction_dual", "kernel_fit_tractions_impl", "func_source_traction"):
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    parts.append(ast.unparse(ast.fix_missing_locations(Transform(name).visit(original))) + "\n\n")
parts.append('''
@qd.kernel
def kernel_query_prior_forces(result: qd.Tensor, contacts: StressContactState, tractions: StressTractionState, info: StressTractionInfo, prior: StressSourcePriors):
    for i_q,i_s,i_b in qd.ndrange(result.shape[0], result.shape[1], result.shape[2]):
        result[i_q,i_s,i_b] = func_prior_traction(i_q,i_s,0,i_b,contacts,tractions,info,prior)

@qd.kernel
def kernel_scatter_prior_forces(forces: qd.Tensor, state: qd.template(), tractions: StressTractionState, info: StressTractionInfo, model: qd.template(), surface: qd.template()):
    for i_n,i_b in qd.ndrange(state.force.shape[0],state.force.shape[1]):
        state.force[i_n,i_b] = qd.Vector.zero(gs.qd_float,3)
    for i_q,i_s,i_b in qd.ndrange(forces.shape[0], forces.shape[1], forces.shape[2]):
        if i_q < tractions.sample_count[i_s,i_b] and tractions.profile[i_q,i_s,i_b] > 0.0:
            from_basis = qd.Vector.zero(gs.qd_float,6)
            face = 0
            face, from_basis = func_prior_sample_basis(i_q,i_s,i_b,tractions,surface,info)
            for local in qd.static(range(6)):
                node = model.surface_nodes[face,local]
                for dimension in qd.static(range(3)):
                    qd.atomic_add(state.force[node,i_b][dimension],from_basis[local]*forces[i_q,i_s,i_b][dimension])

from genesis.engine.solvers.rigid.stress.traction import func_traction_shape as func_prior_sample_basis
''')
target = root / "source_prior_candidate_v1.py"
assert not target.exists()
target.write_text("".join(parts))
print(str(target), flush=True)
