"""Restrict the identical native fan rule's launch grid to actual candidate faces; do not change integration."""

import ast
import copy
from pathlib import Path

root = Path(__file__).parent
tree = ast.parse(Path("genesis/engine/solvers/rigid/stress/footprint.py").read_text())
original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kernel_surface_fan_profiles")
assert len(original.body) == 4 and all(isinstance(node, ast.For) for node in original.body)


def statements(text):
    return ast.parse(text).body


def phase(name, body, extra=()):
    result = copy.deepcopy(original)
    result.name, result.body = name, body
    result.args.args.extend(ast.arg(arg=arg, annotation=ast.parse(annotation, mode="eval").body) for arg, annotation in extra)
    return ast.unparse(ast.fix_missing_locations(result)) + "\n\n"


header = '''"""Unselected native fan-frontier launch experiment. Same geometry,samples,profiles and acceptance."""

from dataclasses import dataclass
from typing import ClassVar
import weakref
import quadrants as qd
import genesis as gs
from genesis.utils.array_class import V, DataKind
from genesis.utils.misc import qd_to_numpy
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState, StressTractionInfo, func_surface_fan_bary
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState, StressFootprintStatus, func_footprint_point_distance
from genesis.engine.solvers.rigid.stress.provenance import func_closest_triangle

@dataclass(frozen=True)
class FanFrontier:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    indices: qd.Tensor
    count: qd.Tensor
    maximum: qd.Tensor

workspaces = {}

def workspace(tractions):
    key = id(tractions)
    if key not in workspaces:
        shape = tractions.fan_barycentric.shape
        value = FanFrontier(indices=V(dtype=gs.qd_int,shape=shape),count=V(dtype=gs.qd_int,shape=shape[1:]),maximum=V(dtype=gs.qd_int,shape=()))
        workspaces[key] = (weakref.ref(tractions, lambda _ref: workspaces.pop(key,None)),value)
    return workspaces[key][1]

'''
first = copy.deepcopy(original.body[:2])
first[1].body.extend(statements('''
candidate = False
if contact_state.valid[i_c,i_b] and footprint_state.status[i_c,i_b] == StressFootprintStatus.SUCCESS:
    circle = traction_state.circle.face[i_c,i_b]
    if circle >= 0:
        candidate = i_f == circle
    else:
        origin,edges = surface_info.face_origin[i_f],surface_info.face_edges[i_f]
        center = origin + bary[1]*edges[0,:] + bary[2]*edges[1,:]
        candidate = (center-contact_state.center[i_c,i_b]).norm() < contact_state.radius[i_c,i_b]
frontier.indices[i_f,i_c,i_b] = i_f if candidate else -1
'''))
first = statements('''
for _ in range(1):
    frontier.maximum[None] = 0
''') + first + statements('''
for i_c,i_b in qd.ndrange(contact_state.valid.shape[0],contact_state.valid.shape[1]):
    count = 0
    for i_f in range(surface_info.face_origin.shape[0]):
        face = frontier.indices[i_f,i_c,i_b]
        if face >= 0:
            frontier.indices[count,i_c,i_b] = face
            count += 1
    frontier.count[i_c,i_b] = count
    qd.atomic_max(frontier.maximum[None],count)
''')
sample = copy.deepcopy(original.body[2])
sample.target = ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Store()) for name in ("i_task", "i_c", "i_b")], ctx=ast.Store())
sample.iter.args[0] = ast.parse("maximum * 3 * surface_info.gauss.shape[0] ** 2", mode="eval").body
sample.body = statements('''
n_local = 3 * surface_info.gauss.shape[0] ** 2
i_work = i_task // n_local
''') + [ast.If(test=ast.parse("i_work < frontier.count[i_c,i_b]", mode="eval").body,
                   body=statements("i_q = frontier.indices[i_work,i_c,i_b] * n_local + i_task % n_local\nif traction_state.circle.face[i_c,i_b] >= 0:\n    i_q = i_task % n_local") + sample.body,
                   orelse=[])]
parts = [header, phase("kernel_fan_frontier", first, (("frontier", "FanFrontier"),)), phase("kernel_fan_frontier_samples", [sample, copy.deepcopy(original.body[3])], (("frontier", "FanFrontier"), ("maximum", "int")))]
parts.append('''
def kernel_surface_fan_profiles(contacts,footprints,tractions,model,surface,paths,info,epsilon):
    frontier = workspace(tractions)
    kernel_fan_frontier(contacts,footprints,tractions,model,surface,paths,info,epsilon,frontier)
    maximum = int(qd_to_numpy(frontier.maximum,transpose=True))
    kernel_fan_frontier_samples(contacts,footprints,tractions,model,surface,paths,info,epsilon,frontier,maximum)
''')
target = root / "fan_frontier_candidate_v1.py"
assert not target.exists()
target.write_text("".join(parts))
print(str(target), flush=True)
