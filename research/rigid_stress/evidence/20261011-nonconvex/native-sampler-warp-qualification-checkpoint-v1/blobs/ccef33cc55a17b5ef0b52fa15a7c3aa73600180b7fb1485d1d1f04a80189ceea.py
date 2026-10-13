"""Remove prototype dispatch without changing the qualified CUDA kernel body."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).parent
primary = Path("/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery")
prototype = Path("/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-sampler-warp")
relative = "genesis/engine/solvers/rigid/stress/surface_frontier.py"
source = (prototype / relative).read_text()
original = (primary / relative).read_bytes()
assert original == subprocess.check_output(["git", "show", f"eba3aeb2:{relative}"], cwd=primary)
nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
baseline = nodes["kernel_fan_frontier_samples"]
candidate = nodes["kernel_fan_frontier_samples_aggregated"]
lines = source.splitlines(keepends=True)
prefix = "".join(lines[: min(node.lineno for node in baseline.decorator_list) - 1])
kernel = "".join(lines[min(node.lineno for node in candidate.decorator_list) - 1 : candidate.end_lineno])
kernel = kernel.replace("def kernel_fan_frontier_samples_aggregated(", "def kernel_fan_frontier_samples(", 1)
wrapper = """


def kernel_surface_fan_profiles(contacts, footprints, tractions, model, surface, paths, info, epsilon):
    frontier = tractions.frontier
    kernel_fan_frontier(contacts, footprints, tractions, model, surface, paths, info, epsilon, frontier)
    maximum = int(qd_to_numpy(frontier.maximum, transpose=True))
    kernel_fan_frontier_samples(
        contacts, footprints, tractions, model, surface, paths, info, epsilon, frontier, maximum
    )
"""
clean = prefix + kernel + wrapper
result = {node.name: node for node in ast.parse(clean).body if isinstance(node, ast.FunctionDef)}
assert len(result) == 3
assert ast.dump(result["kernel_fan_frontier_samples"].args) == ast.dump(candidate.args)
assert [ast.dump(node) for node in result["kernel_fan_frontier_samples"].body] == [
    ast.dump(node) for node in candidate.body
]
assert ast.dump(result["kernel_fan_frontier"]) == ast.dump(nodes["kernel_fan_frontier"])
# CPU uses the unchanged footprint implementation: frontier storage is CUDA-only.
for name in ("footprint.py", "traction.py", "intrinsic.py"):
    tracked = f"genesis/engine/solvers/rigid/stress/{name}"
    assert (primary / tracked).read_bytes() == (prototype / tracked).read_bytes()
(primary / relative).write_text(clean)
proof = {
    "prototype_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "baseline_source_sha256": hashlib.sha256(original).hexdigest(),
    "clean_source_sha256": hashlib.sha256(clean.encode()).hexdigest(),
    "qualified_cuda_kernel_arguments_and_body_ast_identical": True,
    "frontier_kernel_ast_identical": True,
    "cpu_dispatch_and_numerical_route_unchanged": True,
    "cleanup": "Remove unused original CUDA kernel, private experiment flag and host dispatch; rename qualified aggregated kernel to existing name.",
}
(root / "native-sampler-warp-clean-proof-v1.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof, indent=2), flush=True)
