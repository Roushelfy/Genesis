"""Strictly compare original, shared-serial and cooperative force-line anchors.

Scientific execution requires Slurm rtx-mid/image 1_26. Only physical source
and surface inputs enter native preparation. The original serial functions
come from fixed Git bytes; no CPU-computed anchor or elastic field is uploaded.
This is an accuracy and lifecycle audit, without a timing claim.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ORIGINAL = "0ec2e4a7aa767608c4c22a7ece073ec3fe142332"
CONTACT = "genesis/engine/solvers/rigid/stress/contact.py"
QUADRILATERAL = "genesis/engine/solvers/rigid/stress/quadrilateral.py"
INPUT_FIELDS = ("position", "force", "normal", "friction", "radius", "valid", "radius_status")
OUTPUT_FIELDS = (
    "anchor_face",
    "center",
    "status",
    "is_refined",
    "force_error",
    "moment_error",
    "candidate_count",
    "evaluations",
)


def _api():
    from .check_planar_loading_gpu import (
        _api as loading_api,
        build_native_mother,
        prepare_physical_contacts,
        unbatched_array,
    )

    api = loading_api()
    from genesis.engine.solvers.rigid.stress.data import allocate_stress_info

    return SimpleNamespace(
        **vars(api),
        build_native_mother=build_native_mother,
        prepare_physical_contacts=prepare_physical_contacts,
        unbatched_array=unbatched_array,
        allocate_stress_info=allocate_stress_info,
    )


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _original(args, writer):
    """Compile original expressions with the current contact/surface tensor types."""
    revision = subprocess.check_output(
        ["git", "-C", str(args.git_source), "rev-parse", args.serial_source_revision + "^{commit}"], text=True
    ).strip()
    blobs = {
        name: subprocess.check_output(["git", "-C", str(args.git_source), "show", f"{revision}:{name}"])
        for name in (CONTACT, QUADRILATERAL)
    }
    tree = ast.parse(blobs[CONTACT])
    wanted = {"kernel_anchor", "func_force_frame", "func_contact_force_admissible", "func_anchor"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise ValueError("The frozen control must provide the original serial anchor functions.")
    anchor = next(node for node in functions if node.name == "kernel_anchor")
    if len(anchor.args.args) != 3 or ast.unparse(anchor.decorator_list[0]) != "qd.kernel":
        raise ValueError("The serial control must precede the cooperative wrapper.")
    quad_path = writer.path("frozen-original/quadrilateral.py")
    quad_path.write_bytes(blobs[QUADRILATERAL])
    writer.register(quad_path)
    quad_name = "_planar_anchor_original_quadrilateral"
    _import_file(quad_name, quad_path)
    control = writer.path("frozen-original/anchor.py")
    header = (
        "import quadrants as qd\n"
        "import genesis as gs\n"
        "from genesis.engine.solvers.rigid.stress.contact import StressContactState\n"
        "from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo\n"
        f"from {quad_name} import func_quad_inverse\n\n"
    )
    control.write_text(header + "\n\n".join(ast.unparse(node) for node in functions) + "\n")
    writer.register(control)
    module = _import_file("_planar_anchor_original_contact", control)
    writer.json(
        "original-source.json",
        {
            "revision": revision,
            "git_source": str(args.git_source),
            "git_blobs_sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
            "extracted_functions": sorted(wanted),
            "extracted_anchor_sha256": _hash(control),
            "scope": "Original serial expressions and original Q4 inverse from fixed Git bytes. Only dataclass tensor annotations use the current engine types. No source expression, predicate or arithmetic is rewritten.",
        },
    )
    return module


def _snapshot(api, contacts, names):
    return {name: api.unbatched_array(api, vars(contacts)[name]).copy() for name in names}


def _same_bits(actual, expected):
    return actual.shape == expected.shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def _difference(np, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"shape_or_dtype_difference": True}
    changed = (
        actual.copy().view(np.uint8).reshape((*actual.shape, actual.dtype.itemsize))
        != expected.copy().view(np.uint8).reshape((*expected.shape, expected.dtype.itemsize))
    ).any(axis=-1)
    return {"changed_elements": int(changed.sum()), "first_indices": np.argwhere(changed)[:32].tolist()}


def _upload(contacts, values):
    for name, value in values.items():
        vars(contacts)[name].from_numpy(value)


def _audit(api, writer, checks, original, label, surface, source, names, expected=None, pool=True):
    """Keep three independent states through slot/release/reset transitions."""
    np = api.np
    values = _snapshot(api, source, INPUT_FIELDS)
    n_contacts, n_envs = values["valid"].shape
    states = {
        mode: api.create_contacts(n_contacts, n_envs, 1e-4, pool, pool, True)
        for mode in ("original_serial", "shared_serial", "cooperative")
    }
    for contacts in states.values():
        contacts.center.fill(7.25)
        contacts.anchor_face.fill(surface.info.face_origin.shape[0] - 1)
    writer.arrays(f"{label}/physical-inputs.npz", **values)
    writer.json(f"{label}/cases.json", names)
    epsilon = float(np.finfo(np.float64).eps)
    phases = ("initial", "old_failure", "slot_reverse", "release_all", "reload", "release_env0", "reload_env0")
    rows = []
    for phase in phases:
        current = {name: value.copy() for name, value in values.items()}
        if phase == "slot_reverse":
            current = {name: value[::-1].copy() for name, value in current.items()}
        elif phase == "release_all":
            current["valid"].fill(False)
        elif phase == "release_env0":
            current["valid"][:, 0] = False
        outputs = {}
        for mode, contacts in states.items():
            _upload(contacts, current)
            if phase in ("initial", "old_failure", "reload", "reload_env0"):
                contacts.status.fill(5)
                contacts.force_error.fill(3.0)
                contacts.moment_error.fill(4.0)
                contacts.candidate_count.fill(17)
                contacts.evaluations.fill(19)
                contacts.is_refined.fill(True)
                contacts.active_count.fill(-7)
            before = _snapshot(api, contacts, INPUT_FIELDS)
            if mode == "original_serial":
                original.kernel_anchor(epsilon, contacts, surface.info)
            else:
                api.kernel_anchor(epsilon, contacts, surface.info, mode == "cooperative")
            api.qd.sync()
            after = _snapshot(api, contacts, INPUT_FIELDS)
            checks.condition(
                f"{label}/{phase}/{mode}/physical_input_bits",
                all(_same_bits(after[name], value) for name, value in before.items()),
            )
            outputs[mode] = _snapshot(api, contacts, OUTPUT_FIELDS)
            writer.arrays(f"{label}/{phase}/{mode}.npz", **outputs[mode])
        differences = {}
        for mode in ("shared_serial", "cooperative"):
            differences[mode] = {}
            for name in OUTPUT_FIELDS:
                actual, old = outputs[mode][name], outputs["original_serial"][name]
                passed = _same_bits(actual, old)
                differences[mode][name] = _difference(np, actual, old)
                checks.condition(f"{label}/{phase}/{mode}/{name}_bits", passed, **differences[mode][name])
        if expected is not None and phase in ("initial", "old_failure", "reload", "reload_env0"):
            for i_env, record in enumerate(expected):
                for field, value in record.items():
                    checks.exact(
                        f"{label}/{phase}/authored/{i_env}/{field}",
                        outputs["original_serial"][field][0, i_env],
                        value,
                    )
        if pool and api.gs.backend == api.gs.cuda:
            # Invalid sources and zero force must not inherit or consume stale packed slots.
            statuses = outputs["cooperative"]["status"]
            admitted = current["valid"] & (statuses != 1) & (np.linalg.norm(current["force"], axis=-1) > 1e-30)
            count = api.qd_to_numpy(states["cooperative"].active_count).item()
            pairs = api.qd_to_numpy(states["cooperative"].active_pairs, transpose=True)[:count]
            expected_pairs = sorted(map(tuple, np.argwhere(admitted).tolist()))
            actual_pairs = sorted(map(tuple, pairs.tolist()))
            checks.condition(f"{label}/{phase}/fresh_source_pool", actual_pairs == expected_pairs, count=count)
        rows.append({"phase": phase, "comparisons": differences})
    return {
        "fixture": label,
        "sources": int(values["valid"].sum()),
        "environments": n_envs,
        "pool": pool,
        "phases": rows,
    }


def _surface(api, writer, label, patches, *, triangles=False):
    """Upload authored physical corners; prepare every floating surface coefficient in Quadrants."""
    np = api.np
    n_faces, corners = len(patches), 3 if triangles else 4
    vertices = np.array(patches, dtype=api.gs.np_float).reshape((-1, 3))
    faces = np.zeros((n_faces, 6), dtype=api.gs.np_int)
    faces[:, :corners] = np.arange(n_faces * corners).reshape((n_faces, corners))
    opposite = None
    if triangles:
        opposite = np.arange(len(vertices), len(vertices) + n_faces, dtype=api.gs.np_int)
        vertices = np.concatenate((vertices, vertices[faces[:, 0]] + [0.0, 0.0, -0.01]))
    load_info = api.allocate_stress_info(len(vertices), 0, n_faces, 0, 0, 0 if triangles else len(vertices))
    load_info.vertices.from_numpy(vertices)
    load_info.surface_nodes.from_numpy(faces)
    if triangles:
        load_info.surface_opposite_vertices.from_numpy(opposite)
    else:
        load_info.load_nodes.fill(0)
        load_info.load_offsets.fill(0.0)
    writer.arrays(f"{label}/physical-surface.npz", vertices=vertices, faces=faces)
    return api.StressSurface(16, load_info)


def _source(api, rows):
    np = api.np
    n_envs, n_contacts = len(rows), 2
    contacts = api.create_contacts(n_contacts, n_envs, 1e-4, True, True, True)
    values = {
        "position": np.zeros((n_contacts, n_envs, 3), dtype=api.gs.np_float),
        "force": np.zeros((n_contacts, n_envs, 3), dtype=api.gs.np_float),
        "normal": np.zeros((n_contacts, n_envs, 3), dtype=api.gs.np_float),
        "friction": np.ones((n_contacts, n_envs), dtype=api.gs.np_float),
        "radius": np.full((n_contacts, n_envs), 1e-4, dtype=api.gs.np_float),
        "valid": np.ones((n_contacts, n_envs), dtype=bool),
        "radius_status": np.zeros((n_contacts, n_envs), dtype=api.gs.np_int),
    }
    for i_env, row in enumerate(rows):
        values["position"][0, i_env] = row.get("point", [0.0, 0.0, 0.0001])
        values["force"][0, i_env] = row.get("force", [0.0, 0.0, -0.05])
        values["normal"][0, i_env] = row.get("normal", [0.0, 0.0, -1.0])
        values["radius"][0, i_env] = row.get("radius", 1e-4)
        values["friction"][0, i_env] = row.get("friction", 1.0)
        values["valid"][0, i_env] = row.get("valid", True)
        # A distinct second source makes slot reversal nontrivial, including a zero-force valid slot.
        values["position"][1, i_env] = [0.0, 0.0, 0.0]
        values["normal"][1, i_env] = [0.0, 0.0, -1.0]
        values["force"][1, i_env] = [0.0, 0.0, -0.025 if i_env % 2 else 0.0]
    _upload(contacts, values)
    return contacts


def _patch(x=0.0, z=0.0):
    return [[x - 0.002, -0.002, z], [x + 0.002, -0.002, z], [x + 0.002, 0.002, z], [x - 0.002, 0.002, z]]


def _primitives(api, writer, checks, original):
    fixtures = []
    patches = [_patch(x=1.0 + i) for i in range(70)]
    for index, x in ((0, 0.0), (63, 0.0), (31, 0.01), (32, 0.01), (1, 0.02), (64, 0.02)):
        patches[index] = _patch(x=x)
    rows = [
        {"name": "tie-0-63", "point": [0.0, 0.0, 0.0]},
        {"name": "tie-31-32", "point": [0.01, 0.0, 0.0001]},
        {"name": "tie-1-64", "point": [0.02, 0.0, -0.0001]},
        {"name": "slab-axis-parallel", "point": [0.002, 0.0, 0.0001]},
        {"name": "edge-near-outside", "point": [math.nextafter(0.002, math.inf), 0.0, 0.0001]},
        {"name": "miss", "point": [0.0, 1.0, 0.0001]},
        {"name": "parallel", "force": [-0.05, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0]},
        {
            "name": "grazing-above",
            "point": [0.0, 0.0, 0.0],
            "force": [-0.05, 0.0, -1e-13],
            "normal": [-1.0, 0.0, -2e-12],
        },
        {
            "name": "grazing-below",
            "point": [0.0, 0.0, 0.0],
            "force": [-0.05, 0.0, -2.5e-14],
            "normal": [-1.0, 0.0, -5e-13],
        },
        {"name": "NaN-force", "force": [math.nan, 0.0, -0.05]},
        {"name": "infinite-force", "force": [0.0, 0.0, -math.inf]},
        {"name": "NaN-point", "point": [math.nan, 0.0, 0.0]},
        {"name": "NaN-normal", "normal": [0.0, math.nan, -1.0]},
        {"name": "NaN-radius", "radius": math.nan},
        {"name": "zero-radius", "radius": 0.0},
        {"name": "NaN-friction", "friction": math.nan},
        {"name": "outside-cone", "force": [0.1, 0.0, -0.05], "friction": 0.1},
        {"name": "zero-force", "force": [0.0, 0.0, 0.0]},
        {"name": "released", "valid": False},
    ]
    expected = [{"anchor_face": index, "status": 0} for index in (0, 31, 1)]
    expected.extend({} for _ in rows[len(expected) :])
    surface = _surface(api, writer, "q4-boundaries", patches)
    contacts = _source(api, rows)
    names = [row["name"] for row in rows]
    fixtures.append(_audit(api, writer, checks, original, "q4-boundaries", surface, contacts, names, expected))
    fixtures.append(_audit(api, writer, checks, original, "q4-no-pool", surface, contacts, names, expected, False))
    for triangles in (False, True):
        label = "triangle-near-tie" if triangles else "q4-near-tie"
        near = math.nextafter(2.0**-10, 0.0)
        patches = [_patch(x=1.0 + i) for i in range(70)]
        patches[31], patches[32] = _patch(z=2.0**-10), _patch(z=near)
        if triangles:
            patches = [[patch[0], patch[1], patch[3]] for patch in patches]
        surface = _surface(api, writer, label, patches, triangles=triangles)
        rows = [{"name": "one-ULP-nearer-later-face", "point": [-0.0005, -0.0005, 0.0]}]
        fixtures.append(
            _audit(
                api,
                writer,
                checks,
                original,
                label,
                surface,
                _source(api, rows),
                [rows[0]["name"]],
                [{"anchor_face": 32, "status": 0}],
            )
        )
    book = [_patch(), [[0.0, -0.002, 0.0], [0.0, 0.002, 0.0], [0.0, 0.002, 0.004], [0.0, -0.002, 0.004]]]
    rows = [
        {"name": "book-edge-tie", "point": [0.0, 0.0, 0.0], "force": [-0.02, 0.0, -0.02], "normal": [-1.0, 0.0, -1.0]},
        {
            "name": "book-nearest-sheet",
            "point": [0.0005, 0.0, 0.0005],
            "force": [-0.02, 0.0, -0.02],
            "normal": [-1.0, 0.0, -1.0],
        },
        {
            "name": "book-parallel-first",
            "point": [0.0001, 0.0, 0.001],
            "force": [-0.05, 0.0, 0.0],
            "normal": [-1.0, 0.0, 0.0],
        },
    ]
    surface = _surface(api, writer, "q4-book", book)
    fixtures.append(
        _audit(api, writer, checks, original, "q4-book", surface, _source(api, rows), [row["name"] for row in rows])
    )
    return fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--serial-source-revision", default=ORIGINAL)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stage", choices=("all", "banks", "primitives"), default="all")
    parser.add_argument("--banks", choices=("egg75", "torus8"), nargs="+", default=["egg75", "torus8"])
    parser.add_argument("--quadrature", type=int, choices=(10, 16, 24, 48), default=16)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    if os.environ.get("GENESIS_IMAGE_VER") != "1_26":
        parser.error("Use image 1_26 for the frozen native comparison.")
    args.output, args.reference, args.git_source = (
        path.expanduser().resolve() for path in (args.output, args.reference, args.git_source)
    )
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh project data output under /mnt/data/zhaofeng.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "genesis.planar-loading-reference.v1" or not manifest["passed"]:
        raise ValueError("Use a successful pinned planar-loading physical input export.")
    for relative, record in manifest["files"].items():
        if relative.startswith("inputs/") and _hash(args.reference / relative) != record["sha256"]:
            raise ValueError(f"Physical input changed: {relative}.")
    started = time.perf_counter()
    api = _api()
    api.gs.init(backend=api.gs.cuda if args.backend == "cuda" else api.gs.cpu, precision="64", logging_level="warning")
    writer, checks = api.Writer(args.output, api), api.Checks(api)
    writer.json(
        "start.json",
        {
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "identity": api.runtime_identity(),
            "reference_manifest_sha256": _hash(manifest_path),
            "native_package": api.gs.__file__,
            "native_contact_sha256": _hash(Path(api.gs.__file__).parent / "engine/solvers/rigid/stress/contact.py"),
        },
    )
    original = _original(args, writer)
    fixtures = []
    if args.stage in ("all", "banks"):
        for bank in args.banks:
            native = api.build_native_mother(api, args.reference, args.output, bank, quadrature=args.quadrature)
            surface = api.StressSurface(args.quadrature, native.model.load_info)
            contacts, cases, _valid, _sides = api.prepare_physical_contacts(
                api, args.reference, bank, native.model.options.young, native.model.options.poisson
            )
            fixtures.append(
                _audit(api, writer, checks, original, bank, surface, contacts, [row["name"] for row in cases])
            )
    if args.stage in ("all", "primitives"):
        fixtures.extend(_primitives(api, writer, checks, original))
    checks.flush(writer)
    passed = all(row["passed"] for row in checks.rows)
    writer.json("manifest.json", {"schema": "genesis.planar-anchor-parity.v1", "files": dict(writer.inventory)})
    writer.json(
        "result.json",
        {
            "complete": True,
            "passed": passed,
            "primary_timing": False,
            "bitwise_contract": True,
            "cooperative_anchor_requested": True,
            "anchor_mode": "separate_warp" if args.backend == "cuda" else "separate_serial",
            "original_revision": args.serial_source_revision,
            "checks": len(checks.rows),
            "failed_checks": [row["name"] for row in checks.rows if not row["passed"]],
            "fixtures": fixtures,
            "seconds_including_native_preparation": time.perf_counter() - started,
            "scope": "Fixed original serial, shared serial and cooperative anchor fields must match bitwise, including invalid/zero and unchanged stale outputs. Authored anchors add independent exact tie/nearest-sheet checks. No stress tolerance or numerical gate is relaxed; all negative readbacks are retained.",
        },
    )
    if not passed:
        raise ArithmeticError("The cooperative anchor changed original anchor fields or source/lifecycle behavior.")


if __name__ == "__main__":
    main()
