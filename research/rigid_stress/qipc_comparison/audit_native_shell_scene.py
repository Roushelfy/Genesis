"""Independent CPU full-field recovery checks for captured native shell scene frames."""

import argparse
from pathlib import Path

import numpy as np

from research.rigid_stress.shell_cpu.core import ShellModel

from .artifacts import file_id, runtime_identity, write_json


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scene", type=Path, required=True)
    cli.add_argument("--asset", type=Path, required=True)
    cli.add_argument("--young", type=float, default=1e10)
    cli.add_argument("--poisson", type=float, default=0.3)
    cli.add_argument("--thickness", type=float, default=0.0005)
    cli.add_argument("--density", type=float, default=2000.0)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        dict(
            identity=runtime_identity(require_gpu_inventory=False),
            asset=file_id(args.asset),
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        ),
    )
    with np.load(args.asset) as asset:
        model = ShellModel(
            asset["vertices"],
            asset["quads"],
            E=args.young,
            nu=args.poisson,
            h=args.thickness,
            density=args.density,
            directors=asset.get("directors"),
            gauge="translation",
        )
    inertia = (
        model.nodal_mass[:, None, None]
        * args.thickness**2
        / 12
        * (np.eye(3)[None] - model.directors[:, :, None] * model.directors[:, None, :])
    )
    lever = model.vertices - model.center_of_mass
    paths = sorted(args.scene.glob("field-*.npz"))
    assert paths, args.scene
    rows = []
    for path in paths:
        with np.load(path) as data:
            omega = data["angular_velocity"]
            force = data["nodal_force"].reshape((len(omega), -1)).T
            acceleration = np.cross(omega[None], np.cross(omega[None], lever[:, None]))
            momentum = np.einsum("nij,bj->nbi", inertia, omega)
            centrifugal = np.concatenate(
                (-model.nodal_mass[:, None, None] * acceleration, -np.cross(omega[None], momentum)), axis=2
            )
            raw = force + centrifugal.transpose(0, 2, 1).reshape((-1, len(omega)))
            expected_u = model.solve(raw)
            native_u = data["displacement"].reshape((len(omega), -1)).T
            response = model.stress(expected_u)
            tensor = np.moveaxis(np.stack((response["sigma_top"], response["sigma_bottom"]), axis=1), -1, 0)
            expected = np.stack(
                [tensor[..., a, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1
            )
            errors = dict(
                complete_field=relative(data["stress_tensor"], expected),
                complete_residual=float(model.residual(native_u, raw)),
            )
            for label in ("membrane", "bending"):
                local = response["N" if label == "membrane" else "M"]
                # These are independent shell resultants, already scaled by h or h^3/12 in the CPU model.
                basis = model.axes[:, :, :, :2]
                local_tensor = np.zeros((*local.shape[:2], 2, 2, local.shape[-1]))
                local_tensor[:, :, 0, 0], local_tensor[:, :, 1, 1] = local[:, :, 0], local[:, :, 1]
                local_tensor[:, :, 0, 1] = local_tensor[:, :, 1, 0] = local[:, :, 2]
                world = np.einsum("egik,egklb,egjl->begij", basis, local_tensor, basis)
                values = np.stack(
                    [world[..., a, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1
                )
                errors[label] = relative(data["shell_" + label], values)
            transverse = np.einsum("egij,egjb->begi", model.axes[:, :, :, :2], response["Q"])
            errors["transverse"] = relative(data["shell_transverse"], transverse)
            rows.append(dict(frame=path.name, input=file_id(path), errors=errors))
            np.savez_compressed(
                args.output / (path.stem + "-reference.npz"), expected_field=expected, expected_displacement=expected_u
            )
        write_json(args.output / "result.json", dict(passed=False, complete=False, rows=rows))
        assert max(errors.values()) < 1e-8, rows[-1]
    write_json(
        args.output / "result.json",
        dict(
            passed=True,
            complete=True,
            rows=rows,
            scope="Independent implementation error for the same shell model and captured generalized loads; no 3D physical peak certification.",
        ),
    )


if __name__ == "__main__":
    main()
