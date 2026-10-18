"""Measure actual mass COM, deformation and gripper-relative drift for complete common grasps."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation
from .reference_uncertainty import frame_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dpr", type=Path, required=True)
    parser.add_argument("--scr", type=Path, required=True)
    parser.add_argument("--shell", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--protocol", choices=("prescribed", "capped"), required=True)
    parser.add_argument("--legacy-nominal-shell", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.conditions) as saved:
        assert len(saved["position"]) == 1
        delay = saved["delays"][0].item() * 0.01
    model_path = args.reference / "panda-bank-initial.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq == 9
    data = mujoco.MjData(model)
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    assert hand >= 0
    outcomes, payload = [], {}
    for method, folder in (("reference", args.reference), ("dpr", args.dpr), ("scr", args.scr), ("shell", args.shell)):
        receipt = json.loads((folder / "result.json").read_text())
        assert receipt["completed_requested_history"], "Outcome table requires an actually complete history"
        with np.load(folder / "rest.npz") as saved:
            rest = {name: saved[name] for name in saved.files}
        native = method in ("dpr", "scr")
        if native:
            start = json.loads((folder / "start.json").read_text())
            urdf = Path(start["elastic_mesh"]["path"]).parent / "egg_shell.urdf"
            offset = np.fromstring(ElementTree.parse(urdf).find("./link/inertial/origin").attrib["xyz"], sep=" ")
            mass_note = "Actual authored URDF COM; geometry is rigid by the model assumption"
        elif method == "reference":
            authored, masses = rest["authored_vertices_m"], rest["masses_kg"]
            mass_note = "Actual P1 nodal mass weights"
        else:
            authored = rest["vertices_m"]
            if args.legacy_nominal_shell:
                assert "vertex_masses_kg" not in rest, "Use the recorded mass codec when present"
                triangles = authored[rest["faces"]]
                area = (
                    np.linalg.norm(
                        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
                    )
                    / 2
                )
                masses = np.bincount(
                    rest["faces"].reshape(-1), weights=np.repeat(area * 2000 * 0.0005 / 3, 3), minlength=len(authored)
                )
                mass_note = "Explicit legacy nominal500um rho2000 face-area masses; not an unweighted vertex mean"
            else:
                masses = rest["vertex_masses_kg"]
                mass_note = "Recorded actual face/vertex mass weights"
        rows = []
        for tick, path in frame_index(folder).items():
            with np.load(path) as saved:
                q = saved["target_q"] if method == "reference" and args.protocol == "prescribed" else saved["q"]
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                hand_origin, hand_rotation = data.xpos[hand], data.xmat[hand].reshape(3, 3)
                if native:
                    rotation = Rotation.from_quat(saved["link_quaternion_wxyz"][[1, 2, 3, 0]]).as_matrix()
                    center = saved["link_position_m"] + rotation @ offset
                    rms_deformation = 0.0
                else:
                    current = saved["positions_m"] if method == "reference" else saved["vertices_world_m"]
                    center = masses @ current / masses.sum()
                    rotation = corotation(authored, current, masses)
                    predicted = (authored - masses @ authored / masses.sum()) @ rotation.T + center
                    rms_deformation = np.sqrt(
                        np.einsum("v,vi,vi->", masses, current - predicted, current - predicted) / masses.sum()
                    ).item()
                relative = hand_rotation.T @ (center - hand_origin)
                rows.append(
                    {
                        "time_s": tick * 0.01,
                        "mass_com_world_m": center.tolist(),
                        "authored_hand_relative_com_m": relative.tolist(),
                        "mass_weighted_nonrigid_rms_m": rms_deformation,
                    }
                )
        times = np.array([row["time_s"] for row in rows])
        centers = np.array([row["mass_com_world_m"] for row in rows])
        relative = np.array([row["authored_hand_relative_com_m"] for row in rows])
        deformation = np.array([row["mass_weighted_nonrigid_rms_m"] for row in rows])
        baseline = np.abs(times - (delay + 1.8)).argmin()
        selected = (times >= delay + 1.8) & (times <= delay + 5.4)
        weak = (times >= delay + 4.2) & (times <= delay + 4.56)
        drift = np.linalg.norm(relative - relative[baseline], axis=1)
        outcomes.append(
            {
                "method": method,
                "source": file_id(folder / "result.json"),
                "mass_contract": mass_note,
                "maximum_com_height_m": centers[:, 2].max().item(),
                "final_com_world_m": centers[-1].tolist(),
                "maximum_nonrigid_rms_m": deformation.max().item(),
                "maximum_hand_relative_com_drift_during_first_lift_hold_m": drift[selected].max().item()
                if selected.any()
                else None,
                "minimum_com_height_during_weak_first_grip_m": centers[weak, 2].min().item() if weak.any() else None,
                "history": rows,
            }
        )
        payload[f"{method}_time_s"] = times
        payload[f"{method}_com_world_m"] = centers
        payload[f"{method}_hand_relative_com_m"] = relative
        payload[f"{method}_nonrigid_rms_m"] = deformation
    np.savez_compressed(args.output / "trajectories.npz", **payload)
    write_json(
        args.output / "result.json",
        {
            "kind": "actual-common-grasp-outcomes-and-relative-drift",
            "identity": runtime_identity(),
            "mujoco_version": mujoco.__version__,
            "kinematic_model": file_id(model_path),
            "protocol": args.protocol,
            "histories": outcomes,
            "reference_qualified": False,
            "scope": "Actual mass COMs and fitted nonrigid RMS, without unweighted-vertex substitutions. Hand frames use the authored MuJoCo kinematics of each actual saved q; prescribed QIPC uses its shared target q, so its separately measured finite DBC tracking error remains. Hand-relative COM drift is a macroscopic grasp/slip observable, not a local contact stick/slip classification or friction-cone certificate. A zero native geometric deformation is its rigid-model assumption. No physical accuracy ranking or damage threshold follows from these descriptive outcomes.",
        },
    )


if __name__ == "__main__":
    main()
