"""Render complete, unsmoothed fields and physical pad loads after caller admission.

These functions do not qualify a reference or admit an accuracy comparison.
Callers must record their own admission and the exact inputs in the figure packet.
"""

import importlib.util
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from .artifacts import file_id
from .pad_spatial_figure import angular_polygons, tangent_mises

METHODS = ("dpr", "scr", "shell", "reference")


def load_style(path):
    spec = importlib.util.spec_from_file_location("captured_figure_style", path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_nature_style(matplotlib)
    matplotlib.rcParams["savefig.bbox"] = None
    return style


def export_figure(fig, stem, style):
    """Retain exact physical size, editable labels and independently checked exports."""
    text_issues = style.rendered_text_qa(fig)
    paths = style.save_nature_figure(fig, stem, png_dpi=600, preview_dpi=150)
    root = ET.parse(paths["svg"]).getroot()
    width, height = fig.get_size_inches() * 25.4
    expected_pt = fig.get_size_inches() * 72
    actual_pt = np.array([float(root.attrib[key].removesuffix("pt")) for key in ("width", "height")])
    np.testing.assert_allclose(actual_pt, expected_pt, rtol=0, atol=1e-5)
    assert abs(width - 183) < 1e-8 and height <= 170
    text_count = len(root.findall(".//{http://www.w3.org/2000/svg}text"))
    assert text_count > 10, "Editable SVG text is required"
    images = {}
    for name, dpi in (("png", 600), ("preview_png", 150)):
        with Image.open(paths[name]) as image:
            expected = fig.get_size_inches() * dpi
            assert np.max(np.abs(np.array(image.size) - expected)) <= 1
            np.testing.assert_allclose(image.info["dpi"], [dpi, dpi], atol=0.02, rtol=0)
            images[name] = {"pixels": list(image.size), "dpi": list(image.info["dpi"])}
    try:
        fonts = subprocess.run(["pdffonts", paths["pdf"]], capture_output=True, text=True, check=True).stdout
        lines = fonts.splitlines()[2:]
        assert lines and all(" yes " in line for line in lines), "PDF fonts must be embedded"
    except FileNotFoundError:
        fonts = None
    embedded = b"/FontFile2" in Path(paths["pdf"]).read_bytes()
    assert embedded, "An embedded TrueType font is required"
    result = {
        "files": {key: file_id(path) for key, path in paths.items()},
        "width_mm": float(width),
        "height_mm": float(height),
        "svg_text_elements": text_count,
        "embedded_truetype_font": embedded,
        "pdffonts": fonts,
        "raster_exports": images,
        "dense_artist_dpi": matplotlib.rcParams["savefig.dpi"],
        "text_issues": text_issues,
        "preview_inspected": False,
        "submission_ready": False,
    }
    plt.close(fig)
    return result


def surface_data(mesh, directories, record):
    """Use every reference triangle and verify both complete surface tensor arrays."""
    record(mesh)
    with np.load(mesh) as saved:
        unit, faces = saved["material_unit"], saved["mid_faces"]
    unit = unit / np.linalg.norm(unit, axis=1)[:, None]
    values, positions, reference = {}, {}, {}
    for method, directory in directories.items():
        path = directory / "surface-fields-0.1-0.npz"
        record(path)
        with np.load(path) as saved:
            for side in ("outer", "inner"):
                current = saved[side + "_rest_position_m"]
                tensor = saved[side + "_tangent_pa"]
                assert current.shape == (len(faces), 3)
                assert tensor.shape == (2, len(faces), 2, 2)
                assert np.isfinite(current).all() and np.isfinite(tensor).all()
                if side in positions:
                    np.testing.assert_array_equal(current, positions[side])
                    np.testing.assert_allclose(tensor[1], reference[side], rtol=1e-12, atol=1e-8)
                else:
                    positions[side], reference[side] = current, tensor[1]
                values[method + "_" + side] = tangent_mises(tensor[0])
                values["reference_" + side] = tangent_mises(tensor[1])
    assert len(values) == 8
    return {"unit": unit, "faces": faces, **positions}, values


def render_surface(geometry, values, stem, style, reference_label):
    polygons = angular_polygons(geometry["unit"], geometry["faces"])
    maximum = max(float(value.max()) for value in values.values())
    assert maximum > 0
    norm = matplotlib.colors.Normalize(0, maximum / 1000)
    fig, axes = plt.subplots(2, 4, figsize=style.mm_to_inches(183, 120))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.90, wspace=0.22, hspace=0.32)
    titles = ("DPR, 2 mm", "SCR, 2 mm", "Shell, repaired", reference_label)
    for row, side in enumerate(("outer", "inner")):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            for shift in (-360, 0, 360):
                collection = PolyCollection(
                    [polygon + [shift, 0] for polygon in polygons],
                    array=values[method + "_" + side] / 1000,
                    norm=norm,
                    cmap="viridis",
                    edgecolors="none",
                    antialiased=False,
                    rasterized=True,
                )
                ax.add_collection(collection)
            ax.set(xlim=(-180, 180), ylim=(180, 0), xticks=[-180, 0, 180], yticks=[0, 45, 90, 135, 180])
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(side.capitalize() + " polar angle (°)")
            if row:
                ax.set_xlabel("Azimuth (°)")
            else:
                ax.set_title(titles[col], pad=9)
            style.add_panel_label(ax, chr(97 + row * 4 + col), x=-0.13, y=1.10)
    fig.colorbar(collection, cax=fig.add_axes([0.91, 0.23, 0.018, 0.54]), label="Tangent Mises stress (kPa)")
    np.savez_compressed(stem.parent / (stem.name + "-data.npz"), **geometry, **values)
    return {
        "export": export_figure(fig, stem, style),
        "profile": {
            "faces_per_panel": len(geometry["faces"]),
            "panels": 8,
            "complete_polar_and_azimuth_domain": True,
            "shared_scale_pa": [0, maximum],
            "display_centroid_maxima_pa": {key: float(value.max()) for key, value in values.items()},
            "display_is_peak_definition": False,
        },
    }


def traction_data(directory, record):
    """Recompute densities without excluding any force-bearing sample."""
    arrays = {}
    for method in METHODS:
        for level in range(2):
            path = directory / f"traction-{method}-level{level}.npz"
            record(path)
            with np.load(path) as saved:
                current = {key: saved[key] for key in saved.files}
            p, f, a = (current[key] for key in ("applied_points_world_m", "force_n", "reference_projected_area_m2"))
            assert p.shape == f.shape == (len(a), 3)
            assert all(np.isfinite(value).all() for value in current.values())
            active = np.any(f != 0, axis=1)
            np.testing.assert_array_equal(current["active"], active)
            np.testing.assert_array_equal(current["upper"], p[:, 2] > 0)
            assert active.any() and np.all(a[active] > 0) and np.all(a >= 0)
            normal, tangent = np.zeros(len(a)), np.zeros((len(a), 2))
            normal[active] = np.abs(f[active, 2]) / a[active]
            tangent[active] = f[active, :2] / a[active, None]
            np.testing.assert_allclose(current["normal_density_pa"], normal, rtol=1e-12, atol=0)
            np.testing.assert_allclose(current["tangent_density_pa"], tangent, rtol=1e-12, atol=0)
            arrays[method, level] = current
    for level in range(2):
        # Compare the original dimensional primitives under the data reader's
        # declared tolerances. A force tolerance in N cannot be reused for Pa;
        # each density has already been independently re-derived above.
        for key in ("applied_points_world_m", "final_points_world_m", "force_n", "reference_projected_area_m2"):
            np.testing.assert_allclose(arrays["dpr", level][key], arrays["scr", level][key], rtol=1e-10, atol=1e-12)
        for key in ("active", "upper"):
            np.testing.assert_array_equal(arrays["dpr", level][key], arrays["scr", level][key])
    return {method: arrays[method, 1] for method in ("dpr", "shell", "reference")}


def render_traction(samples, stem, style, reference_label, *, tangent=False):
    maximum, extent, max_force = 0.0, 0.0, 0.0
    for sample in samples.values():
        active, f = sample["active"], sample["force_n"]
        density = np.linalg.norm(sample["tangent_density_pa"], axis=1) if tangent else sample["normal_density_pa"]
        maximum = max(maximum, float(density.max()))
        extent = max(extent, float(np.abs(sample["applied_points_world_m"][active, :2]).max()))
        for upper in (True, False):
            max_force = max(max_force, float(np.linalg.norm(f[active & (sample["upper"] == upper), :2].sum(axis=0))))
    assert maximum > 0 and extent > 0 and (not tangent or max_force > 0)
    limit = extent * 1.08 * 1000
    norm = matplotlib.colors.Normalize(0, maximum / 1000)
    fig, axes = plt.subplots(2, 3, figsize=style.mm_to_inches(183, 147))
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.18, top=0.91, wspace=0.20, hspace=0.46)
    profile = []
    for col, (method, sample) in enumerate(samples.items()):
        p, f, active = (sample[key] for key in ("applied_points_world_m", "force_n", "active"))
        density = np.linalg.norm(sample["tangent_density_pa"], axis=1) if tangent else sample["normal_density_pa"]
        for row, upper in enumerate((True, False)):
            ax = axes[row, col]
            selected = active & (sample["upper"] == upper)
            assert selected.any()
            scatter = ax.scatter(
                p[selected, 0] * 1000,
                p[selected, 1] * 1000,
                c=density[selected] / 1000,
                s=3 if method == "dpr" else 0.7,
                norm=norm,
                cmap="viridis",
                linewidths=0,
                rasterized=True,
            )
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            if row:
                ax.set_xlabel("x (mm)")
            else:
                ax.set_title(("DPR / SCR contact", "Shell, repaired", reference_label)[col], pad=10)
            if not col:
                ax.set_ylabel(("Upper" if upper else "Lower") + ": y (mm)")
            style.add_panel_label(ax, chr(97 + row * 3 + col), x=-0.18, y=1.06)
            total = f[selected].sum(axis=0)
            box = ax.get_position()
            center, baseline = (box.x0 + box.x1) / 2, box.y0 - 0.070
            if tangent:
                delta = total[:2] / max_force * 0.035
                delta[1] *= 183 / 147
                fig.add_artist(
                    FancyArrowPatch(
                        (center - delta[0] / 2, baseline - delta[1] / 2),
                        (center + delta[0] / 2, baseline + delta[1] / 2),
                        transform=fig.transFigure,
                        arrowstyle="-|>",
                        mutation_scale=5,
                        linewidth=0.7,
                        color="black",
                    )
                )
                label = f"ΣFx = {total[0] * 1000:.3f}; ΣFy = {total[1] * 1000:.3f} mN"
            else:
                label = f"|ΣFz| = {abs(total[2]) * 1000:.3f} mN; n = {selected.sum()}"
            fig.text(center, baseline - 0.020, label, ha="center", fontsize=5)
            profile.append(
                {
                    "method": method,
                    "upper": upper,
                    "all_force_bearing_points": int(selected.sum()),
                    "resultant_n": total.tolist(),
                    "maximum_density_pa": float(density[selected].max()),
                }
            )
    quantity = "Tangential magnitude" if tangent else "Normal magnitude"
    fig.colorbar(scatter, cax=fig.add_axes([0.91, 0.27, 0.018, 0.48]), label=quantity + " / reference area (kPa)")
    return {
        "export": export_figure(fig, stem, style),
        "profile": {
            "panels": profile,
            "shared_xy_limits_mm": [-limit, limit],
            "shared_density_scale_pa": [0, maximum],
            "arrow_scale_mn_per_mm": max_force * 1000 / (0.035 * 183) if tangent else None,
            "arrows_are_net_force_not_local_vectors": tangent,
            "all_force_bearing_samples_included": True,
        },
    }


def history_series(rows):
    series = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        assert len(selected) == 400 and [int(row["step"]) for row in selected] == list(range(1, 401))
        keys = [key for key in selected[0] if key != "method"]
        series[method] = {key: np.array([float(row[key]) for row in selected]) for key in keys}
        assert all(np.isfinite(value).all() for value in series[method].values())
        np.testing.assert_allclose(series[method]["time_s"], np.arange(1, 401) * 0.001, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(series[method]["level"], np.repeat([0, 1], 200))
        assert np.all(series[method]["mass_kg"] > 0)
    assert len(rows) == 1600
    for key in series["dpr"]:
        np.testing.assert_allclose(series["dpr"][key], series["scr"][key], rtol=1e-10, atol=1e-12)
    return {method: series[method] for method in ("dpr", "shell", "reference")}


def render_response(series, stem, style, reference_label):
    fig, axes = plt.subplots(1, 2, figsize=style.mm_to_inches(183, 96))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.24, top=0.80, wspace=0.29)
    palette = style.WONG_PALETTE
    handles, labels, profile = [], [], {}
    for method, name, color, marker in (
        ("dpr", "DPR / SCR mechanics", palette["blue"], "o"),
        ("shell", "Shell, repaired", palette["vermillion"], "s"),
        ("reference", reference_label, palette["bluish_green"], "^"),
    ):
        data = series[method]
        closure, time = data["plate_gap_closure_m"] * 1e6, data["time_s"]
        handle = axes[0].plot(closure, data["upper_reaction_n"] * 1000, color=color, marker=marker, markevery=49)[0]
        axes[0].plot(closure, data["lower_reaction_n"] * 1000, color=color, linestyle=":")
        axes[1].plot(time, -data["tangential_force_x_n"] * 1000, color=color, marker=marker, markevery=49)
        axes[1].plot(time, data["body_force_x_n"] * 1000, color=color, linestyle=":")
        handles.append(handle)
        labels.append(name)
        profile[method] = {
            "rows": len(time),
            "mass_kg": float(data["mass_kg"][0]),
            "closure_range_um": [float(closure.min()), float(closure.max())],
            "final_closure_um": float(closure[-1]),
            "final_body_force_x_n": float(data["body_force_x_n"][-1]),
            "final_opposing_reaction_x_n": float(-data["tangential_force_x_n"][-1]),
            "final_reaction_y_n": float(data["tangential_force_y_n"][-1]),
        }
    axes[0].set(xlabel="Clearance-corrected closure (µm)", ylabel="Normal reaction (mN)", title="Normal compliance")
    axes[1].set(xlabel="Time (s)", ylabel="Tangential force (mN)", title="Sustained tangential support")
    for index, ax in enumerate(axes):
        style.style_axes(ax)
        style.add_panel_label(ax, chr(97 + index), x=-0.13, y=1.10)
        from matplotlib.lines import Line2D

        names = ("Upper plate", "Lower plate") if index == 0 else ("Opposing reaction, −ΣFx", "Body force, actual mass")
        ax.legend(
            [Line2D([], [], color="black"), Line2D([], [], color="black", linestyle=":")],
            names,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.26),
            ncol=2,
        )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.98), ncol=3)
    return {"export": export_figure(fig, stem, style), "profile": profile}
