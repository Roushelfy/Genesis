import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.fixture(scope="session")
def grid_sheet_path(asset_tmp_path):
    def make(n_x, n_y, size_x, size_y):
        """Write a flat rectangular sheet in the xy-plane, centered at the origin, each cell split along alternating
        diagonals, and return the path of the mesh file."""
        path = asset_tmp_path / f"shell_grid_{n_x}x{n_y}_{size_x}x{size_y}.obj"
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, n_x + 1) * size_x, np.linspace(-0.5, 0.5, n_y + 1) * size_y)
        verts = np.stack((xs.T.reshape(-1), ys.T.reshape(-1), np.zeros(xs.size)), axis=-1)
        faces = []
        for i in range(n_x):
            for j in range(n_y):
                v00, v10, v11, v01 = (
                    i * (n_y + 1) + j,
                    (i + 1) * (n_y + 1) + j,
                    (i + 1) * (n_y + 1) + j + 1,
                    i * (n_y + 1) + j + 1,
                )
                faces += [(v00, v10, v11), (v00, v11, v01)] if (i + j) % 2 == 0 else [(v00, v10, v01), (v10, v11, v01)]
        trimesh.Trimesh(verts, np.array(faces), process=False).export(path)
        return str(path)

    return make


@pytest.fixture
def lever_mjcf():
    # A bar hinged about the y-axis at the origin of its body, with a ball at its free end, 0.2 m away
    mjcf = ET.Element("mujoco", model="lever")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="lever", pos="0 0 0.3")
    ET.SubElement(body, "joint", name="hinge", type="hinge", axis="0 1 0")
    ET.SubElement(body, "geom", type="box", size="0.1 0.01 0.005", pos="0.1 0 0", density="1000")
    ET.SubElement(body, "geom", type="sphere", size="0.01", pos="0.2 0 0", density="1000")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_membrane_stretching_stiffness(n_envs, grid_sheet_path, show_viewer):
    # A strip hanging from its top edge stretches under its own weight by rho * g * L^2 / (2 * E), for a Poisson's
    # ratio of zero which leaves its width free. The Green strain stiffens the strip at finite strain, by 1.5 times
    # the strain at most, so the strain is kept below 1e-3.
    GRAVITY, LENGTH, E, RHO = 9.81, 0.4, 1e7, 1000.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.8),
        ),
        show_viewer=show_viewer,
    )
    strip = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(16, 4, LENGTH, 0.1),
            pos=(0.0, 0.0, 0.8),
            euler=(0.0, 90.0, 0.0),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            E=E,
            nu=0.0,
            damping=0.05,
        ),
    )
    scene.build(n_envs=n_envs)

    init_verts = strip.init_verts
    verts_top = np.flatnonzero(init_verts[:, 2] > init_verts[:, 2].max() - gs.EPS)
    verts_bottom = np.flatnonzero(init_verts[:, 2] < init_verts[:, 2].min() + gs.EPS)
    strip.fix_verts(verts_top)
    for _ in range(80):
        scene.step()

    verts_pos = tensor_to_array(strip.get_verts_pos())
    elongation = init_verts[verts_bottom, 2].mean() - verts_pos[..., verts_bottom, 2].mean(axis=-1)
    assert_allclose(elongation, RHO * GRAVITY * LENGTH**2 / (2.0 * E), rtol=2e-3)
    assert_allclose(strip.get_verts_vel(), 0.0, atol=1e-5)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
def test_stiff_sheet_bending(grid_sheet_path, show_viewer):
    # Two stiff cantilevered strips, one at the origin and one far from it, sag under their own weight. Zeroing the
    # velocities every step relaxes them to their static deflection, the beam deflection rho * g * h * L^4 / (8 * D)
    # of the free length L. The dihedral hinges make a strip on a structured mesh softer than the beam by about 20%,
    # the geometric nonlinearity of a 20% deflection partly compensating. The far strip must sag exactly as much,
    # which single precision positions far from the origin could not resolve.
    GRAVITY, LENGTH, E, THICKNESS, RHO = 9.81, 0.4, 1e9, 2e-3, 1000.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.1,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.8, 1.2),
            camera_lookat=(0.0, 0.0, 0.95),
        ),
        show_viewer=show_viewer,
    )
    strips = [
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=grid_sheet_path(40, 10, LENGTH, 0.1),
                pos=(pos_x, 0.0, 1.0),
            ),
            material=gs.materials.Shell(
                rho=RHO,
                E=E,
                nu=0.0,
                thickness=THICKNESS,
            ),
        )
        for pos_x in (0.0, 100.0)
    ]
    scene.build()

    verts_tip = []
    for strip in strips:
        init_verts = strip.init_verts
        # Fixing the first two columns of vertices clamps the strip
        strip.fix_verts(np.flatnonzero(init_verts[:, 0] < init_verts[:, 0].min() + 0.011))
        verts_tip.append(np.flatnonzero(init_verts[:, 0] > init_verts[:, 0].max() - 1e-3))
    for _ in range(30):
        scene.step()
        for strip in strips:
            strip.set_verts_vel(0.0)

    deflections = [
        1.0 - tensor_to_array(strip.get_verts_pos())[verts, 2].mean() for strip, verts in zip(strips, verts_tip)
    ]
    assert_allclose(deflections[1], deflections[0], atol=1e-5)
    free_length = LENGTH - 0.01
    beam_deflection = RHO * GRAVITY * THICKNESS * free_length**4 / (8.0 * E * THICKNESS**3 / 12.0)
    assert_allclose(deflections[0], 1.2 * beam_deflection, rtol=0.03)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_tearing_at_tensile_strength(grid_sheet_path, show_viewer):
    # Three strips are pulled apart at their ends, faster in the second environment. Fracture splits a strip once its
    # membrane stress E * G (Green strain G, Poisson's ratio zero) reaches the tensile strength, and the weak strip
    # breaks into separate pieces while the strong one stays whole. The third strip is as weak but evaluates its damage
    # alone, so that it reports the same damage index as the weak one up to their failure, then keeps its mesh.
    # Resetting an environment restores its mesh.
    E, TENSILE_STRENGTH, LENGTH = 1e6, 3e4, 0.4
    PULL_SPEEDS = (0.05, 0.1)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.0, 1.0),
            camera_lookat=(0.0, 0.15, 0.5),
        ),
        show_viewer=show_viewer,
    )
    strips = [
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=grid_sheet_path(16, 4, LENGTH, 0.1),
                pos=(0.0, 0.3 * i, 0.5),
            ),
            material=gs.materials.Shell(
                E=E,
                nu=0.0,
                tensile_strength=tensile_strength,
                fracture=fracture,
            ),
        )
        for i, (tensile_strength, fracture) in enumerate(
            ((TENSILE_STRENGTH, True), (1e3 * TENSILE_STRENGTH, True), (TENSILE_STRENGTH, False))
        )
    ]
    scene.build(n_envs=2)

    for strip in strips:
        init_verts = strip.init_verts
        verts_left = np.flatnonzero(init_verts[:, 0] < init_verts[:, 0].min() + gs.EPS)
        verts_right = np.flatnonzero(init_verts[:, 0] > init_verts[:, 0].max() - gs.EPS)
        strip.fix_verts(np.concatenate((verts_left, verts_right)))
        for i_b, speed in enumerate(PULL_SPEEDS):
            strip.set_verts_vel((-0.5 * speed, 0.0, 0.0), verts_left, envs_idx=[i_b])
            strip.set_verts_vel((0.5 * speed, 0.0, 0.0), verts_right, envs_idx=[i_b])

    def count_pieces(strip):
        faces = tensor_to_array(strip.get_faces())
        n_pieces = []
        for faces_env in faces:
            adjacency = sp.coo_matrix(
                (np.ones(faces_env.size), (faces_env.reshape(-1), faces_env[:, (1, 2, 0)].reshape(-1))),
                shape=(strip.n_verts_max, strip.n_verts_max),
            )
            _, labels = csgraph.connected_components(adjacency, directed=False)
            n_pieces.append(len(np.unique(labels[faces_env])))
        return np.array(n_pieces)

    # The strip breaks at the strain whose Green strain reaches TENSILE_STRENGTH / E
    strain_break = np.sqrt(1.0 + 2.0 * TENSILE_STRENGTH / E) - 1.0
    steps_break = [strain_break * LENGTH / speed / scene.sim.dt for speed in PULL_SPEEDS]
    for i in range(int(1.15 * max(steps_break))):
        scene.step()
        n_pieces = count_pieces(strips[0])
        strain = (i + 1) * scene.sim.dt * np.array(PULL_SPEEDS) / LENGTH
        damage = tensor_to_array(strips[2].get_peak_damage())
        for i_b in range(2):
            if i + 1 < 0.9 * steps_break[i_b]:
                assert n_pieces[i_b] == 1
                assert_allclose(strips[0].get_peak_damage()[i_b], damage[i_b], tol=1e-9)
            if 0.5 * steps_break[i_b] < i + 1 < 0.9 * steps_break[i_b]:
                assert_allclose(damage[i_b], E * (strain[i_b] + 0.5 * strain[i_b] ** 2) / TENSILE_STRENGTH, tol=0.02)
    assert (count_pieces(strips[0]) >= 2).all()
    assert_equal(count_pieces(strips[1]), 1)
    assert_equal(strips[1].get_n_verts(), strips[1].n_verts)
    assert (tensor_to_array(strips[2].get_peak_damage()) > 1.0).all()
    assert (tensor_to_array(strips[2].get_failure_face()) >= 0).all()
    assert_equal(count_pieces(strips[2]), 1)
    assert_equal(strips[2].get_faces(), np.broadcast_to(strips[2].init_faces, (2, *strips[2].init_faces.shape)))

    scene.reset(envs_idx=[1])
    assert_equal(count_pieces(strips[0]), (count_pieces(strips[0])[0], 1))
    assert_equal(strips[0].get_n_verts()[1], strips[0].n_verts)
    assert_equal(strips[0].get_faces()[1], strips[0].init_faces)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_contact_forces(n_envs, grid_sheet_path, lever_mjcf, show_viewer):
    # A sheet dropped on a free box resting on the ground settles half its thickness above the box top, and the
    # ground then carries the weight of both. Aside, a ball rests inside a triangle of a sheet pinned at its corners,
    # away from its vertices and edges: the triangle holds it half the sheet thickness above its surface and carries its
    # weight. Further aside, the ball at the end of a hinged lever rests on another pinned triangle, whose reaction
    # balances the moment of the weight of the lever about its hinge. Last, two boxes rest on a pinned sheet inclined by
    # THETA. The one whose friction coefficient exceeds tan(THETA) sticks, the other slides down at the acceleration
    # g * (sin(THETA) - mu * cos(THETA)) of Coulomb friction, its front edge crossing a row of vertices of the sheet.
    GRAVITY, SIZE, THICKNESS, RHO, BOX_HEIGHT, BALL_RADIUS = 9.81, 0.15, 2e-3, 1000.0, 0.05, 0.01
    THETA, BOX_SIZE = np.deg2rad(20.0), 0.06
    FRICTIONS = (0.6, 0.2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -2.5, 1.2),
            camera_lookat=(1.5, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, BOX_HEIGHT),
            pos=(0.0, 0.0, 0.5 * BOX_HEIGHT),
        ),
        material=gs.materials.Rigid(
            rho=100.0,
        ),
    )
    sheet = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(10, 10, SIZE, SIZE),
            pos=(0.0, 0.0, BOX_HEIGHT + 0.01),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            E=1e6,
            thickness=THICKNESS,
            damping=0.01,
        ),
    )
    # The first triangle of a one-cell sheet spans its corners (-0.05, -0.05), (0.05, -0.05) and (0.05, 0.05)
    support = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(1, 1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.2),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            thickness=THICKNESS,
        ),
    )
    ball_center = np.array((1.0 + 0.05 / 3.0, -0.05 / 3.0, 0.2 + 0.5 * THICKNESS + BALL_RADIUS))
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=BALL_RADIUS,
            pos=ball_center,
        ),
    )
    # The ball of the lever, at (2.2, 0.0, 0.3) once placed, lies above the centroid of the first triangle
    lever = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=lever_mjcf,
            pos=(2.0, 0.0, 0.0),
        ),
    )
    lever_support = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(1, 1, 0.1, 0.1),
            pos=(2.2 - 0.05 / 3.0, 0.05 / 3.0, 0.3 - BALL_RADIUS - 0.5 * THICKNESS),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            thickness=THICKNESS,
        ),
    )
    incline = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(16, 16, 0.4, 0.4),
            pos=(3.0, 0.0, 0.2),
            euler=(np.rad2deg(THETA), 0.0, 0.0),
        ),
        material=gs.materials.Shell(
            thickness=THICKNESS,
        ),
    )
    normal = np.array((0.0, -np.sin(THETA), np.cos(THETA)))
    downhill = np.array((0.0, -np.cos(THETA), -np.sin(THETA)))
    incline_boxes = [
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
                pos=np.array((pos_x, 0.0, 0.2)) - 0.05 * downhill + (0.5 * THICKNESS + 0.5 * BOX_SIZE) * normal,
                euler=(np.rad2deg(THETA), 0.0, 0.0),
            ),
            material=gs.materials.Rigid(
                rho=500.0,
                coup_friction=friction,
            ),
        )
        for pos_x, friction in zip((2.9, 3.1), FRICTIONS)
    ]
    scene.build(n_envs=n_envs)
    support.fix_verts()
    lever_support.fix_verts()
    incline.fix_verts()

    incline_boxes_vel = []
    for i in range(100):
        scene.step()
        if i + 1 in (50, 100):
            incline_boxes_vel.append([tensor_to_array(incline_box.get_vel()) for incline_box in incline_boxes])

    box_top = tensor_to_array(box.get_pos())[..., 2] + 0.5 * BOX_HEIGHT
    verts_height = tensor_to_array(sheet.get_verts_pos())[..., 2] - box_top[..., None]
    assert_allclose(verts_height, 0.5 * THICKNESS, atol=5e-4)
    sheet_mass = RHO * THICKNESS * SIZE**2
    ground_force = -tensor_to_array(plane.get_links_net_contact_force())[..., 0, 2]
    assert_allclose(ground_force, (box.get_mass() + sheet_mass) * GRAVITY, rtol=2e-3)
    assert_allclose(ball.get_pos(), np.broadcast_to(ball_center, ball.get_pos().shape), atol=1e-5)
    ball_force = tensor_to_array(ball.get_links_net_contact_force())[..., 0, :]
    assert_allclose(ball_force[..., :2], 0.0, atol=1e-9)
    assert_allclose(ball_force[..., 2], ball.get_mass() * GRAVITY, rtol=2e-3)
    lever_link = lever.get_link("lever")
    assert_allclose(lever.get_dofs_velocity(), 0.0, atol=1e-6)
    lever_force = tensor_to_array(lever.get_links_net_contact_force())[..., lever_link.idx_local, 2]
    hinge_x = tensor_to_array(lever.get_joint("hinge").get_anchor_pos())[..., 0]
    lever_com = lever.get_links_pos(ref=gs.link_ref_frame.link_COM)
    lever_arm = tensor_to_array(lever_com)[..., lever_link.idx_local, 0] - hinge_x
    lever_ball = next(geom for geom in lever_link.geoms if geom.type == gs.GEOM_TYPE.SPHERE)
    lever_length = tensor_to_array(lever_ball.get_pos())[..., 0] - hinge_x
    assert_allclose(lever_force, tensor_to_array(lever.get_mass()) * GRAVITY * lever_arm / lever_length, rtol=2e-3)
    assert_allclose(incline_boxes_vel[1][0], 0.0, atol=1e-4)
    accel = GRAVITY * (np.sin(THETA) - FRICTIONS[1] * np.cos(THETA))
    slide_vel_change = incline_boxes_vel[1][1] - incline_boxes_vel[0][1]
    assert_allclose(slide_vel_change @ downhill, accel * 50 * scene.sim.dt, rtol=5e-3)
    assert_allclose(slide_vel_change @ normal, 0.0, atol=1e-4)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_contact_conserves_momentum(n_envs, grid_sheet_path, show_viewer):
    # Without gravity, a fast ball hits a free sheet off its center: the contact impulses on the ball and the sheet are
    # opposite, so that the momentum of the pair is conserved through the impact, up to the residual of the linear
    # solves, which the tight tolerances bring to the floating-point floor. The ball, which crosses the thickness of the
    # sheet within a substep, stays above it without any vertex of the sheet inside it.
    SIZE, THICKNESS, RHO, BALL_RADIUS = 0.2, 2e-3, 1000.0, 0.02

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        shell_options=gs.options.ShellOptions(
            pcg_tolerance=1e-10,
            pcg_velocity_tolerance=1e-10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.4, 0.3),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    sheet = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(8, 8, SIZE, SIZE),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            E=1e6,
            thickness=THICKNESS,
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=BALL_RADIUS,
            pos=(0.05, 0.02, 0.03),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    scene.build(n_envs=n_envs)
    ball.set_dofs_velocity([0.0, 0.0, -3.0, 0.0, 0.0, 0.0])

    # Lumped vertex masses: a third of the mass of every adjacent triangle
    init_verts, init_faces = sheet.init_verts, sheet.init_faces
    faces_area = 0.5 * np.linalg.norm(
        np.cross(
            init_verts[init_faces[:, 1]] - init_verts[init_faces[:, 0]],
            init_verts[init_faces[:, 2]] - init_verts[init_faces[:, 0]],
        ),
        axis=-1,
    )
    verts_mass = np.bincount(init_faces.reshape(-1), np.repeat(RHO * THICKNESS * faces_area / 3.0, 3), sheet.n_verts)
    ball_mass = tensor_to_array(ball.get_mass())
    momentum_start = ball_mass * tensor_to_array(ball.get_vel())
    for _ in range(40):
        scene.step()

    verts_pos = tensor_to_array(sheet.get_verts_pos())[..., : sheet.n_verts, :]
    verts_vel = tensor_to_array(sheet.get_verts_vel())[..., : sheet.n_verts, :]
    ball_pos = tensor_to_array(ball.get_pos())
    assert np.linalg.norm(verts_pos - ball_pos[..., None, :], axis=-1).min() > BALL_RADIUS + 0.5 * THICKNESS - 1e-4
    assert (ball_pos[..., 2] > verts_pos[..., 2].min(axis=-1)).all()
    assert_allclose(verts_mass @ verts_vel + ball_mass * tensor_to_array(ball.get_vel()), momentum_start, atol=1e-9)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_state_restore_reproduces_contact(n_envs, grid_sheet_path, show_viewer):
    # A ball rolls and slides on a sheet pinned at its edges, which it sags into. Restoring a snapshot replays the
    # motion that followed it, the warm start of the solver included, and so do restoring a checkpoint of the scene and
    # loading a copy of the scene pickled there. Resetting one environment to the initial state replays its motion from
    # the start while the other one carries on.
    THICKNESS, BALL_RADIUS = 2e-3, 0.02

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.4, 0.3),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    sheet = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(8, 8, 0.2, 0.2),
        ),
        material=gs.materials.Shell(
            E=1e6,
            thickness=THICKNESS,
            damping=0.01,
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=BALL_RADIUS,
            pos=(0.0, 0.0, 0.5 * THICKNESS + BALL_RADIUS),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
            coup_friction=0.5,
        ),
    )
    scene.build(n_envs=n_envs)
    init_verts = sheet.init_verts
    sheet.fix_verts(np.flatnonzero(np.abs(init_verts[:, :2]).max(axis=-1) > 0.1 - gs.EPS))
    ball.set_dofs_velocity([0.2, 0.1, 0.0, 0.0, 0.0, 3.0])
    state_init = scene.get_state()

    def run(scene, n_steps):
        sheet, ball = scene.entities
        trajectory = []
        for _ in range(n_steps):
            scene.step()
            trajectory.append((tensor_to_array(sheet.get_verts_pos()), tensor_to_array(ball.get_pos())))
        return trajectory

    trajectory_ref = run(scene, 30)
    scene.reset(state=state_init)
    for (sheet_a, ball_a), (sheet_b, ball_b) in zip(trajectory_ref[:15], run(scene, 15)):
        assert_allclose(sheet_b, sheet_a, tol=1e-9)
        assert_allclose(ball_b, ball_a, tol=1e-9)
    state_mid = scene.get_state()
    checkpoint_mid = scene.__getstate__()
    scene_mid_bytes = pickle.dumps(scene)
    for (sheet_a, ball_a), (sheet_b, ball_b) in zip(trajectory_ref[15:], run(scene, 15)):
        assert_allclose(sheet_b, sheet_a, tol=1e-9)
        assert_allclose(ball_b, ball_a, tol=1e-9)
    scene.reset(state=state_mid)
    for (sheet_a, ball_a), (sheet_b, ball_b) in zip(trajectory_ref[15:], run(scene, 15)):
        assert_allclose(sheet_b, sheet_a, tol=1e-9)
        assert_allclose(ball_b, ball_a, tol=1e-9)
    scene.__setstate__(checkpoint_mid)
    for (sheet_a, ball_a), (sheet_b, ball_b) in zip(trajectory_ref[15:], run(scene, 15)):
        assert_allclose(sheet_b, sheet_a, tol=1e-9)
        assert_allclose(ball_b, ball_a, tol=1e-9)
    scene_mid = pickle.loads(scene_mid_bytes)
    for (sheet_a, ball_a), (sheet_b, ball_b) in zip(trajectory_ref[15:], run(scene_mid, 15)):
        assert_allclose(sheet_b, sheet_a, tol=1e-9)
        assert_allclose(ball_b, ball_a, tol=1e-9)
    if n_envs > 0:
        scene.reset(state=state_mid)
        scene.reset(state=state_init, envs_idx=[1])
        for (sheet_a, ball_a), (sheet_c, ball_c), (sheet_b, ball_b) in zip(
            trajectory_ref[:15], trajectory_ref[15:], run(scene, 15)
        ):
            assert_allclose(sheet_b[1], sheet_a[1], tol=1e-9)
            assert_allclose(ball_b[1], ball_a[1], tol=1e-9)
            assert_allclose(sheet_b[0], sheet_c[0], tol=1e-9)
            assert_allclose(ball_b[0], ball_c[0], tol=1e-9)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_plastic_stretching(grid_sheet_path, show_viewer):
    # Two strips are stretched by 10% and released. The one whose yield stress the stretching exceeds keeps a
    # permanent elongation, the elastic one recovers its length.
    E, LENGTH = 1e6, 0.4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.0, 1.0),
            camera_lookat=(0.0, 0.15, 0.5),
        ),
        show_viewer=show_viewer,
    )
    strips = [
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=grid_sheet_path(16, 4, LENGTH, 0.1),
                pos=(0.0, 0.3 * i, 0.5),
            ),
            material=gs.materials.Shell(
                E=E,
                nu=0.0,
                damping=0.02,
                yield_stress=yield_stress,
            ),
        )
        for i, yield_stress in enumerate((0.02 * E, None))
    ]
    scene.build()

    verts_ends = []
    for strip in strips:
        init_verts = strip.init_verts
        verts_left = np.flatnonzero(init_verts[:, 0] < init_verts[:, 0].min() + gs.EPS)
        verts_right = np.flatnonzero(init_verts[:, 0] > init_verts[:, 0].max() - gs.EPS)
        strip.fix_verts(np.concatenate((verts_left, verts_right)))
        strip.set_verts_vel((-0.2, 0.0, 0.0), verts_left)
        strip.set_verts_vel((0.2, 0.0, 0.0), verts_right)
        verts_ends.append((verts_left, verts_right))
    for _ in range(100):
        scene.step()
    for strip in strips:
        strip.release_verts()
        strip.set_verts_vel(0.0)
    for _ in range(200):
        scene.step()

    lengths = []
    for strip, (verts_left, verts_right) in zip(strips, verts_ends):
        verts_pos = tensor_to_array(strip.get_verts_pos())
        lengths.append(verts_pos[verts_right, 0].mean() - verts_pos[verts_left, 0].mean())
    assert lengths[0] > 1.03 * LENGTH
    assert_allclose(lengths[1], LENGTH, rtol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_unsupported_shell_inputs(grid_sheet_path, asset_tmp_path):
    with pytest.raises(gs.GenesisException, match="Poisson"):
        gs.materials.Shell(nu=0.5)

    # Two triangles sharing a single vertex form two fans around it.
    bowtie_path = asset_tmp_path / "shell_bowtie.obj"
    trimesh.Trimesh(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0]], dtype=float),
        np.array([[0, 1, 2], [0, 3, 4]]),
        process=False,
    ).export(bowtie_path)
    scene = gs.Scene()
    with pytest.raises(gs.GenesisException, match="not manifold"):
        scene.add_entity(morph=gs.morphs.Mesh(file=str(bowtie_path)), material=gs.materials.Shell())

    scene = gs.Scene(coupler_options=gs.options.SAPCouplerOptions())
    scene.add_entity(morph=gs.morphs.Mesh(file=grid_sheet_path(2, 2, 0.1, 0.1)), material=gs.materials.Shell())
    with pytest.raises(gs.GenesisException, match="LegacyCouplerOptions"):
        scene.build()
