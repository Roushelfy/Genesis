import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu
from genesis.ext import pyrender
from genesis.ext.pyrender.trackball import Trackball
from genesis.ext.pyrender.viewer import _viewport_index_at, _viewport_local_point, _viewport_rects


@pytest.mark.parametrize(
    ("viewport_size", "expected"),
    [
        ((1280, 480), ((0, 0, 640, 480), (640, 0, 640, 480))),
        ((1281, 480), ((0, 0, 640, 480), (641, 0, 640, 480))),
    ],
)
def test_dual_viewport_rects(viewport_size, expected):
    assert _viewport_rects(viewport_size, 2) == expected


def test_single_viewport_rect_covers_window():
    assert _viewport_rects((641, 479), 1) == ((0, 0, 641, 479),)


def test_pointer_routing_uses_right_viewport_origin():
    viewport_size = (1281, 480)
    right_viewport = _viewport_rects(viewport_size, 2)[1]

    assert _viewport_index_at(640, viewport_size, 2) == 0
    assert _viewport_index_at(641, viewport_size, 2) == 1
    np.testing.assert_array_equal(_viewport_local_point(650, 11, right_viewport), [9, 11])


def test_viewer_options_default_to_one_viewport():
    options = gs.options.ViewerOptions()

    assert options.viewer_count == 1
    assert options.secondary_camera_pos == (3.5, -0.5, 2.5)


def test_viewer_options_reject_unsupported_count():
    with pytest.raises(gs.GenesisException, match="input should be 1 or 2"):
        gs.options.ViewerOptions(viewer_count=3)


def test_right_drag_across_divider_keeps_primary_trackball_unchanged():
    viewer = pyrender.Viewer.__new__(pyrender.Viewer)
    viewer._viewport_size = (1280, 480)
    viewer._viewer_count = 2
    viewer._active_viewport = None
    lookat = np.array((0.0, 0.0, 0.5))
    up = np.array((0.0, 0.0, 1.0))
    viewer._trackball = Trackball(
        gu.pos_lookat_up_to_T(np.array((3.5, 0.5, 2.5)), lookat, up),
        (640, 480),
        1.0,
        lookat,
    )
    viewer._secondary_trackball = Trackball(
        gu.pos_lookat_up_to_T(np.array((3.5, -0.5, 2.5)), lookat, up),
        (640, 480),
        1.0,
        lookat,
    )
    viewer._viewer_flags = {"mouse_pressed": False, "use_perspective_cam": True}
    primary_pose = viewer._trackball.pose.copy()
    secondary_pose = viewer._secondary_trackball.pose.copy()

    viewer.on_mouse_press(900, 240, pyrender.viewer.pyglet.window.mouse.RIGHT, 0)
    viewer.on_mouse_drag(500, 280, -400, 40, pyrender.viewer.pyglet.window.mouse.RIGHT, 0)

    np.testing.assert_array_equal(viewer._trackball.pose, primary_pose)
    assert not np.array_equal(viewer._secondary_trackball.pose, secondary_pose)
    assert viewer._active_viewport == 1
