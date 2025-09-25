import genesis as gs
import numpy as np

def test_link_filter():
    """Test the link_filter functionality"""
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),
            IPC_self_contact=False,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add Franka robot
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # Test link filtering - only include hand and fingers in IPC
    # First, let's see what links are available
    print("Available links in Franka:")
    for i, link in enumerate(franka._links):
        print(f"  {i}: {link.name}")

    # Set filter to only include hand and finger links
    franka.set_ipc_link_filter(link_names=["left_finger", "right_finger"])

    print(f"IPC link filter set to: {franka._ipc_link_filter}")

    scene.build()

    print("Scene built successfully with link filtering!")
    print("Only specified links should participate in IPC simulation.")

    # Run a few steps to test
    for i in range(10):
        scene.step()

    return scene

if __name__ == "__main__":
    scene = test_link_filter()