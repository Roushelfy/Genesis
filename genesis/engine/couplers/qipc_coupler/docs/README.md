# QIPCCoupler

QIPC coupler integrates [cuda-graph-qipc](https://github.com/Genesis-Embodied-AI/cuda-graph-qipc) as the physics backend for rigid, articulated, volumetric FEM, cloth, and sealed-gas shell entities in Genesis.

## Installing cuda-graph-qipc

Requirements are Python 3.12, CUDA Toolkit 12.8+, `uv`, and a supported C++
compiler. The coupler requires a current QIPC build with the grouped
`qipc.contact.Adhesion`/`Bond`/`Release` API and `Scene.reset`; legacy wheels
with the flat contact-table API are rejected at scene build. Clone QIPC and
pass the Genesis checkout explicitly:

```bash
git clone https://github.com/Genesis-Embodied-AI/cuda-graph-qipc.git
cd cuda-graph-qipc
python build.py
python build.py --genesis /absolute/path/to/Genesis
```

The first build initializes QIPC's own build environment. `--genesis` then
creates or uses `/absolute/path/to/Genesis/.venv`, installs Genesis there when
needed, and installs the QIPC wheel into that environment. It does not target
whichever virtual environment happens to be active.

Run Genesis from the Genesis checkout: running from `cuda-graph-qipc` makes its
source-only `qipc/` directory shadow the installed wheel and native extension.
Rebuild after C++/CUDA changes, then run gates with the Genesis interpreter:

```bash
cd /absolute/path/to/Genesis
.venv/bin/python -m pytest tests/test_qipc_sealed_gas.py --backend gpu -n 0 -x
```

## Quick Start

```python
import genesis as gs

gs.init(precision="64")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
    ),
    coupler_options=gs.options.QIPCCouplerOptions(
        contact_enable=False,
    ),
)

robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file="urdf/simple/two_cube_revolute.urdf",
        pos=(0, 0, 0.3),
        fixed=True,
    ),
    material=gs.materials.Rigid(
        qipc_abd_kappa=1e8,
        qipc_kappa_pivot=1e5,
        qipc_kappa_axis=1e5,
        qipc_default_kp=500.0,
        qipc_default_kv=50.0,
    ),
)

scene.build()

for _ in range(100):
    robot.control_dofs_position(0.5)
    scene.step()
```

Affine clusters are queued after the FEM/rigid entities are added and before
build. Initial selections and runtime operations use FEM-entity-local element
indices:

```python
cluster = scene.sim.coupler.add_affine_cluster(
    cloth,
    proxy_entity=robot,       # omit for an implicit affine proxy
    proxy_link="tool_link",
    initial_tris=coil_tris,
)
scene.build()

cluster.detach(tris=released_tris)
cluster.join(tris=refrozen_tris)
```

The handle survives `scene.reset()`. Reset restores authored bonds first and
then the `initial_*` membership; runtime join/detach changes are not retained.

See `examples/qipc/` for more examples.

FEM material mapping, sealed-gas runtime semantics, and validation commands are
documented in [fem_design.md](fem_design.md). The frozen post-f249 dispenser,
ringless URDF-only variant, contact and reset contracts, and gs-core composition are
documented in [tape_dispenser_design.md](tape_dispenser_design.md). Current
implementation status and next work are tracked in [roadmap.md](roadmap.md).
