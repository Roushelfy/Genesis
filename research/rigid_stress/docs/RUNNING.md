# Run native rigid stress recovery

Install Genesis and its normal dependencies using the repository README.
The example and benchmark use the native rigid solver without a research
backend. The [API](NATIVE_API.md) defines configuration, max/full observations,
units, reset behavior and supported loads. The
[source-prior model](BODY_SOURCE_PRIOR_MODEL.md) records footprint assumptions;
[application results](PRACTICAL_APPLICATION_RESULTS_20261016.md) retain physical
uncertainty and phase-dependent reward limitations.
For the optional thin-shell DPR backend, use the
[shell running guide](RUNNING_NATIVE_SHELL_DPR.md) and its matched condition
fixture. Shell outputs are in-plane membrane/bending stress with separate
resultants; volume remains the default.

## Environment and outputs

Run from the repository root. Set `RIGID_STRESS_DATA_ROOT` to a writable data
directory and `RIGID_STRESS_ENV` to the installed Python environment before
sourcing `research/rigid_stress/scripts/cluster_env.sh`. The local scientific
environment is `/mnt/data/zhaofeng/venvs/rigid-stress-physical`; wrappers also
accept `PHYSICAL_VALIDATION_ENV`. Storage locations are documented in
[DATA_LOCATION.md](../../../DATA_LOCATION.md).

On this cluster, all scientific execution uses Slurm and
`GENESIS_IMAGE_VER=1_26 gs-srun`. Create each job script and its stdout/stderr
under a fresh data-side run. The outer script sourcing the shared container
helper uses `set -eo pipefail`: that helper reads optional unset variables.
Set per-job CUDA, Genesis, Quadrants, Numba and temporary paths explicitly.
For example, inside an allocated GPU job:

```bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash "$task_run/workload.sh"
```

The workload script sources `cluster_env.sh` and sets its output/cache paths.
See the [throughput guide](RUNNING_PRACTICAL_THROUGHPUT.md) for a complete
submission script, exact source reconstruction and condition-bank identities.

## Normal example and benchmark

Inside that allocation/container, use the intrinsic source-prior configuration:

```bash
task_bank="$PWD/research/rigid_stress/evidence/20261017-throughput/source-recipes-v1/inputs/conditions.npz"
python examples/rigid/franka_egg_stress.py \
  --envs 8 --steps 600 --varied --seed 623016 --conditions "$task_bank" \
  --contact-model intrinsic \
  --intrinsic-load-partition body_source_prior --egg-geometry mesh64 \
  --collision-method mpr --contact-patch off --output-mode max

python -m examples.speed_benchmark.rigid_stress \
  --scope live --envs 8 --warmup 900 --steps 16 --repetitions 1 \
  --minimum-seconds 0 --varied --seed 623016 --conditions "$task_bank" \
  --contact-model intrinsic \
  --intrinsic-load-partition body_source_prior --history 0 \
  --egg-geometry mesh64 --collision-method mpr --contact-patch off \
  --output "$task_run/benchmark-smoke.json"
```

Both commands replay the accepted eight-condition bank. Randomly generated
`--varied` cases can legitimately fail the finite-footprint load-admissibility
checks; preserve those failures rather than interpreting every random draw as
a qualified application. The second command is a bounded workflow smoke check.
Accepted performance measurements use the frozen bank,900 warmup and three2400-step repeats in
separate `rigid`, `live`, `policy-rigid` and `policy` processes. Recommended
history0/B16384 and measured hardware, memory and capacity limits are in the
[final results](PRACTICAL_THROUGHPUT_RESULTS_20261017.md). Use the complete
[reproduction guide](RUNNING_PRACTICAL_THROUGHPUT.md) for those measurements.
Policy scope measures inference rollout rather than RL training.

`--output-mode full` enables complete corner tensors/von Mises observations;
its storage and cost differ from max mode. `--viewer` enables the example
viewer when a display is available. The example's historical CLI defaults
select the simpler pressure law; pass the options above for the recommendation.

## Regression tests and independent references

All stress tests and fixtures remain under `tests/rigid/`. They cover changing
convex/nonconvex contacts, friction, global maxima/full fields, complete
residuals, history invalidation, environment isolation and partial resets.
The independent CPU and scikit-fem reference modules remain in
`research/rigid_stress/` because those tests import them. Their optional
dependencies are listed in [requirements-reference.txt](../requirements-reference.txt):
it includes the CPU oracle dependencies, tested scikit-fem and ipctk versions
for independent FEM/contact replay. Production use does not require these
reference solvers. [requirements-checks.txt](../requirements-checks.txt) supplies
the pytest tooling.

Inside a Slurm allocation/container, a focused check is:

```bash
python -m pytest -n 0 --backend cpu -q \
  -o cache_dir="$task_run/pytest-cache" --basetemp "$task_run/pytest-tmp" \
  tests/rigid/test_stress.py::test_native_full_field_same_state_and_invalid_solve \
  tests/rigid/test_stress_independent_fem.py \
  tests/rigid/test_stress_pair_geometry.py \
  tests/rigid/test_stress_intrinsic_lifecycle.py
```

Use `--backend cuda` in a GPU allocation for the corresponding device check.
`scripts/check_nonconvex_final.sh` accepts an explicit test selection and
captures source hashes, logs and JUnit results under its fresh run directory.
Unchanged physics does not require repeating historical convergence campaigns.

`test_stress_pair_geometry.py` checks the current-geometry component with
analytic primitive/nonconvex distances, material visibility, current relative
velocity, link-pair wrench grouping and partial reset. The
[retained CPU/GPU runs](../evidence/20261019-qualified-ranking/native-pair-accessible-v1/README.md)
include the exact source patch and test files. These are component checks;
they do not validate an automatic pressure law or measure production throughput.
The accessible-boundary case compares primitive/mesh representations and
reversed component ordering, rejects transverse blockers and checks common
rigid transforms. The primitive case also checks eighteen open-segment queries.

## Evidence and historical tools

Current application reproduction remains in
[RUNNING_PRACTICAL_APPLICATION.md](RUNNING_PRACTICAL_APPLICATION.md).
Its compact packets retain exact inputs, source closures and hashes.
The [archive index](../archive/README.md) locates older reports, standalone
backends, launchers, patches and evidence at a fixed Git revision, with exact
retrieval commands. Restore historical code into an isolated data-side tree;
historical instructions do not replace the current feature contract or goal.
