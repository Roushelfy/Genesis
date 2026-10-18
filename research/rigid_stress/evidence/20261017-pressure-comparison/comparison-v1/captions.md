# Figure captions and manuscript handoff

## Performance and memory

Claim anchor: `pressure-comparison-performance`.
Placement: comparison report, after the scaling/overhead tables.

**Paired native rollout scaling and separate memory diagnostics.**
(a,b) Median valid aggregate environment steps per second for scripted and
policy-inference rollouts on the fixed480-P2-tet/2430-DOF Franka/egg workload.
Each width runs rigid, pressure and intrinsic source-prior scopes in separate
processes on one allocated RTX PRO6000 Blackwell GPU, after900 warmup steps.
Points/ranges retain all three2400-step repeats and their minimum/maximum,
rather than confidence intervals; near-identical repeats can overlap.
Each full timed repeat includes3B partial resets and has zero invalid
environment steps. (c) Incremental live stress batch latency subtracts the
same-allocation rigid median. (d) Separate100ms NVML process peaks include
setup/growth and are sampled lower bounds. Memory observations, cold/JIT,
traces and single-repeat capacity runs do not contribute headline rates.
Pressure is the literal max/history0 public default; intrinsic changes the
load model to `body_source_prior`. Both use the same Q10/FP64 physical inputs.
Blue circles/solid lines denote pressure, orange squares/dashed lines intrinsic,
gray triangles/dotted lines rigid. Width uses a logarithmic axis.

Alt text: Four panels compare scripted throughput, policy inference throughput,
incremental stress latency and sampled process memory as batch width increases
from32 to32768. Markers and line styles distinguish rigid, pressure and
intrinsic source-prior. Numerical values and individual repeats are available
in `comparison.json` and `performance-source.csv`.

Interpretation: Compare the useful large-batch shoulder with the memory/latency
increase. This is workload-specific inference rollout performance and does not
measure full reinforcement-learning training or physical accuracy.

## Model difference and representative cost

Claim anchor: `pressure-comparison-model-error-cost`.
Placement: comparison report, after accuracy/support and rejection evidence.

**Shared egg footprint bias and different local/topological admissibility.**
(a) Four nominal6mm frozen egg states show volume-weighted material-frame
Frobenius tensor change M/L against representative B32 live overhead from the
separate current native rollout. Gray segments pair the same reference state;
they do not represent a fitted curve. Timing and error are different declared
workloads: the x coordinate is not a frozen-state solve time. L is the
full-inertia linear replay of the independent nonlinear contact reference;
M changes its contact loads to a native finite footprint. (b) The separate
3/6mm physical-radius sensitivity retains all four states/model; clustered
points can overlap. Neither model is experimental ground truth.
(c) Exact affine-volume tensor changes on the smooth convex and close-sheet
controls, for opposing (Opp), asymmetric (Asym), near-sliding (Slip) and
multiple-contact (Multi) states, at radii shown in millimeters. Cell labels
give percentages on a shared logarithmic grayscale. Four pressure free-moment
inputs are unsupported (N/A), rather than zero error. Source-prior near-slip
results therefore do not have a pressure counterpart.
(d) Fraction of integrated sample-force norm outside local material-normal
friction/compression cones. The fraction counts violations beyond64 FP64
epsilon times summed original source-force norm; raw violations remain in the
summary. Only matched-supported control cases enter panel(d), while egg includes
both radii/all states. Horizontal offsets separate marks without scientific
meaning. Pressure's own supplied-pad-frame constraints remain a separate
diagnostic. Blue circles and orange squares denote pressure and source-prior.

Alt text: The upper panels show similar egg tensor changes for the two models,
despite a much higher representative intrinsic overhead, and substantial radius
sensitivity. The lower-left matrix shows large pressure changes on the gap
control and unsupported pressure near-slip cases. The lower-right scatter shows
pressure local-normal cone violations in egg/gap cases, while intrinsic remains
at roundoff. Exact values, signed source-wrench changes and raw hotspot locations
are retained in `model-comparison.json` and `accuracy-summary.json`.

Interpretation: Extra intrinsic cost is justified by local admissibility and
support where those matter. It does not eliminate shared egg reconstruction,
feedback, geometry or raw-peak reference uncertainty. Do not turn M/L or a
successful algebraic solve into an absolute physical accuracy claim.

## Whole-episode and lift-window reward

Claim anchor: `pressure-comparison-reward-phase`.
Placement: comparison report, after complete-grasp reward sensitivity.

**Phase-dependent peak ranking on complete recorded grasps.**
Paired raw maximum-stress difference under the two commanded strong-force
limits, peak(4N) minus peak(0.2N), across eight frozen conditions over(a) every phase of the complete1200-step recorded episode
and(b) only the predeclared0.3–0.6 lift phase. Each model/radius retains three
complete episodes after900 warmup; markers show all three paired observations
with small horizontal presentation offsets. Repeated trajectories nearly
coincide, so some marks overlap. Blue filled marks denote pressure; orange
open marks intrinsic source-prior. Circle/triangle distinguishes nominal6/3mm
radius before the unchanged bank/phase schedule. Positive differences favor
0.2N for this raw-peak observable. Model/radius alternatives preserve rank sign
within each phase for8/8 conditions, but conditions0 and3 reverse when the time
scope changes. Both force choices reach at least0.149m in the recorded lift window.
The observed model/radius amplitude bands and counterfactual threshold-decision
intervals are reported separately; no physical threshold or early termination
was fitted or executed.

Alt text: Two scatter panels show raw high-force minus low-force peak differences.
Whole-episode differences are negative in conditions0 and3, whereas all
lift-window differences are positive. Models and radii change amplitude while
retaining within-phase rank signs. `reward-comparison.json` and the four portable
raw NPZ traces retain every recorded step and phase-specific reduction.

Interpretation: A favorable lift-window stress reward does not imply a safer
whole grasp. Sensitivity bands are observed configuration ranges, rather than
physical confidence bounds or calibrated fracture limits.
