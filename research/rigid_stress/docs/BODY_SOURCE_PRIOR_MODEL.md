# Explicit body source-prior traction law

`RigidStressOptions(intrinsic_load_partition="body_source_prior")` selects an
alternative finite-area law for intrinsic recovery. The default remains
`"componentwise"`, with the earlier individually conserved point wrenches.
This option changes the physical load representation and requires a new
performance baseline. It is not a solver tolerance relaxation.

For every actual contact i on one stress-observed link in one environment,
retain its original force F_i, application point p_i, free moment M_i,
reference anchor a_i, radius R_i and actual friction coefficient mu_i.
Construct its own topology-aware disk and positive area/profile weights
alpha_iq using the existing integration law and true inward material normals.
No geometric adjacency is created between separate disks. Contact positions,
counts, directions, radii and coefficients remain runtime inputs.

With P_i=sum_q alpha_iq, P=sum_i P_i, w_iq=alpha_iq/P and
g_iq=alpha_iq F_i/P_i, solve

```
minimize sum_iq ||f_iq - g_iq||² / (2 w_iq)
subject to sum_iq f_iq = sum_i F_i
           sum_iq (x_iq-a_0)×f_iq
             = sum_i [M_i + (p_i-a_0)×F_i]
           f_iq·n_iq >= 0
           ||f_iq-(f_iq·n_iq)n_iq|| <= mu_i (f_iq·n_iq)
```

The original point wrenches identify the local priors; their complete sum
identifies the conserved constraint. This is an explicit partition model
for the link's aggregate wrench, including different collision pairs and
separate surface islands. Individual recovered disk resultants may differ
from their priors. Every disk keeps its own geometry, radius, friction and
normal, and every actual contact remains in the model. Source forces are not
clipped or erased. Signed P2 basis values scatter the resulting physical
tractions onto the existing elastic mesh.

Unlike a prior formed only from the net force, this objective retains local
compression when opposed contact forces cancel. The zero-wrench regression
uses two opposing 0.05 N loads and requires nonzero source forces and global
stress. The original material/mass parameters still identify the shared
elastic operators; recovered stress does not alter rigid motion.

Production snapshots, profile reductions, group wrench construction, dual
fitting, Coulomb projection, acceptance and scatter use Quadrants. Python
selects configuration and sizes scratch storage. Actual contact count is
read from the existing packer; when necessary the complete workspace grows
before any source is loaded. It does not truncate contacts at the configured
initial block size. All physical sources remain valid during the auxiliary
constraint partition. Raw original forces, points and free moments stay
unchanged; source moments committed back to raw metadata remain their
original values, rather than the aggregate working target.

The fit is conditioned with an original-source strength that survives net
cancellation. Internal termination remains 9e-9, and final wrench/cone
acceptance remains independent. The final force budget uses 1e-8 times the
aggregate force norm, or the retained source strength when it is exactly
zero. Moment acceptance is about the unchanged leader application point,
with the original physical radius. Complete equilibrium, all six gauge rows,
independent full stress fields and global peak retain their existing budgets.
If disk integration is refined, its prior normalization is refreshed before
retry. Warm multipliers are initial guesses only; current inputs and final
results still receive complete validation.

Source ambiguity, geometry errors, unresolved integration, numerical failure
and strictly incompatible continuous friction cones retain diagnosed failure
semantics. An invalid member invalidates the aggregate observation; diagnostics
prefer its specific source/integration error over a generic group fit error.
The first preserved failure includes all member count and original summed
force/moment, expressed about that diagnostic's original application point.
The unchanged original loads remain available for further auditing.

The native model and sampling frontier together pass 6 CPU cases in 104.59 s
and 6 CUDA cases in 184.10 s. They include 24 independent full-field samples:
worst normalized wrench error 7.387e-10, complete equilibrium budget fraction
0.045857 and peak difference 3.694e-7 Pa. All 56 legacy cases pass on CPU and
CUDA before the frontier merge; the complete pre-merge CUDA suite passes
62 cases in 828.97 s. Its CPU suite passes 61 cases with one diagnostic-test
classification mismatch (strict continuous wrench code36 instead of strict
continuous force code35); the corrected test accepts either physical proof
without changing numerical acceptance and passes after merging the frontier.

An initial B1024 copied-bank policy rollout completes three 1280-step repeats
and 1184 resets each, with zero invalid steps or overflow. It measures
25440.881169 valid env-step/s and 24.844611 batch-step/s on RTX PRO6000
Blackwell Server Edition before combining the frontier. This is inference
rollout throughput. There is no valid old-model policy speedup ratio because
the old run aborted. The policy maximum complete residual budget fraction
is 0.005985.

The full archive, original/failed attempts, source closures, JUnit numerical
properties, independent oracle, policy logs and verifier are retained in
[`native-source-priors-checkpoint-v1`](../evidence/20261011-nonconvex/native-source-priors-checkpoint-v1/manifest.json):
950 entries, 138 unique blobs, 3,154,183 bytes. Manifest SHA256:
`b4ce87c709c5106b0c6dce7a9c628fc1cd713077437516599f0bde83b7868ae9`.

Broader random conditions expose unresolved warmup contacts: B2048/seed623102
env1914 has numerical-fit status33, R=0.00409532256 m, mu=0.564764758;
B4096/seed623103 env2174 has sampled-infeasible/continuous-unresolved status34,
R=0.00408657883 m, mu=0.709940432. Both are single-contact groups. Neither
run has a valid throughput result. Failure capture and independent CPU
classification are ongoing; they are not declared unsupported or physical
incompatibility without a proof. Native contact robustness and the overall
goal remain incomplete. Wider valid trajectories, grasp outcomes, scaling
and a measured optimization plateau are still required.

Earlier isolated evidence is in `NATIVE_SOURCE_PRIORS_20261012.md`. Same-mesh
agreement does not establish physical mesh convergence, and no such study is
run in this phase.

Example invocation with the explicit alternative model:

```bash
python examples/rigid/franka_egg_stress.py --envs 32 --steps 2000 --varied \
  --contact-model intrinsic --intrinsic-load-partition body_source_prior \
  --egg-geometry mesh64 --collision-method gjk --contact-patch on
```
