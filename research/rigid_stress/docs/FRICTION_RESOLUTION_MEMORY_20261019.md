# Friction resolution, exact reference caching and process memory

This checkpoint completes additional numerical and memory controls, not the
[qualified ranking goal](QUALIFIED_STRESS_RANKING_GOAL.md). The independent
fine StVK friction reference remains in progress. No continuum friction error
rank follows from an admitted Shell trajectory or a small timestep change.
The [verified packet](../evidence/20261019-qualified-ranking/friction250-cache-memory-v1/README.md)
retains executed source, inputs, terminal receipts and the reported controls.

## True250um Shell friction control

The earlier `compliant-pad-friction-stability-v1/group1/fine` uses a500um
polar mesh. It must not be identified as the250um configuration used in the
normal ranking. The new `compliant-pad-friction250-history-v1` (Slurm728391)
uses the250um/two-wall-layer mesh and its corresponding physical-pad geometry,
with the same repaired friction source `4ce5542f9104a68ae8b7075abacea2e971bd10ac`.
The free full egg retains E10GPa, nu0.3, density2000, physical pad stiffness
1e9N/m3, mu0.3 and the declared lagged regularized friction law. The two0.1N
levels use a0.05s ramp and0.15s hold each; horizontal acceleration1m/s2 begins
at0.2s. Relative PCG1e-4, absolute velocity tolerance1e-10 and cap8000 are
unchanged. No velocities are cleared and no vertices are pinned.

Both dt1ms/400-step and dt0.5ms/800-step histories complete. Every solver
failure flag is zero; maximum full-history equation residuals are1.36143e-6
and3.71436e-5 against the existing Shell1e-3 gate. Each final50ms normal/shear
window passes reaction, velocity, inertia, kinetic, equation and pad-law checks.
The shear windows additionally pass friction law, instantaneous cone,
sustained tangential balance and external-moment checks. The final body force
is0.007485217264N and opposing tangential reaction is approximately
0.007485217273N, with COM creep about6.47456e-6m/s. This is sustained nonzero
friction, not a shear load that has relaxed away.

`shell-friction250-field-controls-v1` (728445) compares complete fields,
retaining all faces and both wall boundaries. Same-mesh timestep comparison
uses area weighting, rigid corotation and exact Simpson integration of the
linear fiber stress. Relative changes are:

| Endpoint | Tangent | Membrane | Bending | Raw VM | Raw tension |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 3.4623e-9 | 2.6797e-9 | 4.4096e-9 | 4.9524e-10 | 3.3909e-11 |
| Sustained friction | 3.6816e-7 | 2.8361e-7 | 4.7017e-7 | 1.8542e-7 | 3.5574e-9 |

Both pass the unchanged0.5% field/1% peak numerical-sensitivity gates.
Spatial changes are much larger. On common angular columns with25 surface
points per reference face, the500-to250um relative changes in percent are:

| Endpoint | Tangent | Membrane | Bending | Outer | Inner | Raw VM | Raw tension |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 35.0095 | 3.9256 | 55.7060 | 42.9595 | 48.0340 | 4.7745 | 7.4809 |
| Sustained friction | 35.0066 | 3.9253 | 55.7057 | 42.9578 | 48.0313 | 4.7386 | 7.4852 |

The9-point angular check retains the same large changes (normal tangent
34.9947%, bending55.6825%). These are measured configuration sensitivities,
not a continuum-error bound. The published normal ranking remains a comparison
of the explicitly fixed250um Shell configuration against the independently
qualified solid reference; it does not establish mesh-independent Shell
accuracy. Smaller Newton/PCG tolerances or timestep changes cannot stand in
for spatial refinement. Neither these controls nor the normal-reference bounds
qualify the still-pending nonlinear friction reference.

## Exact StVK evaluation cache

The preceding CPU profile identified repeated P2 gradient interpolation as
a substantial cost. The optional `--cache-constitutive-evaluations` stores
only the current exact displacement, its gradient and its nonlinear force
correction. It owns its displacement key and uses `array_equal`, so in-place
state changes invalidate the cache. Returned corrections cannot overwrite it.
Reference geometry/material remain fixed for the model lifetime. Full Piola
force/energy is still independently assembled; no equation tolerance, contact
law, constitutive approximation or stopping cap changes. Default behavior
remains uncached.

`solid-pad-stvk-cache-control-v1` (728385) passes analytic energy, rotation,
translation, small-strain and direct-correction tests, including exact cached
versus uncached equality for repeated and in-place-mutated states and a
returned-array alias test. Its complete400-step normal/friction history and
current-position momentum audit pass. Against the prior direct-correction
run728255, both endpoint complete tangent/membrane/bending errors and raw
VM/tension changes are exactly zero at the retained comparison precision.
Measured dynamics time is913.4129s versus1142.2647s in the earlier uncached
execution. These are individual executions, not a replicated speedup estimate.
The cache is enabled for the separately captured250um reference run728389;
its pending result is excluded from this packet and from qualification.

## Separately admitted process memory

All12 fresh-process profiles in `pad-memory-profile-v1` (728294) and their
own `audit_pad_batch` checks pass. Each observes an entire200-step0.1N
normal-load trajectory, including build/warm/runtime, using every-step stress.
The requested0.1s sampler follows the same process before driver import and
includes non-PyTorch GPU allocations. Sampled peak process GPU memory in MiB:

| Method | B1 full | B1 max | B4 full | B4 max |
| --- | ---: | ---: | ---: | ---: |
| DPR | 2952 | 2952 | 3016 | 2984 |
| SCR | 4490 | 4458 | 7946 | 7946 |
| Shell250um | 1858 | 1858 | 2082 | 2054 |

Actual median sampling intervals span approximately0.12–0.19s; maximum gaps
span3.85–6.53s. These measured samples can miss brief peaks and do not give a
hard capacity/OOM bound. CPU maximum RSS and complete sample records are
retained in each profile. Memory is not inferred from endpoint PyTorch
allocations, and these every-step profiles do not measure sparse-cadence
memory. All profile timing is explicitly non-primary. Native B4 physical-pad
mechanics remains serial per environment within the batched workflow.

The full admitted B1/B4 primary timing aggregation and separate synchronized
phase profiles are tracked independently. No profile can admit a different
primary timing run. Fine friction spatial qualification, corresponding
end-to-end errors/uncertainty and the final cross-case comparison remain open.

## Reproduction and storage

Use the exact packet `tasks.json`, source inventories and captured modules.
`compare_coupled_shell` reconstructs the same-mesh timestep controls;
`contact_fem_exact_columns --shell-indices 0 1 --common-peaks` reconstructs
the Shell mesh comparison at9/25 angular points without claiming solid-volume
weighting for a Shell reference. `check_p2_stvk` includes the cache controls.
`pad_memory_profile --task DRIVER_TASK_JSON` and `audit_pad_batch` reproduce
the distinct profile/admission stages. Run scientific commands through the
existing rtx-mid / GENESIS_IMAGE_VER=1_26 workflow with fresh outputs on data.

The packet contains257 SHA256/size-verified payloads totaling6,571,125 bytes.
Full fields, step journals and active fine-reference outputs remain under the
established data-side ranking root; source and this bounded publication remain
in the worktree. No live writer, environment or compatibility link is moved.
