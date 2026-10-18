Median valid aggregate env-step/s, n=3 per cell:

| B | Rigid | Pressure live | Intrinsic live | Policy rigid | Pressure policy | Intrinsic policy |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 11,334 | 8,368 | 2,808 | 9,804 | 8,241 | 2,695 |
| 2,048 | 430,880 | 200,668 | 78,029 | 394,689 | 199,596 | 76,587 |
| 8,192 | 751,742 | 273,484 | 105,202 | 740,890 | 272,425 | 104,230 |
| 16,384 | 884,655 | 296,859 | 113,933 | 881,652 | 296,297 | 113,971 |
| 32,768 | 1,000,409 | 312,525 | 118,966 | 998,549 | 312,278 | 118,701 |

Batch / incremental stress latency in ms; time ratio / throughput loss:

| B | Model | Live ms / extra ms | Live ratio / loss | Policy ms / extra ms | Policy ratio / loss |
| --- | --- | --- | --- | --- | --- |
| 32 | Pressure | 3.824 / 1.001 | 1.354 / 26.17% | 3.883 / 0.619 | 1.190 / 15.94% |
| 32 | Intrinsic | 11.396 / 8.573 | 4.036 / 75.22% | 11.874 / 8.610 | 3.638 / 72.51% |
| 2,048 | Pressure | 10.206 / 5.453 | 2.147 / 53.43% | 10.261 / 5.072 | 1.977 / 49.43% |
| 2,048 | Intrinsic | 26.247 / 21.493 | 5.522 / 81.89% | 26.741 / 21.552 | 5.153 / 80.60% |
| 8,192 | Pressure | 29.954 / 19.057 | 2.749 / 63.62% | 30.071 / 19.014 | 2.720 / 63.23% |
| 8,192 | Intrinsic | 77.869 / 66.972 | 7.146 / 86.01% | 78.595 / 67.538 | 7.108 / 85.93% |
| 16,384 | Pressure | 55.191 / 36.671 | 2.980 / 66.44% | 55.296 / 36.712 | 2.976 / 66.39% |
| 16,384 | Intrinsic | 143.803 / 125.283 | 7.765 / 87.12% | 143.755 / 125.172 | 7.736 / 87.07% |
| 32,768 | Pressure | 104.849 / 72.094 | 3.201 / 68.76% | 104.932 / 72.117 | 3.198 / 68.73% |
| 32,768 | Intrinsic | 275.440 / 242.685 | 8.409 / 88.11% | 276.055 / 243.239 | 8.412 / 88.11% |

Separate sampled process peak GiB,100ms lower bounds:

| B | Rigid | Pressure live | Intrinsic live | Policy rigid | Pressure policy | Intrinsic policy |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 2.180 | 2.295 | 2.359 | 2.188 | 2.303 | 2.336 |
| 2,048 | 3.904 | 4.613 | 6.959 | 3.918 | 4.627 | 7.064 |
| 16,384 | 16.066 | 21.369 | 40.496 | 16.131 | 21.434 | 40.529 |
| 32,768 | 29.973 | 40.525 | 78.809 | 30.098 | 40.648 | 78.898 |

Same-GPU B2048 live pressure screen:

| Setting | env-step/s | Change vs default | Full fallback passes/repeat |
| --- | --- | --- | --- |
| default | 204,320 | +0.00% | 868 |
| scatter128 | 209,582 | +2.58% | 0 |
| scatter256 | 209,650 | +2.61% | 0 |
| fast | 205,319 | +0.49% | 868 |
| tuned | 215,610 | +5.53% | 0 |
| history4 | 116,429 | -43.02% | 868 |

Separate same-allocation default/tuned confirmation:

| B | Default / tuned live | Live gain | Default / tuned policy | Policy gain |
| --- | --- | --- | --- | --- |
| 2,048 | 204,921 / 216,220 | +5.51% | 203,554 / 214,472 | +5.36% |
| 16,384 | 299,771 / 319,809 | +6.68% | 299,412 / 318,515 | +6.38% |

Matched max/full, n=3, native production/storage cost:

| B | Model | Max / full env-step/s | Extra ms/batch | Time change | Native MiB change |
| --- | --- | --- | --- | --- | --- |
| 32 | Pressure | 8,248 / 8,219 | +0.014 | +0.36% | +3.28 |
| 32 | Intrinsic | 2,745 / 2,770 | -0.104 | -0.89% | +3.28 |
| 16,384 | Pressure | 301,070 / 296,016 | +0.929 | +1.71% | +1680.00 |
| 16,384 | Intrinsic | 115,781 / 115,127 | +0.803 | +0.57% | +1680.00 |

Actual native routes for normal max/history0 (requested auto settings preserved in JSON):

| B | Model | Solve | Packed block | Rigid-post fused | Wrench reuse | Moment reuse |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | Pressure | inverse | [0] | True | [False] | [False] |
| 32 | Intrinsic | inverse | [0] | False | [False] | [False] |
| 2,048 | Pressure | inverse | [0] | True | [False] | [False] |
| 2,048 | Intrinsic | inverse | [0] | False | [False] | [False] |
| 8,192 | Pressure | inverse | [512] | False | [True] | [True] |
| 8,192 | Intrinsic | inverse | [512] | False | [False] | [False] |
| 16,384 | Pressure | inverse | [512] | False | [True] | [True] |
| 16,384 | Intrinsic | inverse | [512] | False | [False] | [False] |
| 32,768 | Pressure | inverse | [512] | False | [True] | [True] |
| 32,768 | Intrinsic | inverse | [512] | False | [False] | [False] |

