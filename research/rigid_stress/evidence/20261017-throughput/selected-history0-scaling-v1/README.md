# Selected engine with qualified history0

This packet retains editable SVG, embedded-TrueType PDF,450dpi PNG and a150dpi
preview, exact raw-repeat/source CSV, executed renderer/style and compressed
metadata inputs. visual_qa.json records inspection and the byte-identical
PDF-run preview. The untouched runtime qa.json precedes that manual review.
These are repository technical-report supplements, without a journal claim.

Rates use900 warmup steps and three2400-step repeats. Memory is a separate
100ms NVML sampled lower-bound process peak, including setup/growth and one
2400-step pass. Raw ranges are not confidence intervals. Source manifests,
conditions, model/physics and original numerical bounds remain matched;
history0 is explicitly separate configuration tuning. See caption.md,
interpretation.md, figure_contract.json and source_data.csv.

source/history*.json.gz restores the exact plotting input bytes. The independent
metadata audits refer to the original raw benchmark records in the project
data root; those large originals are retained there rather than duplicated.
Source/command reproduction and accuracy limitations are in the
[final report](../../../docs/PRACTICAL_THROUGHPUT_RESULTS_20261017.md),
[running guide](../../../docs/RUNNING_PRACTICAL_THROUGHPUT.md) and
[exact engine/input recipes](../source-recipes-v1/README.md).

Decompress the inputs with Python stdlib into a fresh data-side task_inputs
directory. Set task_packet to this packet and task_output to a fresh data-side
output. Inside a CPU Slurm allocation with TMPDIR/MPLCONFIGDIR under data:

```bash
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 \
  /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python \
  "$task_packet/source/plot.py" \
  --summary "$task_inputs/history0.json" \
  --rate-family history0-four-scopes-b \
  --memory-family history0-memory-four-scopes-b \
  --series-label "selected engine, history0" \
  --style-helper "$task_packet/source/style.py" --output "$task_output"
```

The shared launcher expects optional environment variables to be unset safely;
omit nounset in its outer shell. Scientific/render wrappers keep their own
strict checks. Review the SVG/PNG before adding --export-pdf.
