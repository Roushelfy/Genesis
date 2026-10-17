# Closed practical baseline scaling

This technical report figure covers only B32/256/2048/8192, four closed scopes,
and separate closed 100 ms process-memory observations. It does not establish
the final throughput plateau or capacity limit. See caption.md and source_data.csv.
Rates retain all three 2400-step repeats; memory is a sampled lower-bound peak
from one separate 2400-step pass after the same 900-step warmup. Native buffer
counts and process VRAM are different quantities.

The renderer preserves editable SVG text. visual_qa.json records inspection of
the accepted preview and the byte-identical PDF-run preview; the untouched
runtime qa.json marks visual review as pending because it precedes that review.
The PDF contains an embedded TrueType font dictionary and no Type3 dictionary.
All outputs are repository supplements, not a journal-submission claim.

source/summary.json.gz restores the exact metadata-only checkpoint input.
source/plot_practical_throughput.py and source/nature_mpl_style.py are the exact
executed renderer and archived skill dependency. The canonical renderer is at
research/rigid_stress/scripts/plot_practical_throughput.py. Runtime caches and
full traces remain in the project data root.

To reproduce, decompress the input with Python stdlib into a fresh data-side
run directory. Inside a Slurm allocation with the established container:

```bash
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 \
  /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python \
  "$task_packet/source/plot_practical_throughput.py" \
  --summary "$task_input" \
  --style-helper "$task_packet/source/nature_mpl_style.py" \
  --output "$task_output"
```

Set task_packet to this directory, task_input to the restored JSON, task_output
to a fresh data-side directory, and TMPDIR/MPLCONFIGDIR explicitly under data.
Review the editable SVG/PNG before using --export-pdf. Original orchestration,
source closures and terminal scheduler evidence are retained alongside the plot.
