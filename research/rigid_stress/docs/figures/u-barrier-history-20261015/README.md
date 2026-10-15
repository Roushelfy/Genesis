# U barrier initialization/loading path

This supplement uses all744 original quasi-static stages and three barrier distances at one845-tet/pad5 setting. It does not establish physical convergence. The linked report distinguishes initialization and final-state extrema.

The SVG and PDF keep editable vector text. The main PNG uses1000dpi at183 by140mm; RGB and grayscale previews retain the same layout. Exact source data, the claim/panel contract, caption, body interpretation, accessibility description and accepted-source QA accompany the figure. Matplotlib DejaVu Sans is identified by its recorded font hash.

On a compute allocation, use the repository wrapper with this archived style source and a restored closed comparison run:

```bash
bash research/rigid_stress/scripts/plot_u_barrier_histories.sh FRESH_RUN \
  research/rigid_stress/docs/figures/u-barrier-history-20261015/figure_style_source.txt \
  --comparison CLOSED_COMPARISON
```

After source and rendered QA, add `--accepted-source-qa SOURCE_ACCEPTANCE_JSON --export-pdf` in another fresh directory. Accepted editable SVG bytes and numerical input hashes must match before PDF/1000dpi export. Original Slurm commands and actual font/rendering failures remain in the runtime journal and associated evidence packet.

The unmodified workflow style helper is preserved as a text provenance resource; the wrapper copies its bytes to `figure_style.py` inside the frozen runtime source. It is not a new project implementation.
