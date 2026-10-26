# Initial history diagnostic

This numerical pilot completed before the geometry description was corrected
to place the authored egg on the midsurface. It also constructs force from the
current facet normal. The final trajectory instead prescribes a smooth explicit
body-frame force, matching the historical report's stated complete-timing front
end. Neither choice was selected from the old target error or speedup.

These arrays and the run log remain as diagnostic evidence. The accompanying
history source snapshots preserve the trajectory and algorithm, but the final
core geometry differs. They are not the final reproducible configuration and
their numerical values must not be joined to final timings. The history factor
factory was refactored during the pilot to reuse the identical mother factor
instead of building a duplicate factor. Both used the same operator/scaling.
The final run binds its complete source files and repeats all controls.
