# Public source reconstruction

The measured isolated engine1ff39b077 can be reconstructed by source content
from published task commit6b851ece0526d29f90ec9beb7a33da35aa9e83bb and the
exact overlay here. Only examples/speed_benchmark/rigid_stress.py differs;
the native engine is already identical. All858 source files totaling225263082
bytes were verified against the original manifest after applying the overlay.
This does not assume that the private isolated Git object remains available.

Decode gzip copies into a working directory. Check out the public commit in
a fresh source checkout, then apply native-public-source-overlay-v1.patch.
Validate each manifest file by its relative path, SHA-256 and size. Keep current
research workers from their separately captured source or the published task
branch. If committing the reconstructed working tree, use that actual local
SCM identity for new executions; do not reuse the old label as a false HEAD.
Original historical source receipts stay unchanged. No runtime data or active
source checkout is moved.
