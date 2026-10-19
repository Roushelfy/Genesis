# native-production-measurements-v1

Terminal15-case B1/B8/B32 rigid/DPR/SCR max/full scene timings, complete motion/output audits and separate stage profiles. Zero invalid-step counts and timing receipts pass, but strict motion conformance fails; these are measured baseline rates, not a fully admitted production recommendation. Retain the original SCR profile-normalization failure and corrected observer-only profile. Completed motion diagnostics reproduce the discrepancy without any stress updates; the completed controller-only control verifies bitwise-identical joint/phase/grip/limit inputs. Physical collider-load errors remain pending.

Large JSON and raw logs use deterministic gzip. Decompression preserves every original byte; the manifest records both encoded and original SHA-256/size. Use `python -m gzip -d FILE.gz` on a separate working copy to inspect those records. Full trajectory/source-history NPZ files remain on data with exact identities.
