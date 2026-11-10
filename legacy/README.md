# Preliminary truncated-stream analysis (superseded)

These files come from the preliminary analysis that preceded the full-stream evaluation. They are kept only
because the paper cites them in two places:

- Sect. 4.5: on truncated streams, ROAD-HRV gave AUROC 0.67-0.75
  (`FINAL/CONSOLIDATED_master_auroc.csv`: memoryless 0.748, ESN 0.730, QRC 0.666).
- Sect. 5, Limitations: trained quantum models (VQC, QLSTM, QGRU) were examined only here (`QUANTUM/`).

No other reported result depends on these files.

## How this analysis differed from the final one
- each (subject, state) block truncated to its first 30 windows (`--cap 30`);
- baseline windows placed before stress windows in each stream;
- calibration on the calibration subjects' baseline windows scored on their own; fold seeding as in `qrc_harness.py`.

Shared with the final analysis: AffectiveROAD threshold 0.60, seeds 42 3 123 37 73, full readout, noise-free.
Run settings for each cell are in the `harness_manifest_*.json` files.

## Reproduce (from the repository root)
    python qrc_harness.py --dataset road --modality hrv --cap 30 --methods memoryless esn qrc --seeds 42 3 123 37 73 --rich --road-threshold 0.6 --out legacy_rerun
    python legacy/harness_consolidate.py --root legacy_rerun --metric auroc --out legacy_rerun

Repeat the first command for the other dataset and modality pairs. The memoryless per-fold AUROCs for
ROAD-HRV were reproduced to 1e-16 with the current code.