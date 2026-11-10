# Noise as a Resource, Not a Liability — code and artifacts

Subject-independent (LOSO) evaluation of quantum reservoir computing (QRC) for
continuous autonomic stress monitoring on WESAD and AffectiveROAD.

Every number in the paper's tables and figures is read from the CSVs written by
the pipeline below; nothing is hard-coded.

---

## Requirements

Python 3.10+.
 
```bash
pip install -r requirements.txt
```

Key pins: PennyLane 0.45.1, NumPy 2.4.6, scikit-learn 1.9.0, SciPy 1.17.1,
NeuroKit2 0.2.13

## Data

Neither dataset is redistributed here. Obtain both, then build the feature CSVs:

```bash
python build_features.py --wesad-root "<WESAD root with S*/S*.pkl>" --out Dataset
python build_features.py --road-root  "<AffectiveROAD .../Database>" --wrist Left --out Dataset
python build_features.py --selftest          # sanity check, no data needed
```

Writes `Dataset/{wesad_hrv,wesad_eda,wesad_resp,road_hrv,road_eda}.csv`: 60 s
windows at 5 s hop, sliced within maximal single-condition runs so no window
crosses a label boundary.

WESAD folds on `subject` (15). AffectiveROAD folds on `driver_id`, **not**
`drive_id` — some drivers drove several drives, so folding on drive would leak
identity. Nine of the ten participants yield usable drives (13 of 14).

---

## Pipeline

Stage by stage. `--road-threshold 0.6` binarises the AffectiveROAD continuous
stress metric (prevalence 0.681); the threshold is ours, not the dataset's.

### Stage 1 — noise sweep and memory capacity (§5, Fig. 2)

```bash
python qrc_experiment.py   --dataset wesad --all-modalities --noise-sweep --jobs 3 --out Output/FINAL
python qrc_experiment.py   --dataset road  --all-modalities --noise-sweep --road-threshold 0.6 --jobs 3 --out Output/FINAL
python qrc_memory_regen.py --dataset wesad --all-modalities --out Output/FINAL
python qrc_memory_regen.py --dataset road  --all-modalities --out Output/FINAL
```

Run `qrc_memory_regen.py` after `qrc_experiment.py`: it supersedes the
memory-capacity curves, driving the reservoir with each modality's real
input-angle sequence instead of a generic i.i.d. probe. Probe settings are pinned
a priori (400 steps, max delay 15, 5 seeds).

### Stage 2 — main benchmark (Tables 1–2, §6)

```bash
python qrc_harness.py --dataset wesad --modality hrv  --methods memoryless esn qrc --seeds 42 3 123 37 73 --rich --jobs 3 --out Output/FINAL
python qrc_harness.py --dataset wesad --modality eda  --methods memoryless esn qrc --seeds 42 3 123 37 73 --rich --jobs 3 --out Output/FINAL
python qrc_harness.py --dataset wesad --modality resp --methods memoryless esn qrc --seeds 42 3 123 37 73 --rich --jobs 3 --out Output/FINAL
python qrc_harness.py --dataset road  --modality hrv  --methods memoryless esn qrc --seeds 42 3 123 37 73 --rich --jobs 3 --road-threshold 0.6 --out Output/FINAL
python qrc_harness.py --dataset road  --modality eda  --methods memoryless esn qrc --seeds 42 3 123 37 73 --rich --jobs 3 --road-threshold 0.6 --out Output/FINAL
```

### Stage 2b — quantum family (§7.1, Fig. 5)

```bash
python qrc_harness.py --dataset road  --modality eda --methods qrc vqc qlstm qgru --seeds 42 3 --nq 4 --rich --jobs 3 --road-threshold 0.6 --out Output/QUANTUM
python qrc_harness.py --dataset road  --modality hrv --methods qrc vqc qlstm qgru --seeds 42 3 --nq 4 --rich --jobs 3 --road-threshold 0.6 --out Output/QUANTUM
python qrc_harness.py --dataset wesad --modality eda --methods qrc vqc qlstm qgru --seeds 42 3 --nq 4 --rich --jobs 3 --out Output/QUANTUM
```

`--nq 4` sets the PCA angle-embedding width for VQC/QLSTM/QGRU only. The QRC
reservoir always uses one qubit per autonomic branch, Q ∈ {5,6}.

### Stage 3 — conformal p-streams and operating curve (Fig. 4, §6.4)

```bash
python qrc_harness.py --dataset wesad --modality hrv  --methods esn qrc --seeds 42 3 123 37 73 --rich --dump-streams --jobs 3 --out Output/STREAMS
python qrc_harness.py --dataset wesad --modality eda  --methods esn qrc --seeds 42 3 123 37 73 --rich --dump-streams --jobs 3 --out Output/STREAMS
python qrc_harness.py --dataset wesad --modality resp --methods esn qrc --seeds 42 3 123 37 73 --rich --dump-streams --jobs 3 --out Output/STREAMS
python qrc_harness.py --dataset road  --modality hrv  --methods esn qrc --seeds 42 3 123 37 73 --rich --dump-streams --jobs 3 --road-threshold 0.6 --out Output/STREAMS
python qrc_harness.py --dataset road  --modality eda  --methods esn qrc --seeds 42 3 123 37 73 --rich --dump-streams --jobs 3 --road-threshold 0.6 --out Output/STREAMS

python harness_operating_curve.py --root Output/STREAMS --m 3 --far-targets 0.05 0.10 0.15 --lat-targets 3 5 10 --out Output/STREAMS
```

### Stage 4 — NISQ robustness curve (Fig. 6, §7.2)

```bash
python qrc_harness.py --dataset wesad --modality eda --methods qrc --seeds 42 --nq 4 --rich --noise-curve --kappas 0.0 0.5 1.0 1.5 2.0 --jobs 3 --out Output/NOISE
```

### Stage 5 — consolidation (§6.2, §7.1)

```bash
python harness_consolidate.py --root Output/FINAL   --metric auroc      --out Output/FINAL
python harness_consolidate.py --root Output/FINAL   --metric auroc_ewma --out Output/FINAL
python harness_consolidate.py --root Output/QUANTUM --metric auroc      --out Output/QUANTUM
python harness_consolidate.py --root Output/QUANTUM --metric auroc_ewma --out Output/QUANTUM
```

### Stage 6 — figures

```bash
python reproduce_figures.py --root Output --out Output/figures   # Figs. 2-6, PDF + PNG
python make_pipeline_figure.py                                   # Fig. 1
```

`reproduce_figures.py` exits naming any missing input, so it doubles as a
completeness check on Stages 1-5. `make_pipeline_figure.py` creates
`Output/figures/` itself and needs no data.

`paper_figures.py` renders an alternative view of the same CSVs. Its output
filenames overlap `reproduce_figures.py`, so send it elsewhere:

```bash
python paper_figures.py --root Output/FINAL --out Output/figures_alt
```

`qrc_figures.py` renders 15 per-modality diagnostics — three figures per cell,
three WESAD modalities plus two ROAD — none of which appear in the paper:

```bash
python qrc_figures.py --dataset wesad --all-modalities --out Output/FINAL
python qrc_figures.py --dataset road  --all-modalities --out Output/FINAL
```

Each triple (`fig_noise_resource_<mod>`, `fig_memory_capacity_<mod>`,
`fig_perfold_box_<mod>`) is written beside the CSVs it read: into
`FINAL/<dataset>/<mod>/` when that directory exists — so, after Stage 2 — and
otherwise into WESAD's flat `FINAL/<mod>/`. Inputs are Stage 1's
`noise_sweep_*`, `memory_capacity_*` and `perfold_*` (the clean-vs-noisy E1
table, not Stage 2's `harness_perfold_*`); a missing input prints `[skip]` and
drops that one figure. Filenames are modality-suffixed and land per cell, so
there is no collision with the figures above.

---

## Output layout

```
Output/
  FINAL/<modality>/{noise_sweep,memory_capacity,perfold}_<modality>.csv      # WESAD
  FINAL/road/<modality>/{noise_sweep,memory_capacity,perfold}_<modality>.csv # ROAD
  FINAL/<dataset>/<modality>/harness_perfold_<modality>.csv            # memoryless/esn/qrc
  FINAL/<dataset>/<modality>/fig_{noise_resource,memory_capacity,perfold_box}_<modality>.png
  FINAL/CONSOLIDATED_*.csv
  QUANTUM/<dataset>/<modality>/harness_perfold_<modality>.csv          # qrc/vqc/qlstm/qgru
  QUANTUM/CONSOLIDATED_*.csv
  NOISE/wesad/eda/harness_noise_curve_eda.csv
  STREAMS/<dataset>/<modality>/harness_streams_<modality>.csv
  STREAMS/OPCURVE_{population,matched_tests}.csv
  figures/
```

`<dataset>` ∈ {wesad, road}, `<modality>` ∈ {hrv, eda, resp}. `_rescache/`
directories are recomputation caches and are gitignored.

## Files

| File | Role |
|---|---|
| `build_features.py` | raw signals → windowed feature CSVs |
| `qrc_base.py` | Config, `SuperNodeEncoder`, `load_dataset`, `alarm_metrics`, `tost_paired`, `bootstrap_ci`, `conformal_pvalues`, `calibration_metrics`, `rankings_metrics`, `_coupling_edges` |
| `qrc_core.py` | quantum reservoir (Eq. 1), `QRCConfig`, ridge readout |
| `qrc_experiment.py` | noise sweep, memory-capacity probe, `ClassicalESN` |
| `qrc_harness.py` | LOSO benchmark: all methods, conformal alarm, noise curve |
| `qrc_memory_regen.py` | data-driven memory-capacity curves |
| `harness_consolidate.py` | per-cell and benchmark tables, TOST, paired contrasts |
| `harness_operating_curve.py` | α-sweep latency/false-alarm curves, matched tests |
| `reproduce_figures.py` | Figs. 2–6 |
| `paper_figures.py` | alternative renderings of the same CSVs; not the paper's figures |
| `qrc_figures.py` | 15 per-modality diagnostics from the Stage 1 CSVs; not the paper's figures |
| `make_pipeline_figure.py` | Fig. 1 (schematic, no data) |

`qrc_base.py` and `qrc_core.py` are imported, never run directly.
`qrc_experiment.py` is both a script and a module.

## Pinned settings

Leak ε = 0.15 · washout 5 windows · V = 3 virtual sub-steps · ridge λ = 1.0 ·
coupling scale 0.6 · ring topology · reservoir seed 20260408 · γ = 1 ·
calibration fraction 0.35 · alarm α = 0.10 · sustain m = 3 · EWMA half-life 5.
None was tuned on a reported outcome.

Readout dimension is (3Q+|E|)V ∈ {60,72}. The classical ESN uses 64 units,
spectral radius 0.9, input scaling 0.5, tanh, the same leak, and a trained ridge
readout.

## Scope

The benchmark of §6–§7.1 runs the **noiseless** configuration (κ = 0). The
simulated NISQ channel enters only through the κ sweep of §5 and the robustness
curve of §7.2. Its four components — single-qubit depolarizing (p₁ = 0.01),
amplitude damping (γ₁ = 0.05), phase damping (γ₂ = 0.02), and a two-qubit
depolarizing proxy (p₂ = 0.10) per coupled pair — are all scaled by κ.

The Ising coupling web is drawn once from a fixed seed and held fixed across
runs, so seed variation covers data splits and the ESN draw only. Conformal
coverage is empirical, not guaranteed, since exchangeability fails within and
across subjects.