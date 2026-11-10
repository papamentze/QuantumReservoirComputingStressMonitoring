"""
================================================================================
 qrc_base.py -- shared foundation for the quantum reservoir computing study.

 Holds only what qrc_core / qrc_experiment / qrc_harness / qrc_memory_regen
 import: the run configuration, the dataset loader, the super-node angle
 encoder, the entangling-graph helpers, split-conformal calibration, and the
 metric and statistics routines.

 CONTENTS
   Config                  run configuration; call .finalize() after construction
   load_dataset            WESAD / AffectiveROAD -> (X, feature_names, states, subjects)
   SuperNodeEncoder        physiological features -> Q input angles via branch pooling
   _coupling_edges         ring / full entangling graphs for Q qubits
   conformal_pvalues       split-conformal p-values from baseline-only calibration
   alarm_metrics           debounced alarm rate and detection latency
   calibration_metrics     ECE, NLL, Brier
   ranking_metrics         AUROC, AUPRC, balanced accuracy, prevalence
   bootstrap_ci            percentile bootstrap confidence interval
   tost_paired             two one-sided tests for paired equivalence

 The label convention is BASE / AMUSE / STRESS. Detectors are fitted on BASE
 windows only; stress labels are used for evaluation, never for fitting or
 calibration.
================================================================================
"""

import warnings

warnings.filterwarnings("ignore")


import os


from dataclasses import dataclass


import numpy as np


import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             balanced_accuracy_score, brier_score_loss)


# cfg
@dataclass
class Config:
    # data
    data_dir: str   = "Dataset"
    dataset: str    = "wesad"          # qiriro file stem -> Dataset/<dataset>_hrv.csv
    road_threshold: float = 0.5        # AffectiveROAD 0..1 metric -> BASE/STRESS split
    modality: str   = "hrv"            # HRV is the autonomic substrate of the paper

    # encoding / circuit
    n_qubits: int   = 6                # Q = |B| <= 6, one autonomic branch per qubit
    layers: int     = 1                # L in {1,2}; small for noise tolerance
    entangle_topology: str = "ring"    # 'ring' (sparse, NISQ-friendly) | 'full' (all-to-all IQP)
    gamma: float    = 1.0              # angle bandwidth: theta = pi*tanh(gamma*g); >1 opens up
                                       # the saturated encoding (Eq.1), a label-free knob to sweep
    # LABEL-FREE kernel-bandwidth selection (per fold, baseline-only)
    # a-priori quantum-arm pinning (pre-registration; fixes seed-varying arm) ---

    # LOSO / subsampling
    fit_cap: int    = 350              # max baseline windows used to build K_fit
    cal_cap: int    = 250              # max baseline windows for conformal calibration
    noisy_fit_cap: int = 200           # smaller cap for the (heavier) NISQ condition
    cal_frac: float = 0.35             # fraction of the normal pool reserved (by subject) for calibration

    # Q-OCSVM

    # quantum isolation (distributional) kernel arm (novel)
    # robust (soft-assignment) IK variant "IKr" (item: noise-hardening)
    # Motivated a-priori by the shot-noise mechanism (hard argmax flips near ties on
    # a few subjects). SOFT membership = softmax(iso_soft_beta * similarity) over the
    # psi references, so a small fidelity perturbation nudges membership instead of
    # flipping the cell. Applied IDENTICALLY to the quantum arm and its Euclidean
    # twin. All three constants are PINNED a-priori -- never tuned on AUROC/TOST.
                                       # null (identical noise deltas, ~equal clean AUROC).
                                       # Set True ONLY to reproduce the soft-assignment ablation.
                                       # LABEL-FREE per arm to hit it (never tuned on AUROC)

    # Conformal / alarm rule
    alpha: float    = 0.10             # conformal level (alarm threshold on p-value)
    debounce_m: int = 3                # consecutive windows required to raise an alarm

    # NISQ noise model (matched to prior benchmark)
    depol_1q: float = 0.01
    depol_2q: float = 0.10
    amp_damp: float = 0.05             # gamma_1 (T1 proxy)
    phase_damp: float = 0.02           # gamma_2 (T2 proxy)
    readout_err: float = 0.20          # symmetric readout assignment error

    # evaluation / stats
    n_boot: int     = 2000             # bootstrap resamples for 95% CIs
    ci_level: float = 0.95
    tost_margins: tuple = (0.01, 0.02, 0.03, 0.05) # equivalence margins (1-5 percentage points)

    # run control
    seed: int       = 42
    noisy: bool     = False            # also run the NISQ condition (secondary)
    quick: bool     = False            # tiny caps for a fast smoke test
    synthetic: bool = False            # generate synthetic data if the CSV is absent
    tag: str        = None

    branch_order: tuple = ("vagal", "sympathovagal", "interval",
                           "nonlinear", "relativeRR", "other")

    def finalize(self):
        if str(self.modality).lower() == "eda" and tuple(self.branch_order) == HRV_ORDER:
            self.branch_order = EDA_ORDER       # switch super-node layout to electrodermal
        elif str(self.modality).lower() == "resp" and tuple(self.branch_order) == HRV_ORDER:
            self.branch_order = RESP_ORDER      # switch super-node layout to respiratory
        if self.quick:
            self.fit_cap, self.cal_cap, self.noisy_fit_cap = 90, 70, 60
            self.n_boot = 300
        if self.tag is None:
            self.tag = f"qrc_{self.dataset}_{self.modality}_L{self.layers}" \
                       + ("_noisy" if self.noisy else "")
        return self


# Condition string -> {BASE, AMUSE, STRESS}, mapping from the raw label so it is
# independent of any precomputed numeric encoding. Covers BOTH benchmarks:
#   WESAD : baseline / amusement / stress (TSST)            -> BASE / AMUSE / STRESS
#   SWELL : no-stress(N) / time-pressure(T) / interruption(I)-> BASE / STRESS / STRESS
# SWELL has no amusement block, so AMUSE is simply absent there (the pipeline
# treats AMUSE as an optional withheld specificity probe).  Strings are matched
# case-insensitively after stripping spaces/underscores/hyphens.
COND_MAP = {
    # WESAD
    "baseline": "BASE", "rest": "BASE", "neutral": "BASE", "0": "BASE",
    "amusement": "AMUSE", "amused": "AMUSE", "fun": "AMUSE", "1": "AMUSE",
    "stress": "STRESS", "stressed": "STRESS", "tsst": "STRESS", "2": "STRESS",
    # SWELL-KW
    "nostress": "BASE", "no stress": "BASE", "n": "BASE", "rest hc": "BASE",
    "timepressure": "STRESS", "time pressure": "STRESS", "t": "STRESS",
    "interruption": "STRESS", "interruptions": "STRESS", "i": "STRESS",
}


LABEL_CANDS = ["condition", "Condition", "label", "Label", "state", "y",
               "Stress", "stress_condition", "blok", "Blok"]


SUBJ_CANDS  = ["subject id", "subject_id", "subjectId", "subject", "Subject",
               "participant", "Participant", "PP", "pp", "pid", "ID", "id"]


# HRV base features (qiriro "stress" set). Any missing column is silently skipped.
BASE_HRV = ["MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR",
            "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW", "MEAN_REL_RR",
            "MEDIAN_REL_RR", "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR",
            "SDRR_RMSSD_REL_RR", "KURT_REL_RR", "SKEW_REL_RR", "VLF", "VLF_PCT",
            "LF", "LF_PCT", "LF_NU", "HF", "HF_PCT", "HF_NU", "TP", "LF_HF",
            "HF_LF", "sampen", "higuci"]


# Column-name recognition for META columns to EXCLUDE from the feature set.
# Matched precisely (exact name or structural prefix) rather than by loose
# substring, so legitimate features like SCR_RiseTime / RecoveryTime survive.
META_EXACT  = {"subject", "participant", "pid", "id", "idx", "pp", "label",
               "condition", "state", "class", "target", "blok", "y", "n"}


META_PREFIX = ("unnamed", "time", "timestamp", "datetime", "date", "epoch",
               "window", "segment")


# Any column containing one of these as a WHOLE token (split on space/_/-) is a
# label/target/annotation column, never a physiological feature -- e.g.
# 'condition label', 'Condition Label', 'SSSQ Label', 'NasaTLX class'. This
# closes a target-leakage path on EDA files whose label columns are numeric.
LABEL_TOKENS = {"label", "class", "condition", "state", "target", "stress",
                "sssq", "nasatlx", "blok", "annotation", "score"}


# ==============================================================================
#  DATA LOADING  (schema auto-detection; NOT a copy of any prior preprocessing)
# ==============================================================================
def _detect(cols, cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_dataset(cfg: Config):
    """Return (X, feat_cols, y_state, subjects) for the wearable benchmark.

    y_state in {BASE, AMUSE, STRESS}; subjects is the subject identifier used by
    the LOSO grouping. Imputation/scaling are deliberately NOT applied here --
    they are fit per fold on N_fit only, so the LOSO protocol stays leakage-free.
    """
    path = os.path.join(cfg.data_dir, f"{cfg.dataset}_{cfg.modality}.csv")
    if not os.path.exists(path):
        if cfg.synthetic:
            print(f"   [synthetic] {path} not found -> generating synthetic cohort.")
            return _synthetic_cohort(cfg)
        raise FileNotFoundError(
            f"{path} not found. Place the qiriro CSV there, or pass --synthetic "
            f"to run the machinery on a synthetic cohort.")
    df = pd.read_csv(path)

    # ---- ROAD (AffectiveROAD): continuous target -> binarized BASE/STRESS ----
    # ROAD ships a continuous 0..1 subjective stress metric (stress_target) and a
    # driver grouping (driver_id), not categorical condition blocks. We binarize
    # at cfg.road_threshold to slot ROAD into the same BASE/STRESS pipeline as
    # WESAD, and group LOSO by DRIVER. Meta columns are excluded explicitly.
    if str(cfg.dataset).lower() == "road":
        ROAD_META = {"driver_id", "drive_id", "t_sec", "motion", "n_beats", "stress_target"}
        if "stress_target" not in df.columns or "driver_id" not in df.columns:
            raise ValueError(f"{path}: ROAD file needs 'stress_target' and 'driver_id' columns.")
        thr = float(cfg.road_threshold)
        state = np.where(df["stress_target"].to_numpy(float) >= thr, "STRESS", "BASE")
        keep = np.isfinite(df["stress_target"].to_numpy(float))
        mod = str(cfg.modality).lower()
        if mod == "hrv":
            feats = [c for c in BASE_HRV if c in df.columns]
            if len(feats) < cfg.n_qubits:
                feats = [c for c in df.columns if c not in ROAD_META
                         and pd.api.types.is_numeric_dtype(df[c])]
        else:
            feats = [c for c in df.columns if c not in ROAD_META
                     and pd.api.types.is_numeric_dtype(df[c])]
        if not feats:
            raise ValueError(f"{path}: no usable {cfg.modality} features after meta removal.")
        X = (df.loc[keep, feats].apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan).to_numpy(np.float64))
        y_state = state[keep]
        subjects = np.array([f"S{v}" for v in df.loc[keep, "driver_id"].values])
        nbr = len({branch_of(c, mod) for c in feats})
        prev = float((y_state == "STRESS").mean())
        print(f"   road/{cfg.modality}: {len(X)} windows | {len(feats)} feats -> {nbr} branches "
              f"| {len(np.unique(subjects))} drivers | thr={thr} prevalence(STRESS)={prev:.2f} "
              f"| states { {s:int((y_state==s).sum()) for s in ['BASE','STRESS']} }")
        return X, feats, y_state, subjects

    lab = _detect(df.columns, LABEL_CANDS)
    subj = _detect(df.columns, SUBJ_CANDS)
    if lab is None or subj is None:
        raise ValueError(f"{path}: need a condition column ({LABEL_CANDS}) and a "
                         f"subject column ({SUBJ_CANDS}); found {lab}, {subj}.")

    def _norm(v):  # normalise a raw label for COND_MAP lookup
        return str(v).strip().lower().replace("_", " ").replace("-", " ")
    state = np.array([COND_MAP.get(_norm(v), COND_MAP.get(_norm(v).replace(" ", ""), None))
                      for v in df[lab].values])
    keep = np.array([s is not None for s in state])
    if not keep.any():
        raise ValueError(f"{path}: no rows matched COND_MAP from column '{lab}'. "
                         f"Unique values seen: {sorted(map(str, pd.unique(df[lab])))[:12]}")

    # --- feature columns ---------------------------------------------------
    mod = str(cfg.modality).lower()
    meta = {lab, subj}
    def _is_meta(c):
        cl = c.strip().lower().replace("-", " ").replace("_", " ")
        toks = set(cl.split())
        return (c in meta or cl in META_EXACT or cl.replace(" ", "") in META_EXACT
                or cl.startswith(META_PREFIX) or bool(toks & LABEL_TOKENS))
    if mod == "hrv":
        feats = [c for c in BASE_HRV if c in df.columns]
        if len(feats) < cfg.n_qubits:                       # fall back to numeric-non-meta
            feats = [c for c in df.columns
                     if not _is_meta(c) and pd.api.types.is_numeric_dtype(df[c])]
    else:                                                   # EDA (and any non-hrv): take all numeric, non-meta
        feats = [c for c in df.columns
                 if not _is_meta(c) and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise ValueError(f"{path}: no usable {cfg.modality} feature columns after "
                         f"removing meta columns. Columns present: {list(df.columns)[:20]}")

    X = (df.loc[keep, feats]
           .apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan)
           .to_numpy(dtype=np.float64))
    y_state = state[keep]
    subjects = np.array([f"S{v}" for v in df.loc[keep, subj].values])
    nbr = len({branch_of(c, mod) for c in feats})
    print(f"   {cfg.dataset}/{cfg.modality}: {len(X)} windows | {len(feats)} {cfg.modality} feats "
          f"-> {nbr} branches | {len(np.unique(subjects))} subjects | "
          f"states { {s:int((y_state==s).sum()) for s in ['BASE','AMUSE','STRESS']} }")
    return X, feats, y_state, subjects


def _synthetic_cohort(cfg: Config, n_subj=12, per_state=60):
    """A schema-faithful synthetic WESAD-like cohort for smoke testing only.
    Each subject has an idiosyncratic autonomic baseline (the source of the
    cross-subject gap); STRESS shifts sympathovagal/interval features, AMUSE
    a milder, partly overlapping shift -> a realistic specificity probe."""
    rng = np.random.default_rng(cfg.seed)
    feats = BASE_HRV
    D = len(feats)
    rows, states, subjects = [], [], []
    # feature-group masks so the shift is physiologically structured
    def grp(name): return np.array([_hrv_branch(c) == name for c in feats], float)
    symp = grp("sympathovagal") + grp("interval")
    vagal = grp("vagal")
    for s in range(n_subj):
        base_mu = rng.normal(0, 1, D) + rng.normal(0, 0.8, D)   # subject-specific baseline
        scale = rng.uniform(0.7, 1.4, D)
        for st, shift in [("BASE", 0.0), ("AMUSE", 0.6), ("STRESS", 1.5)]:
            mu = base_mu + shift * (symp - 0.5 * vagal)
            X = rng.normal(mu, scale, (per_state, D))
            rows.append(X); states += [st] * per_state
            subjects += [f"S{s:02d}"] * per_state
    X = np.vstack(rows)
    X[rng.random(X.shape) < 0.01] = np.nan      # sprinkle missingness
    print(f"   [synthetic] {len(X)} windows | {D} feats | {n_subj} subjects.")
    return X, feats, np.array(states), np.array(subjects)


# ==============================================================================
#  AUTONOMIC SUPER-NODE ENCODING  (reused branch grouping; one branch per qubit)
# ==============================================================================
def _hrv_branch(col):
    u = col.upper()
    if "REL_RR" in u:                                            return "relativeRR"
    if any(k in u for k in ["SAMPEN", "HIGUC", "KURT", "SKEW", "SD1"]):  return "nonlinear"
    if any(k in u for k in ["RMSSD", "SDSD", "PNN", "HF"]):      return "vagal"
    if any(k in u for k in ["LF", "VLF", "LF_HF", "HF_LF"]):     return "sympathovagal"
    if any(k in u for k in ["RR", "SD2", "HR", "TP", "SDRR"]):   return "interval"
    return "other"


def _eda_branch(col):
    """Group an EDA feature into an electrodermal super-node (one per qubit).

    The tonic/phasic decomposition is the canonical physiological split of the
    EDA signal, so it plays the same role for EDA that the autonomic bands play
    for HRV: tonic (slow SCL drive), phasic (SCR magnitude), responserate (SCR
    frequency), shape (SCR rise/recovery dynamics), variability (dispersion),
    nonlinear (higher-order shape).  Checks run most-specific first.
    """
    u = col.upper()
    if any(k in u for k in ["KURT", "SKEW", "SAMPEN", "ENTROP", "HIGUC", "HJORTH"]): return "nonlinear"
    if any(k in u for k in ["RISE", "RECOV", "HALF", "DECAY", "WIDTH", "ONSET"]):    return "shape"
    if any(k in u for k in ["NSCR", "N_SCR", "SCR_N", "NUMSCR", "PEAK", "RATE",
                            "FREQ", "COUNT", "NUM"]):                                return "responserate"
    if any(k in u for k in ["SCR", "PHASIC", "AMP"]):                               return "phasic"
    if any(k in u for k in ["SCL", "TONIC", "LEVEL"]):                              return "tonic"
    if any(k in u for k in ["STD", "_SD", "SD_", "RANGE", "MIN", "MAX", "SLOPE",
                            "IQR", "VAR"]):                                         return "variability"
    if any(k in u for k in ["MEAN", "MEDIAN", "EDA", "GSR"]):                       return "tonic"
    return "other"


def _resp_branch(col):
    """Group a respiration feature into a super-node (one per qubit).

    WESAD-only modality (ROAD has no respiration waveform). Mirrors the HRV/EDA
    routers; checks most-specific first. Branches: rate / interval / variability
    / amplitude / shape / nonlinear.
    """
    u = col.upper()
    if any(k in u for k in ["KURT", "SKEW", "SAMPEN", "ENTROP", "HIGUC"]):              return "nonlinear"
    if any(k in u for k in ["IE_RATIO", "INHALE", "EXHALE", "INSP", "EXP", "RISE",
                            "DUTY"]):                                                   return "shape"
    if any(k in u for k in ["AMP", "DEPTH", "EXCURSION"]):                              return "amplitude"
    if any(k in u for k in ["STD", "_SD", "SD_", "VAR", "RMSSD", "RANGE", "IQR",
                            "RRV", "_CV"]):                                             return "variability"
    if any(k in u for k in ["BBI", "INTERVAL", "PERIOD", "_BB"]):                       return "interval"
    if any(k in u for k in ["RATE", "FREQ", "BREATH", "DOM_"]):                         return "rate"
    return "other"


def branch_of(col, modality):
    m = str(modality).lower()
    if m == "eda":
        return _eda_branch(col)
    if m == "resp":
        return _resp_branch(col)
    return _hrv_branch(col)


HRV_ORDER = ("vagal", "sympathovagal", "interval", "nonlinear", "relativeRR", "other")


EDA_ORDER = ("tonic", "phasic", "responserate", "shape", "variability", "nonlinear", "other")


RESP_ORDER = ("rate", "interval", "variability", "amplitude", "shape", "nonlinear", "other")


class SuperNodeEncoder:
    """Fit-on-N_fit-only encoder: median impute -> standardize -> branch-mean
    super-nodes -> standardize branch signals -> theta = pi * tanh(g).

    Every statistic (medians, feature mean/std, branch mean/std) is estimated on
    the fit set and applied unchanged to calibration/test, so no test-subject
    information can leak (paper Sec. III-B)."""

    def __init__(self, feat_cols, cfg: Config):
        self.feat_cols = list(feat_cols)
        self.cfg = cfg
        # branch layout truncated to Q qubits, preferred autonomic order
        assign = {c: branch_of(c, cfg.modality) for c in self.feat_cols}
        present = [b for b in cfg.branch_order if b in set(assign.values())]
        self.branches = present[:cfg.n_qubits]
        keep = set(self.branches)
        for c, b in assign.items():               # truncated branches fold into the last kept one
            if b not in keep:
                assign[c] = self.branches[-1]
        self.assign = assign
        self.branch_idx = {b: [i for i, c in enumerate(self.feat_cols) if assign[c] == b]
                           for b in self.branches}

    def fit(self, X):
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        Xi = self._impute(X)
        self.f_mu_, self.f_sd_ = Xi.mean(0), Xi.std(0)
        self.f_sd_[self.f_sd_ == 0] = 1.0
        G = self._to_branches((Xi - self.f_mu_) / self.f_sd_)
        self.g_mu_, self.g_sd_ = G.mean(0), G.std(0)
        self.g_sd_[self.g_sd_ == 0] = 1.0
        return self

    def _impute(self, X):
        Xi = X.copy()
        bad = ~np.isfinite(Xi)
        if bad.any():
            Xi[bad] = np.take(self.median_, np.where(bad)[1])
        return Xi

    def _to_branches(self, Xstd):
        G = np.zeros((Xstd.shape[0], len(self.branches)), np.float64)
        for j, b in enumerate(self.branches):
            idx = self.branch_idx[b]
            if idx:
                G[:, j] = Xstd[:, idx].mean(1)
        return G

    def transform_angles(self, X, gamma=None):
        Xi = self._impute(X)
        G = self._to_branches((Xi - self.f_mu_) / self.f_sd_)
        G = (G - self.g_mu_) / self.g_sd_
        g = self.cfg.gamma if gamma is None else float(gamma)
        return np.pi * np.tanh(g * G)   # Eq. (1) with bandwidth gamma; bounded to (-pi, pi)


def _ring_edges(Q):
    """Inter-branch coupling graph: a ring over the Q autonomic super-nodes,
    so each branch is coupled (e.g. sympathovagal-interval) without a dense
    O(Q^2) two-qubit load."""
    if Q < 2:
        return []
    return [(i, (i + 1) % Q) for i in range(Q)] if Q > 2 else [(0, 1)]


def _full_edges(Q):
    """All-to-all coupling over the Q super-nodes (denser IQP interactions).
    Used by the entanglement-topology ablation; O(Q^2) two-qubit terms."""
    return [(i, j) for i in range(Q) for j in range(i + 1, Q)]


def _coupling_edges(Q, topology):
    return _full_edges(Q) if topology == "full" else _ring_edges(Q)


# ==============================================================================
#  DISTRIBUTION-FREE CONFORMAL CALIBRATION + ALARM RULE
# ==============================================================================
def conformal_pvalues(a_cal, a_test):
    """Split-conformal p-value (Eq. 5) calibrated on baseline scores a_cal:
       p(x) = (1 + #{ a_cal >= a(x) }) / (c + 1).
    Returns p (valid, super-uniform) and the calibrated anomaly prob pi = 1 - p."""
    c = len(a_cal)
    counts = (a_cal[None, :] >= a_test[:, None]).sum(axis=1)
    p = (1.0 + counts) / (c + 1.0)
    return p, 1.0 - p


def alarm_metrics(p_seq, is_stress_seq, alpha, m):
    """Operate the alarm rule on a time-ordered monitoring session.
       Alarm at i iff p<=alpha for m consecutive windows ending at i.
       Returns (detection_latency_in_windows or None, false_alarm_rate)."""
    below = (p_seq <= alpha).astype(int)
    sustained = np.zeros_like(below)
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        sustained[i] = 1 if run >= m else 0
    onset = np.argmax(is_stress_seq) if is_stress_seq.any() else None
    # false alarms: sustained alarms among true-baseline windows (before onset)
    base_mask = ~is_stress_seq
    far = float(sustained[base_mask].mean()) if base_mask.any() else np.nan
    latency = None
    if onset is not None:
        post = np.where(sustained[onset:] == 1)[0]
        latency = int(post[0]) if len(post) else None    # windows after onset
    return latency, far


# ==============================================================================
#  EVALUATION METRICS
# ==============================================================================
def calibration_metrics(pi, y, n_bins=15):
    """ECE (equal-mass bins), NLL, Brier on the calibrated anomaly prob pi vs the
    binary anomaly label y in {0,1}. Calibration is assessed at evaluation only."""
    pi = np.clip(pi, 1e-7, 1 - 1e-7)
    nll = float(-(y * np.log(pi) + (1 - y) * np.log(1 - pi)).mean())
    brier = float(brier_score_loss(y, pi)) if len(np.unique(y)) > 1 else float(((pi - y) ** 2).mean())
    # adaptive (equal-mass) ECE
    order = np.argsort(pi)
    ece, n = 0.0, len(pi)
    for b in np.array_split(order, min(n_bins, max(1, n))):
        if len(b) == 0:
            continue
        conf, acc = pi[b].mean(), y[b].mean()
        ece += (len(b) / n) * abs(conf - acc)
    return float(ece), nll, brier


def ranking_metrics(pi, y):
    """AUPRC (primary; stress is the minority), AUROC, balanced accuracy, plus the
    positive PREVALENCE and a prevalence-normalised AUPRC.

    AUPRC's chance level is the positive prevalence pi_+ = mean(y), NOT 0.5, so an
    AUPRC must always be read against pi_+. We therefore also report:
      * prevalence : pi_+ (the AUPRC chance floor for this fold);
      * auprc_norm : (AUPRC - pi_+) / (1 - pi_+), clipped to [0,1]-ish, i.e. the
                     fraction of the achievable head-room above chance that is
                     captured (0 = chance, 1 = perfect). This makes AUPRC
                     comparable across the WESAD (~0.35 prevalence) and SWELL
                     (~0.61 prevalence) cells, which a raw AUPRC does not."""
    if len(np.unique(y)) < 2:
        return dict(auprc=np.nan, auroc=np.nan, bacc=np.nan,
                    prevalence=float(np.mean(y)) if len(y) else np.nan, auprc_norm=np.nan)
    prev = float(np.mean(y))
    ap = float(average_precision_score(y, pi))
    apn = (ap - prev) / (1.0 - prev) if prev < 1.0 else np.nan
    return dict(auprc=ap,
                auroc=float(roc_auc_score(y, pi)),
                bacc=float(balanced_accuracy_score(y, (pi >= 0.5).astype(int))),
                prevalence=prev,
                auprc_norm=float(apn))


# ==============================================================================
#  STATISTICS  (paired Wilcoxon + Holm, bootstrap CIs, TOST equivalence)
# ==============================================================================
def bootstrap_ci(values, n_boot, level, rng):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))


def tost_paired(perfold, metric, contrasts, margins):
    """Two one-sided tests (TOST) for equivalence on per-subject paired diffs.

    Each held-out subject contributes ONE paired difference d_i = metric_a -
    metric_b. Under strict LOSO the folds are over DISJOINT subjects -- they are
    independent sampling units -- and each per-subject metric is already averaged
    over thousands of (overlapping) windows, so its within-subject sampling error
    is negligible. The correct unit of analysis is therefore the per-subject diff
    with an ORDINARY paired standard error. We deliberately do NOT apply a
    Nadeau-Bengio resampled-fold correction: it is meant for repeated random
    splits that share training data, not subject-disjoint LOSO, and feeding it raw
    window counts (rho = n_test_windows / n_fit ~ 20) inflates the SE ~20x and
    destroys all power -- which is exactly why the old equivalence table reported
    'not equivalent' everywhere with p ~ 0.5.

    Reports, per contrast and margin (in percentage points of the metric): the
    mean paired difference, its SE, the 90% CI (the interval a TOST at alpha=.05
    inspects), the TOST p-value, and the equivalence verdict. NOTE: with 15-25
    subjects the CIs are wide, so equivalence will typically NOT hold at tight
    (1-2 pp) margins -- report 'no significant difference' with the CI and the
    margin at which equivalence would hold, rather than overclaiming.
    """
    rows = []
    for a, b in contrasts:
        pa = (perfold[perfold.arm == a][["subject", metric]].dropna()
                .groupby("subject", as_index=False)[metric].mean())
        pb = (perfold[perfold.arm == b][["subject", metric]].dropna()
                .groupby("subject", as_index=False)[metric].mean())
        m = pd.merge(pa, pb, on="subject", suffixes=("_a", "_b"))
        if len(m) < 5:
            continue
        d = m[f"{metric}_a"].values - m[f"{metric}_b"].values
        nfold = len(d); mean = float(d.mean()); dfree = nfold - 1
        se = float(d.std(ddof=1) / np.sqrt(nfold)) if nfold > 1 else np.nan
        half = float(stats.t.ppf(0.95, dfree) * se) if np.isfinite(se) else np.nan
        for delta in margins:
            if not np.isfinite(se) or se == 0:
                p_tost, equiv = np.nan, False
            else:
                p_lo = stats.t.sf((mean + delta) / se, dfree)   # H0: diff <= -delta
                p_hi = stats.t.cdf((mean - delta) / se, dfree)  # H0: diff >=  delta
                p_tost = max(p_lo, p_hi); equiv = bool(p_tost < 0.05)
            rows.append(dict(contrast=f"{a} vs {b}", margin_pp=delta * 100,
                             n=nfold, mean_diff=mean, se=se,
                             ci90_lo=mean - half, ci90_hi=mean + half,
                             p_tost=float(p_tost) if np.isfinite(p_tost) else np.nan,
                             equivalent=equiv))
    return pd.DataFrame(rows)
