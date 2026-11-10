"""
================================================================================
 qrc_harness.py -- EVIDENCE ENGINE
 NOTE: the paper's pipeline uses load_stream, the logistic/ESN/QRC models and _ewma from this module.
 Its command-line interface (default --cap 30) is the preliminary truncated-stream analysis (see legacy/).
 Honest, subject-independent (LOSO) multi-method comparison for CONTINUOUS
 autonomic stress monitoring, on the real WESAD/ROAD per-window feature streams.

 WHAT THIS PRODUCES (the paper's spine, regardless of which model "wins")
   For each method in {memoryless, esn(+skip), qrc(+skip), qsvm, vqc}:
     * per-window ranking   : AUROC / AUPRC / bACC (on the conformal anomaly prob pi)
     * continuous monitoring: EWMA-smoothed streamed AUROC, sustained-alarm
                              detection LATENCY and false-alarm rate (FAR)
     * calibration          : ECE / NLL / Brier
   Then, across methods:
     * EQUIVALENCE  : paired TOST(QRC vs ESN) on per-subject diffs  (tost_paired, reused)
     * BEST-QUANTUM : paired mean diff + Wilcoxon  QRC vs QSVM, QRC vs VQC
     * NISQ ROBUST  : noise-scale (kappa) curve of subject-mean AUROC, QRC vs VQC

 HONESTY CONTRACT (do not weaken)
   * ONE feature representation for all methods: the super-node angles,
     with the encoder fit on FIT SUBJECTS ONLY per fold. The reservoir (quantum or
     classical) is the ONLY thing that differs between arms. This is what makes the
     comparison fair AND what removes the NaN crash (SuperNodeEncoder._impute uses
     TRAIN-only medians -> no NaN reaches any sklearn/quantum call, no test leak).
   * Nothing is selected on a test metric. Reservoir hyperparameters are pinned in
     QRCConfig a-priori. TOST is reported as-is; if QRC is significantly worse than
     ESN, that is the result.
   * "Continuous" is legitimate because rows are time-ordered within (subject,state);
     each fold evaluates a rest->stress monitoring stream for the held-out subject.

 REUSE (verbatim, qrc_base)
   SuperNodeEncoder (train-only angle encoder), conformal_pvalues, alarm_metrics,
   calibration_metrics, ranking_metrics, bootstrap_ci, tost_paired.
   QuantumReservoir / RidgeReadout (qrc_core), ClassicalESN (qrc_experiment).

 USAGE (see the printed RUNBOOK at the bottom, or --help)
   python qrc_harness.py --dataset wesad --modality hrv --cap 30 \
          --methods memoryless esn qrc --seeds 42 --out Output/harness
   python qrc_harness.py --dataset wesad --modality hrv --cap 25 \
          --methods memoryless esn qrc qsvm vqc --seeds 42 7 --nq 4 \
          --noise-curve --out Output/harness
================================================================================
"""

from __future__ import annotations
import os, sys, time, json, argparse, dataclasses
# keep BLAS single-threaded so per-fold parallelism (if added) does not oversubscribe
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qrc_base import (Config, load_dataset, SuperNodeEncoder,
                      conformal_pvalues, alarm_metrics, calibration_metrics,
                      ranking_metrics, bootstrap_ci, tost_paired, _coupling_edges)
from qrc_core import QRCConfig, QuantumReservoir, RidgeReadout
from qrc_experiment import ClassicalESN

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy import stats

import pennylane as qml
from pennylane import numpy as pnp

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False    

# layout conventions (match the rest of the pipeline)
MODALITIES_BY_DATASET = {"wesad": ["hrv", "eda", "resp"], "road": ["hrv", "eda"]}
LEGACY_FLAT_DATASET = os.environ.get("QRC_LEGACY_FLAT_DATASET", "wesad")
ALL_METHODS = ["memoryless", "esn", "qrc", "qsvm", "vqc", "qlstm", "qgru"]

def _out_dir(out_root, dataset, mod):
    return os.path.join(out_root, dataset, mod)

def _scale_noise(cfg: Config, kappa: float) -> Config:
    """
    Return a copy of cfg with the NISQ channel scaled by kappa (kappa=1 -> the
    exact channel from previous work, kappa=0 -> clean). Mirrors qrc_memory_regen._mc_at.
    """
    return dataclasses.replace(
        cfg,
        depol_1q=cfg.depol_1q * kappa,
        depol_2q=cfg.depol_2q * kappa,
        amp_damp=min(0.999, cfg.amp_damp * kappa),
        phase_damp=min(0.999, cfg.phase_damp * kappa),
        readout_err=min(0.5, cfg.readout_err * kappa),
    )

# DATA: build a balanced, time-ordered monitoring stream (BASE/STRESS only)
def load_stream(dataset, modality, cap, seed=42, synthetic=False, road_threshold=0.5):
    """
    Load real features and cap each (subject, state) block to `cap` consecutive
    (time-ordered) windows -> a balanced rest/stress monitoring stream per subject.
    cap<=0 keeps everything. AMUSE is dropped (binary stress task).

    Returns (cfg, X, feats, y_state, subjects, subj_list). NaN is left in X on
    purpose; the per-fold SuperNodeEncoder imputes it with TRAIN medians.
    """
    cfg = Config(dataset=dataset, modality=modality, n_qubits=6, seed=seed,
                 synthetic=synthetic, road_threshold=road_threshold).finalize()
    X, feats, y_state, subjects = load_dataset(cfg)
    keep = np.isin(y_state, ["BASE", "STRESS"])
    X, y_state, subjects = X[keep], y_state[keep], subjects[keep]

    if cap and cap > 0:
        idx_keep = []
        for s in np.unique(subjects):
            for st in ("BASE", "STRESS"):
                pos = np.where((subjects == s) & (y_state == st))[0]  # already time-ordered
                idx_keep.append(pos[:cap])                             # first `cap` windows
        sel = np.sort(np.concatenate(idx_keep))
        X, y_state, subjects = X[sel], y_state[sel], subjects[sel]

    # subjects that actually carry both classes with enough windows
    subj_list = []
    for s in np.unique(subjects):
        m = subjects == s
        if (y_state[m] == "BASE").sum() >= 3 and (y_state[m] == "STRESS").sum() >= 3:
            subj_list.append(s)
    subj_list = np.array(sorted(subj_list))
    return cfg, X, feats, y_state, subjects, subj_list

def _subject_angle_stream(encoder, X, y_state, subjects, s, gamma):
    """
    Time-ordered rest->stress angle stream for one subject:
    BASE windows (in their CSV order) followed by STRESS windows (in their order).
    Uses a STABLE sort so within-block chronology (the monitoring order) survives.
    Returns (thetas (T,Q), y01 (T,)) or (None, None) if too few windows.
    """
    m = subjects == s
    ys = y_state[m]
    keep = np.isin(ys, ["BASE", "STRESS"])
    if keep.sum() < 6:
        return None, None
    th = encoder.transform_angles(X[m][keep], gamma=gamma)      # NaN-safe (train medians)
    ys = ys[keep]
    order = np.argsort(~(ys == "BASE"), kind="stable")          # BASE(0) first, order kept
    return th[order], (ys[order] == "STRESS").astype(int)

def _ewma(x, halflife=5.0):
    """Causal EWMA smoothing of a score stream (continuous-monitoring smoother)."""
    x = np.asarray(x, float)
    a = 1.0 - np.exp(-np.log(2.0) / max(1e-6, halflife))
    out = np.empty_like(x)
    acc = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        out[i] = acc
    return out

def _stream_metrics(pi, is_stress, alpha, m, halflife=5.0):
    """
    Given calibrated anomaly probs pi (or a bounded score) and the binary
    stress stream, return per-window ranking + EWMA-smoothed AUROC. `p` for the
    alarm rule is 1-pi (conformal p); latency/FAR computed by the reused rule.
    """
    rank = ranking_metrics(pi, is_stress)
    ece, nll, brier = calibration_metrics(pi, is_stress)
    # EWMA-smoothed streamed AUROC (does temporal smoothing help the readout?)
    if len(np.unique(is_stress)) > 1:
        auroc_ewma = float(roc_auc_score(is_stress, _ewma(pi, halflife)))
    else:
        auroc_ewma = np.nan
    p_seq = 1.0 - pi
    latency, far = alarm_metrics(p_seq, is_stress.astype(bool), alpha, m)
    return dict(auroc=rank["auroc"], auprc=rank["auprc"], bacc=rank["bacc"],
                prevalence=rank["prevalence"], auprc_norm=rank["auprc_norm"],
                auroc_ewma=auroc_ewma, ece=ece, nll=nll, brier=brier,
                far=far, latency=(np.nan if latency is None else latency))

#  Per-subject Featurizers  (all consume the SAME per-fold super-node angles)
#  Enriched Quantum Reservoir  (pinned a-priori; readout-capacity-matched to ESN)

#  Motivation (pre-registered, NOT tuned on any outcome): the baseline QRC reads
#  only <Z_q> + <Z_aZ_b> once per window (~12 features for Q=6), while the
#  classical ESN feeds its ridge readout n_res=64 states. That readout-dimension
#  asymmetry, not quantum-vs-classical dynamics, could depress QRC. The `--rich`
#  reservoir removes it with two STANDARD, label-free, training-free enrichments
#  on the SAME fixed reservoir (identical couplings/bias/seed -> identical
#  dynamics; only the measurement changes):
#    (1) fuller observable set : <X_q>, <Y_q>, <Z_q> per qubit + <Z_aZ_b> on pairs
#    (2) temporal multiplexing : read at V virtual sub-steps (Fujii-Nakajima),
#                                reusing the exact scheme validated in qrc_fusion.
#  Feature width becomes (3*Q + |edges|) * V  (Q=6 -> 72; Q=5 -> 60), comparable
#  to the ESN's 64 -> the comparison is no longer confounded by readout capacity.
#  gamma stays PINNED (one fewer contestable knob than label-free gamma-select).
RICH_V_MULTIPLEX = 3          # pinned virtual read sub-steps (matches qrc_fusion)

class RichQuantumReservoir:
    """
    Fixed/untrained quantum reservoir with an enriched, ESN-capacity-matched
    readout: <X>,<Y>,<Z> per qubit + <Z_aZ_b> on coupled pairs, read at V
    temporal-multiplex sub-steps. Same fixed reservoir as QuantumReservoir; only
    the readout is richer. All settings pinned a-priori; nothing tuned on outcome.
    """

    def __init__(self, qcfg, noisy=False, n_qubits=None, v_multiplex=RICH_V_MULTIPLEX):
        self.q = qcfg; self.cfg = qcfg.base
        self.Q = int(n_qubits) if n_qubits is not None else qcfg.qubits()
        self.noisy = noisy; self.V = int(v_multiplex)
        self.edges = _coupling_edges(self.Q, getattr(self.cfg, "entangle_topology", "ring"))
        rng = np.random.default_rng(qcfg.reservoir_seed)   # SAME seed -> same physics
        self.coupling = {e: float(qcfg.coupling_scale * rng.uniform(-np.pi, np.pi))
                         for e in self.edges}
        self.bias = qcfg.input_scale * rng.uniform(-0.5, 0.5, size=self.Q)
        self.virt = qcfg.coupling_scale * rng.uniform(-np.pi, np.pi, size=self.Q)
        self._build()

    def _inject(self, theta):
        s = self.q.input_scale
        for q in range(self.Q):
            qml.RX(s * theta[q] + self.bias[q], wires=q)
            qml.RZ(0.5 * s * theta[q], wires=q)
        for (a, b) in self.edges:
            qml.IsingZZ(self.coupling[(a, b)], wires=[a, b])
        if self.noisy:
            for q in range(self.Q):
                qml.DepolarizingChannel(self.cfg.depol_1q, wires=q)
                qml.AmplitudeDamping(self.cfg.amp_damp, wires=q)
                qml.PhaseDamping(self.cfg.phase_damp, wires=q)
            for (a, b) in self.edges:
                qml.DepolarizingChannel(self.cfg.depol_2q, wires=a)
                qml.DepolarizingChannel(self.cfg.depol_2q, wires=b)

    def _virtual_step(self):
        for q in range(self.Q):
            qml.RX(self.virt[q], wires=q)
        for (a, b) in self.edges:
            qml.IsingZZ(0.5 * self.coupling[(a, b)], wires=[a, b])
        if self.noisy:
            for q in range(self.Q):
                qml.DepolarizingChannel(self.cfg.depol_1q, wires=q)
                qml.AmplitudeDamping(self.cfg.amp_damp, wires=q)

    def _build(self):
        dev = qml.device("default.mixed", wires=self.Q)
        obs = ([qml.PauliX(q) for q in range(self.Q)]
               + [qml.PauliY(q) for q in range(self.Q)]
               + [qml.PauliZ(q) for q in range(self.Q)]
               + [qml.PauliZ(a) @ qml.PauliZ(b) for (a, b) in self.edges])
        self._obs = obs
        self.n_features = len(obs) * self.V

        @qml.qnode(dev, interface="numpy")
        def _read(rho_in):
            qml.QubitDensityMatrix(rho_in, wires=range(self.Q))
            return [qml.expval(o) for o in obs]

        @qml.qnode(dev, interface="numpy")
        def _evolve_input(rho_in, theta):
            qml.QubitDensityMatrix(rho_in, wires=range(self.Q))
            self._inject(theta)
            return qml.density_matrix(wires=range(self.Q))

        @qml.qnode(dev, interface="numpy")
        def _evolve_virt(rho_in):
            qml.QubitDensityMatrix(rho_in, wires=range(self.Q))
            self._virtual_step()
            return qml.density_matrix(wires=range(self.Q))

        self._read, self._evolve_input, self._evolve_virt = _read, _evolve_input, _evolve_virt
        self.rho0 = np.zeros((2 ** self.Q, 2 ** self.Q), dtype=complex); self.rho0[0, 0] = 1.0

    def run_sequence(self, thetas):
        eps = self.q.leak
        rho = self.rho0.copy()
        feats = np.empty((len(thetas), self.n_features), dtype=np.float64)
        for t, th in enumerate(thetas):
            rho = (1.0 - eps) * rho + eps * self.rho0
            rho = np.asarray(self._evolve_input(rho, th))
            reads = [np.real(np.asarray(self._read(rho)))]
            for _ in range(self.V - 1):
                rho = np.asarray(self._evolve_virt(rho))
                reads.append(np.real(np.asarray(self._read(rho))))
            feats[t] = np.concatenate(reads)
        return feats

class _MemorylessModel:
    """
    Logistic regression on standardized super-node angles. No reservoir, no
    memory -- the 'is there any cross-subject signal at all' control.
    """
    name = "memoryless"

    def __init__(self, cfg, qcfg, noisy, nq, seed):
        self.C = 1.0

    def fit(self, seqs):                       # seqs: list of (thetas, y01)
        Xtr = np.vstack([th for th, _ in seqs])
        ytr = np.concatenate([y for _, y in seqs])
        self.mu_, self.sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
        self.clf_ = LogisticRegression(max_iter=1000, C=self.C)
        self.clf_.fit((Xtr - self.mu_) / self.sd_, ytr)
        return self

    def score_stream(self, thetas):
        return self.clf_.predict_proba((thetas - self.mu_) / self.sd_)[:, 1]

class _ReservoirModel:
    """
    QRC(+skip) or ESN(+skip): drive a reservoir over each subject's angle
    stream, optionally concatenate the input angles (skip connection), then a
    ridge readout. `kind` in {'qrc','esn'}.
    """

    def __init__(self, cfg, qcfg, noisy, nq, seed, kind="qrc", skip=True, rich=False):
        self.cfg, self.qcfg, self.noisy = cfg, qcfg, noisy
        self.kind, self.skip, self.seed = kind, skip, seed
        self.rich = rich          # enriched quantum readout (QRC only; ESN already 64-dim)
        self.washout = qcfg.washout
        self.name = kind

    def _reservoir(self, Qeff):
        if self.kind == "qrc":
            if self.rich:
                return RichQuantumReservoir(self.qcfg, noisy=self.noisy, n_qubits=Qeff)
            return QuantumReservoir(self.qcfg, noisy=self.noisy, n_qubits=Qeff)
        return ClassicalESN(n_in=Qeff, leak=self.qcfg.leak, seed=self.seed)

    def _feat(self, res, thetas):
        Z = res.run_sequence(thetas)
        if self.skip:
            Z = np.hstack([Z, thetas])         # skip connection: reservoir + raw input
        return Z

    def fit(self, seqs):
        Qeff = seqs[0][0].shape[1]
        self.res_ = self._reservoir(Qeff)
        Xtr, ytr = [], []
        for th, y in seqs:
            Z = self._feat(self.res_, th)[self.washout:]
            yb = y[self.washout:]
            if len(Z):
                Xtr.append(Z); ytr.append(yb)
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        self.readout_ = RidgeReadout(self.qcfg.ridge_lambda).fit(Xtr, ytr)
        return self

    def score_stream(self, thetas):
        Z = self._feat(self.res_, thetas)
        return self.readout_.score(Z)          # bounded (0,1), higher => stress

class _QSVMModel:
    """
    Static quantum baseline: per-window fidelity-kernel SVM. Angles -> PCA(nq)
    -> Y-rotation angle embedding; kernel K(x,x') = |<phi(x')|phi(x)>|^2. Clean
    simulator (default.qubit). O(N^2) circuits -> fit windows are subsampled to
    `qcap`. Memoryless by construction (no reservoir).
    """
    name = "qsvm"

    def __init__(self, cfg, qcfg, noisy, nq, seed, qcap=160):
        self.nq, self.seed, self.qcap = nq, seed, qcap
        self.dev = qml.device("default.qubit", wires=nq)

        @qml.qnode(self.dev)
        def _overlap(x1, x2):
            qml.AngleEmbedding(x1, wires=range(nq), rotation="Y")
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(nq), rotation="Y")
            return qml.probs(wires=range(nq))
        self._overlap = _overlap

    def _pca_scale(self, Xtr):
        self.pca_ = PCA(n_components=self.nq, random_state=self.seed).fit(Xtr)
        Ztr = self.pca_.transform(Xtr)
        self.mu_, self.sd_ = Ztr.mean(0), Ztr.std(0) + 1e-9
        return self._emb(Xtr)

    def _emb(self, X):
        Z = (self.pca_.transform(X) - self.mu_) / self.sd_
        return np.pi * np.tanh(Z)              # bounded rotation angles

    def _kernel(self, A, B):
        K = np.empty((len(A), len(B)))
        for i in range(len(A)):
            for j in range(len(B)):
                K[i, j] = float(self._overlap(A[i], B[j])[0])
        return K

    def fit(self, seqs):
        Xtr = np.vstack([th for th, _ in seqs])
        ytr = np.concatenate([y for _, y in seqs])
        rng = np.random.default_rng(self.seed)
        if len(Xtr) > self.qcap:               # subsample for the O(N^2) kernel
            sel = rng.choice(len(Xtr), self.qcap, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        self.Aref_ = self._pca_scale(Xtr)
        K = self._kernel(self.Aref_, self.Aref_)
        self.svc_ = SVC(kernel="precomputed", C=1.0, probability=False).fit(K, ytr)
        return self

    def score_stream(self, thetas):
        A = self._emb(thetas)
        K = self._kernel(A, self.Aref_)
        d = self.svc_.decision_function(K)
        return 1.0 / (1.0 + np.exp(-d))        # squash to (0,1) so metrics align

class _VQCModel:
    """
    Static quantum baseline WITH trainable parameters (StronglyEntanglingLayers)
    -> the barren-plateau-exposed contrast to the untrained QRC. Angles -> PCA(nq)
    -> AngleEmbedding + variational layers; expval(Z0) -> sigmoid. Trained per fold
    by Adam on subsampled fit windows. default.mixed (with the NISQ channel) when
    noisy -> this is the arm whose robustness we compare against QRC across kappa.
    """
    name = "vqc"

    def __init__(self, cfg, qcfg, noisy, nq, seed, qcap=160, layers=2, steps=40):
        self.cfg, self.nq, self.noisy = cfg, nq, noisy
        self.seed, self.qcap, self.layers, self.steps = seed, qcap, layers, steps
        dev = qml.device("default.mixed" if noisy else "default.qubit", wires=nq)

        @qml.qnode(dev, interface="autograd")
        def _circ(w, x):
            qml.AngleEmbedding(x, wires=range(nq), rotation="Y")
            qml.StronglyEntanglingLayers(w, wires=range(nq))
            if noisy:                          # NISQ channel on the static circuit
                for q in range(nq):
                    qml.DepolarizingChannel(cfg.depol_1q, wires=q)
                    qml.AmplitudeDamping(cfg.amp_damp, wires=q)
                    qml.PhaseDamping(cfg.phase_damp, wires=q)
            return qml.expval(qml.PauliZ(0))
        self._circ = _circ

    def _pca_scale(self, Xtr):
        self.pca_ = PCA(n_components=self.nq, random_state=self.seed).fit(Xtr)
        Ztr = self.pca_.transform(Xtr)
        self.mu_, self.sd_ = Ztr.mean(0), Ztr.std(0) + 1e-9
        return self._emb(Xtr)

    def _emb(self, X):
        return np.pi * np.tanh((self.pca_.transform(X) - self.mu_) / self.sd_)

    def _prob(self, w, X):
        z = np.array([float(self._circ(w, x)) for x in X])
        return 0.5 * (1.0 - z)                 # <Z>=+1 -> 0 (base), <Z>=-1 -> 1 (stress)

    def fit(self, seqs):
        Xtr = np.vstack([th for th, _ in seqs])
        ytr = np.concatenate([y for _, y in seqs]).astype(float)
        rng = np.random.default_rng(self.seed)
        if len(Xtr) > self.qcap:
            sel = rng.choice(len(Xtr), self.qcap, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        A = self._pca_scale(Xtr)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.layers, n_wires=self.nq)
        w = pnp.array(0.1 * np.random.default_rng(self.seed).standard_normal(shape),
                      requires_grad=True)
        yv = pnp.array(ytr, requires_grad=False)

        def loss(w):
            z = pnp.stack([self._circ(w, x) for x in A])
            p = 0.5 * (1.0 - z)
            p = pnp.clip(p, 1e-6, 1 - 1e-6)
            return -pnp.mean(yv * pnp.log(p) + (1 - yv) * pnp.log(1 - p))

        opt = qml.AdamOptimizer(0.1)
        for _ in range(self.steps):
            w = opt.step(loss, w)
        self.w_ = w
        return self

    def score_stream(self, thetas):
        return self._prob(self.w_, self._emb(thetas))

#  Trained Quantum Recurrent Net  (QLSTM / QGRU)  -- the "best among quantum" test

#  Variational quantum circuits inside the recurrent gates: the TRAINED,
#  memory-bearing quantum contrast to the UNTRAINED quantum reservoir (QRC).
#  Same .fit(seqs)/.score_stream(thetas) contract as every other arm, so it runs
#  under the identical honest LOSO + conformal + NISQ protocol. This is the arm
#  that makes "QRC is best among quantum" airtight -- QRC vs a trained quantum
#  recurrent network, not just vs the static VQC.
#
#  QLSTM (Chen et al. 2020): gates f,i,g,o are VQCs; c=f*c+i*g~, h=o*tanh(c).
#  QGRU: gates r,z and candidate g~ (over r*h); h=(1-z)*h+z*g~. Fewer gates ->
#        cheaper, and the variant the literature reports as often strongest.
#
#  COMPUTE: BPTT through quantum gates is heavy. Tractability knobs are class
#  defaults (nq from --nq, layers, train steps, subsequence length/count). This
#  is an OFFLINE arm; defaults are set small so it at least runs interactively.
#  It is built to find the honest answer, not to guarantee QRC wins.
def _sig(z):
    return 1.0 / (1.0 + pnp.exp(-z))

class _QRNNModel:
    name = "qrnn"

    def __init__(self, cfg, qcfg, noisy, nq, seed, kind="qlstm",
                 layers=1, steps=15, seq_len=24, n_seq=8, lr=0.05):
        self.cfg, self.nq, self.noisy, self.seed = cfg, int(nq), noisy, int(seed)
        self.kind = kind
        self.layers, self.steps, self.seq_len, self.n_seq, self.lr = \
            layers, steps, seq_len, n_seq, lr
        self.gates = ["f", "i", "g", "o"] if kind == "qlstm" else ["r", "z", "g"]
        dev = qml.device("default.mixed" if noisy else "default.qubit", wires=self.nq)

        @qml.qnode(dev, interface="autograd")
        def _circ(w, x):
            qml.AngleEmbedding(x, wires=range(self.nq), rotation="Y")
            qml.StronglyEntanglingLayers(w, wires=range(self.nq))
            if noisy:                                  # NISQ channel
                for q in range(self.nq):
                    qml.DepolarizingChannel(cfg.depol_1q, wires=q)
                    qml.AmplitudeDamping(cfg.amp_damp, wires=q)
                    qml.PhaseDamping(cfg.phase_damp, wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(self.nq)]
        self._circ = _circ
        self._wsh = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=self.nq)
        # flat-vector parameter layout (qml optimizers like a single array)
        nq = self.nq
        self._blocks = []                              # (name, shape)
        for g in self.gates:
            self._blocks += [(f"{g}_W", (nq, 2 * nq)), (f"{g}_b", (nq,)),
                             (f"{g}_w", self._wsh)]
        self._blocks += [("out_W", (nq,)), ("out_b", ())]
        self._sizes = [int(np.prod(s)) if s else 1 for _, s in self._blocks]
        self._offsets = np.cumsum([0] + self._sizes)

    # flat <-> dict
    def _unpack(self, v):
        P = {}
        for i, (name, shape) in enumerate(self._blocks):
            seg = v[self._offsets[i]:self._offsets[i + 1]]
            P[name] = seg.reshape(shape) if shape else seg[0]
        return P

    # input embedding: PCA(nq) fit on train windows
    def _emb_setup(self, Xtr):
        self.pca_ = PCA(n_components=self.nq, random_state=self.seed).fit(Xtr)
        Z = self.pca_.transform(Xtr)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-9

    def _x(self, thetas):
        return np.pi * np.tanh((self.pca_.transform(thetas) - self.mu_) / self.sd_)

    # one recurrent pass over a sequence -> per-step stress prob
    def _forward(self, v, X_seq):
        P = self._unpack(v); nq = self.nq
        h = pnp.zeros(nq); c = pnp.zeros(nq); outs = []
        for xt in X_seq:
            def gate(g, hin):
                ang = pnp.pi * pnp.tanh(P[f"{g}_W"] @ pnp.concatenate([xt, hin]) + P[f"{g}_b"])
                return pnp.stack(self._circ(P[f"{g}_w"], ang))
            if self.kind == "qlstm":
                f = _sig(gate("f", h)); i = _sig(gate("i", h))
                g = pnp.tanh(gate("g", h)); o = _sig(gate("o", h))
                c = f * c + i * g; h = o * pnp.tanh(c)
            else:                                      # qgru
                r = _sig(gate("r", h)); z = _sig(gate("z", h))
                g = pnp.tanh(gate("g", r * h)); h = (1 - z) * h + z * g
            outs.append(h)
        H = pnp.stack(outs)
        return _sig(H @ P["out_W"] + P["out_b"])       # [T] stress prob

    def fit(self, seqs):
        Xall = np.vstack([th for th, _ in seqs])
        self._emb_setup(Xall)
        rng = np.random.default_rng(self.seed)
        v = pnp.array(0.1 * rng.standard_normal(int(self._offsets[-1])), requires_grad=True)
        subs = []
        for th, y in seqs:
            X = self._x(th)
            if len(X) < 4:
                continue
            L = min(self.seq_len, len(X))
            for _ in range(2):
                st = int(rng.integers(0, len(X) - L + 1))
                subs.append((X[st:st + L], y[st:st + L].astype(float)))
        rng.shuffle(subs); subs = subs[:self.n_seq]
        if not subs:
            raise RuntimeError("no usable subsequences")

        def loss(v):
            tot, n = 0.0, 0
            for X, y in subs:
                pr = pnp.clip(self._forward(v, [pnp.array(x) for x in X]), 1e-6, 1 - 1e-6)
                yv = pnp.array(y)
                tot = tot - pnp.sum(yv * pnp.log(pr) + (1 - yv) * pnp.log(1 - pr)); n += len(y)
            return tot / max(n, 1)

        opt = qml.AdamOptimizer(self.lr)
        for _ in range(self.steps):
            v = opt.step(loss, v)
        self.v_ = np.asarray(v)
        return self

    def score_stream(self, thetas):
        X = self._x(thetas)
        return np.asarray(self._forward(pnp.array(self.v_), [pnp.array(x) for x in X]))

# each factory takes (cfg, qcfg, noisy, nq, seed, rich); rich only affects QRC
_MODEL_FACTORY = {
    "memoryless": lambda cfg, qcfg, noisy, nq, seed, rich: _MemorylessModel(cfg, qcfg, noisy, nq, seed),
    "esn":  lambda cfg, qcfg, noisy, nq, seed, rich: _ReservoirModel(cfg, qcfg, noisy, nq, seed, kind="esn"),
    "qrc":  lambda cfg, qcfg, noisy, nq, seed, rich: _ReservoirModel(cfg, qcfg, noisy, nq, seed, kind="qrc", rich=rich),
    "qsvm": lambda cfg, qcfg, noisy, nq, seed, rich: _QSVMModel(cfg, qcfg, noisy, nq, seed),
    "vqc":  lambda cfg, qcfg, noisy, nq, seed, rich: _VQCModel(cfg, qcfg, noisy, nq, seed),    "qlstm": lambda cfg, qcfg, noisy, nq, seed, rich: _QRNNModel(cfg, qcfg, noisy, nq, seed, kind="qlstm"),
    "qgru":  lambda cfg, qcfg, noisy, nq, seed, rich: _QRNNModel(cfg, qcfg, noisy, nq, seed, kind="qgru"),
}

#  One method, FULL LOSO  (fit/cal split by subject -> conformal -> stream metrics)
def _fold_worker(s_test, method, cfg, qcfg, X, feats, y_state, subjects,
                 subj_list, fold_ix, noisy, nq, seed, rich, verbose, dump=False):
    """
    Compute one LOSO fold (held-out subject s_test). Returns a metrics dict or
    None. Per-fold RNG is seeded from (cfg.seed, seed, fold_ix) so results are
    identical whether folds run serially or in parallel, and independent of order.
    """
    rng = np.random.default_rng([int(cfg.seed), int(seed), int(fold_ix)])
    train = np.array([u for u in subj_list if u != s_test])
    rng.shuffle(train)
    n_cal = max(1, int(round(cfg.cal_frac * len(train))))
    cal_subj = set(train[:n_cal]); fit_subj = [u for u in train if u not in cal_subj]
    if not fit_subj:
        return None

    fit_mask = np.isin(subjects, fit_subj)
    encoder = SuperNodeEncoder(feats, cfg).fit(X[fit_mask])   # train-only impute
    gamma = cfg.gamma

    fit_seqs = []
    for s in fit_subj:
        th, y = _subject_angle_stream(encoder, X, y_state, subjects, s, gamma)
        if th is not None and len(np.unique(y)) > 1:
            fit_seqs.append((th, y))
    if len(fit_seqs) < 2:
        return None

    try:
        model = _MODEL_FACTORY[method](cfg, qcfg, noisy, nq, seed, rich).fit(fit_seqs)
    except Exception as e:
        if verbose:
            print(f"    [{method}] fit failed on fold {s_test}: {e}")
        return None

    a_cal = []
    for s in cal_subj:
        m = subjects == s
        keep = (y_state[m] == "BASE")
        if keep.sum() < 3:
            continue
        sc = model.score_stream(encoder.transform_angles(X[m][keep], gamma=gamma))
        if len(sc):
            a_cal.append(np.asarray(sc))
    if not a_cal:
        return None
    a_cal = np.concatenate(a_cal)

    th_te, y_te = _subject_angle_stream(encoder, X, y_state, subjects, s_test, gamma)
    if th_te is None or len(np.unique(y_te)) < 2:
        return None
    a_te = np.asarray(model.score_stream(th_te))
    p, pi = conformal_pvalues(a_cal, a_te)
    met = _stream_metrics(pi, y_te, cfg.alpha, cfg.debounce_m)
    n_feat = getattr(getattr(model, "res_", None), "n_features", np.nan)
    met.update(dict(method=method, arm=method, subject=s_test,
                    noisy=int(noisy), seed=seed, rich=int(rich),
                    n_readout=n_feat, n_test=len(y_te)))
    if dump:                       # store the conformal p-stream + labels for offline alpha-sweep
        met["p_seq"] = json.dumps([round(float(v), 6) for v in np.asarray(p)])
        met["y_seq"] = json.dumps([int(v) for v in np.asarray(y_te)])
    if verbose:
        print(f"    [{method}] {s_test}: AUROC={met['auroc']:.3f} "
              f"AUPRC={met['auprc']:.3f} EWMA-AUROC={met['auroc_ewma']:.3f} "
              f"FAR={met['far']:.2f} lat={met['latency']}")
    return met


def run_method(method, cfg, qcfg, X, feats, y_state, subjects, subj_list,
               noisy=False, nq=4, seed=42, rich=False, jobs=1, verbose=True, dump=False):
    """Leave-one-subject-out for a single method. Folds are independent; with
    jobs>1 they run in parallel (loky). Returns a per-subject DataFrame."""
    args = (method, cfg, qcfg, X, feats, y_state, subjects, subj_list)
    tasks = list(enumerate(subj_list))
    if jobs and jobs > 1 and _HAVE_JOBLIB:
        results = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(_fold_worker)(s, *args, ix, noisy, nq, seed, rich, False, dump)
            for ix, s in tasks)
    else:
        results = [_fold_worker(s, *args, ix, noisy, nq, seed, rich, verbose, dump)
                   for ix, s in tasks]
    rows = [r for r in results if r is not None]
    return pd.DataFrame(rows)

# Summary + Cross-Method Statistics
def summarize(perfold, metrics=("auroc", "auprc", "auroc_ewma", "far", "latency")):
    """Subject-mean per method with 95% bootstrap CI on AUROC/AUPRC."""
    out = []
    rng = np.random.default_rng(0)
    for (method,), g in perfold.groupby(["method"]):
        row = dict(method=method, n_subj=int(g["subject"].nunique()))
        for mt in metrics:
            vals = g.groupby("subject")[mt].mean().values if mt in g else np.array([])
            vals = vals[np.isfinite(vals)]
            row[f"{mt}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            if mt in ("auroc", "auprc"):
                lo, hi = bootstrap_ci(vals, 2000, 0.95, rng)
                row[f"{mt}_lo"], row[f"{mt}_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out).sort_values("auroc_mean", ascending=False)

def _paired(perfold, a, b, metric="auroc"):
    """Per-subject paired mean diff (a-b) + Wilcoxon signed-rank p."""
    pa = perfold[perfold.method == a].groupby("subject")[metric].mean()
    pb = perfold[perfold.method == b].groupby("subject")[metric].mean()
    j = pd.concat([pa, pb], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 5:
        return dict(contrast=f"{a}-{b}", n=len(j), mean_diff=np.nan, p_wilcoxon=np.nan)
    d = j["a"].values - j["b"].values
    try:
        _, p = stats.wilcoxon(d, alternative="two-sided")
    except Exception:
        p = np.nan
    return dict(contrast=f"{a}-{b}", n=int(len(j)), mean_diff=float(d.mean()),
                median_diff=float(np.median(d)), p_wilcoxon=float(p) if np.isfinite(p) else np.nan)

def cross_method_stats(perfold, cfg):
    """TOST(QRC vs ESN) + best-quantum diffs (QRC vs QSVM, QRC vs VQC)."""
    methods = set(perfold.method.unique())
    # perfold already carries an 'arm' column (== method), which is exactly the
    # per-fold contract tost_paired expects. Pass it straight through (renaming
    # would create a duplicate 'arm' column and break the reindex).
    tost = pd.DataFrame()
    if {"qrc", "esn"} <= methods:
        tost = tost_paired(perfold, "auroc", [("qrc", "esn")], list(cfg.tost_margins))
    diffs = []
    for comp in ("qsvm", "vqc", "esn", "memoryless"):
        if {"qrc", comp} <= methods:
            diffs.append(_paired(perfold, "qrc", comp, "auroc"))
    return tost, pd.DataFrame(diffs)

#  NISQ Noise-Robustness curve  (QRC vs VQC across kappa)
def noise_curve(cfg, qcfg, X, feats, y_state, subjects, subj_list,
                kappas, nq, seed, methods=("qrc", "vqc"), rich=False, jobs=1):
    """
    Subject-mean AUROC vs noise scale kappa for the two noise-bearing quantum
    methods. kappa=0 is the clean simulator; kappa=1 is the exact channel fro the previous work.
    """
    rows = []
    for k in kappas:
        ck = _scale_noise(cfg, k)
        qk = dataclasses.replace(qcfg, base=ck)
        noisy = (k > 0)
        for meth in methods:
            pf = run_method(meth, ck, qk, X, feats, y_state, subjects, subj_list,
                            noisy=noisy, nq=nq, seed=seed, rich=rich, jobs=jobs,
                            verbose=False)
            if len(pf):
                v = pf.groupby("subject")["auroc"].mean().values
                v = v[np.isfinite(v)]
                lo, hi = bootstrap_ci(v, 2000, 0.95, np.random.default_rng(0))
                rows.append(dict(kappa=k, method=meth, auroc_mean=float(v.mean()),
                                 auroc_lo=lo, auroc_hi=hi, n_subj=len(v)))
                print(f"   [noise-curve] kappa={k:.2f} {meth}: "
                      f"AUROC={v.mean():.3f}  (n={len(v)})")
    return pd.DataFrame(rows)

# Main
def main():
    ap = argparse.ArgumentParser(
        description="Honest multi-method continuous-monitoring LOSO harness")
    ap.add_argument("--dataset", default="wesad", choices=["wesad", "road"])
    ap.add_argument("--modality", default="hrv", choices=["hrv", "eda", "resp"])
    ap.add_argument("--cap", type=int, default=30,
                    help="max consecutive windows per (subject,state); 0 = all")
    ap.add_argument("--methods", nargs="+", default=["memoryless", "esn", "qrc"],
                    choices=ALL_METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel LOSO folds (loky); set to CPU count, e.g. 3")
    ap.add_argument("--nq", type=int, default=4, help="qubits for QSVM/VQC (PCA target)")
    ap.add_argument("--noisy", action="store_true",
                    help="run reservoir/VQC arms under the kappa=1 NISQ channel")
    ap.add_argument("--rich", action="store_true",
                    help="enriched QRC readout (X/Y/Z + ZZ, V=3 multiplex) matched "
                         "to the ESN's ~64-dim readout; pinned a-priori, QRC only")
    ap.add_argument("--dump-streams", action="store_true", dest="dump_streams",
                    help="also write per-window conformal p-streams for the offline "
                         "alpha-sweep operating-curve analysis (harness_operating_curve.py)")
    ap.add_argument("--noise-curve", action="store_true",
                    help="also sweep the QRC-vs-VQC noise-robustness curve")
    ap.add_argument("--kappas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--out", default="Output/harness")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--road-threshold", type=float, default=0.5, dest="road_threshold",
                    help="AffectiveROAD 0..1 metric -> BASE/STRESS split (default 0.5)")
    a = ap.parse_args()

    t0 = time.time()
    cfg, X, feats, y_state, subjects, subj_list = load_stream(
        a.dataset, a.modality, a.cap, seed=a.seeds[0], synthetic=a.synthetic,
        road_threshold=a.road_threshold)
    qcfg = QRCConfig(base=cfg)
    print(f"\n=== HARNESS :: {a.dataset}/{a.modality} | cap={a.cap} | "
          f"{len(subj_list)} usable subjects | methods={a.methods} | "
          f"noisy={a.noisy} ===")

    all_pf = []
    for method in a.methods:
        for sd in a.seeds:
            print(f"\n-- method={method} seed={sd} --")
            ts = time.time()
            pf = run_method(method, cfg, qcfg, X, feats, y_state, subjects,
                            subj_list, noisy=a.noisy, nq=a.nq, seed=sd,
                            rich=a.rich, jobs=a.jobs, dump=a.dump_streams)
            print(f"   ({method} seed {sd}: {time.time()-ts:.0f}s, "
                  f"{pf['subject'].nunique() if len(pf) else 0} folds)")
            all_pf.append(pf)
    perfold = pd.concat(all_pf, ignore_index=True) if all_pf else pd.DataFrame()

    d = _out_dir(a.out, a.dataset, a.modality); os.makedirs(d, exist_ok=True)
    # split the (optional) per-window p-streams out into their own file; keep the
    # perfold table lean (streams enable the offline alpha-sweep operating curve).
    if a.dump_streams and len(perfold) and "p_seq" in perfold.columns:
        perfold.drop(columns=["p_seq", "y_seq"]).to_csv(
            os.path.join(d, f"harness_perfold_{a.modality}.csv"), index=False)
        perfold[["method", "subject", "seed", "p_seq", "y_seq"]].to_csv(
            os.path.join(d, f"harness_streams_{a.modality}.csv"), index=False)
        print(f"   [dump] wrote per-window p-streams -> harness_streams_{a.modality}.csv")
    else:
        perfold.to_csv(os.path.join(d, f"harness_perfold_{a.modality}.csv"), index=False)

    if len(perfold):
        summ = summarize(perfold)
        summ.to_csv(os.path.join(d, f"harness_summary_{a.modality}.csv"), index=False)
        print("\n=== SUBJECT-MEAN SUMMARY (LOSO) ===")
        print(summ.to_string(index=False))

        tost, diffs = cross_method_stats(perfold, cfg)
        if len(tost):
            tost.to_csv(os.path.join(d, f"harness_tost_qrc_vs_esn_{a.modality}.csv"), index=False)
            print("\n=== EQUIVALENCE  TOST(QRC vs ESN) on AUROC ===")
            print(tost.to_string(index=False))
        if len(diffs):
            diffs.to_csv(os.path.join(d, f"harness_bestquantum_{a.modality}.csv"), index=False)
            print("\n=== BEST-QUANTUM / paired diffs on AUROC (QRC - X) ===")
            print(diffs.to_string(index=False))

    if a.noise_curve:
        print("\n=== NISQ NOISE-ROBUSTNESS CURVE (QRC vs VQC) ===")
        nc = noise_curve(cfg, qcfg, X, feats, y_state, subjects, subj_list,
                         a.kappas, a.nq, a.seeds[0], rich=a.rich, jobs=a.jobs)
        if len(nc):
            nc.to_csv(os.path.join(d, f"harness_noise_curve_{a.modality}.csv"), index=False)
            print(nc.to_string(index=False))

    # manifest for provenance
    manifest = dict(dataset=a.dataset, modality=a.modality, cap=a.cap,
                    methods=a.methods, seeds=a.seeds, nq=a.nq, noisy=a.noisy, jobs=a.jobs,
                    noise_curve=a.noise_curve, kappas=a.kappas,
                    encoder="per_fold_fit_train_only_impute",
                    rich=a.rich, rich_v_multiplex=(RICH_V_MULTIPLEX if a.rich else None),
                    rich_observables=("X,Y,Z per qubit + ZZ on edges" if a.rich else None),
                    leak=qcfg.leak, washout=qcfg.washout,
                    reservoir_seed=qcfg.reservoir_seed,
                    cal_frac=cfg.cal_frac, alpha=cfg.alpha, debounce_m=cfg.debounce_m,
                    n_usable_subjects=int(len(subj_list)),
                    wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(d, f"harness_manifest_{a.modality}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s. Wrote -> {d}")

if __name__ == "__main__":
    main()