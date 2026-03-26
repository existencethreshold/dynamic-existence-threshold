#!/usr/bin/env python3
"""
Run 1: Null Model for I-D Anticorrelation (v2)

Three null models:
1. LABEL PERMUTATION — shuffle Wake/N3 labels, recompute AUC. Tests if classification is real.
2. CROSS-BAND SHUFFLE — shuffle which band maps to which layer. Tests if band structure matters.
3. PHASE RANDOMIZATION — randomize phase of each band (preserve power spectrum, destroy structure).
   Tests if temporal dynamics matter.

The original shuffle (within-band time shuffle) was a NO-OP because R and D only depend on
per-band variance, which is preserved by within-band shuffling.
"""
import numpy as np
import os, sys, json, time
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from scipy.signal import butter, sosfilt
from scipy.spatial.distance import jensenshannon
from eeg_loader_direct import load_dataset

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')

N_WORKERS = max(1, cpu_count() - 2)  # leave 2 cores free

# ============================================================
# LAYER / PHI FUNCTIONS
# ============================================================
BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
EFF_THRESHOLD = 2.0

def bandpass(data, low, high, sfreq, order=4):
    nyq = sfreq / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return np.zeros_like(data)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos, data)

def extract_layers(eeg_data, sfreq, window_sec=5.0, step_sec=2.5):
    full_layers = np.zeros((5, len(eeg_data)))
    for i, (low, high) in enumerate(BANDS):
        full_layers[i] = bandpass(eeg_data, low, high, sfreq)
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    windows = []
    start = 0
    while start + window_samples <= full_layers.shape[1]:
        windows.append(full_layers[:, start:start + window_samples])
        start += step_samples
    return windows

def compute_R(layers):
    n_layers = layers.shape[0]
    if n_layers < 2: return 0.0
    powers = np.array([np.var(layers[i]) for i in range(n_layers)])
    total = np.sum(powers)
    if total == 0: return 0.0
    p = powers / total
    H = -np.sum(p * np.log(p + 1e-15))
    eff = float(np.exp(H))
    if eff < EFF_THRESHOLD: return 0.0
    return float(np.clip((eff - 1) / (n_layers - 1), 0.0, 1.0))

def compute_S(layers):
    n_layers = layers.shape[0]
    window_size = layers.shape[1]
    powers = np.array([np.var(layers[i]) for i in range(n_layers)])
    total = np.sum(powers)
    if total == 0: return 0.0
    p = powers / total
    H = -np.sum(p * np.log(p + 1e-15))
    eff = float(np.exp(H))
    if eff < EFF_THRESHOLD: return 0.0
    coverage = eff / n_layers
    H_max = np.log(n_layers)
    evenness = H / H_max if H_max > 0 else 0.0
    n_sub = 8
    sub_size = window_size // n_sub
    if sub_size < 2:
        return 0.0
    dominant = []
    for k in range(n_sub):
        s0, s1 = k * sub_size, (k + 1) * sub_size
        sub_pow = np.array([np.var(layers[i, s0:s1]) for i in range(n_layers)])
        dominant.append(int(np.argmax(sub_pow)))
    pattern_diversity = len(set(dominant)) / n_sub
    return float(np.clip(coverage * (0.5 * evenness + 0.5 * pattern_diversity), 0.0, 1.0))

def compute_D(layers):
    n_layers = layers.shape[0]
    if n_layers < 2: return 0.0
    powers = np.array([np.var(layers[i]) for i in range(n_layers)])
    total = np.sum(powers)
    if total == 0: return 0.0
    p = powers / total
    q = np.ones(n_layers) / n_layers
    return float(jensenshannon(p, q, base=2))

def compute_epoch_I(epoch_data, sfreq):
    windows = extract_layers(epoch_data, sfreq)
    Is = []
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        Is.append(R * S)
    return float(np.mean(Is))

def compute_epoch_ID(epoch_data, sfreq):
    windows = extract_layers(epoch_data, sfreq)
    Is, Ds = [], []
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        D = compute_D(w)
        Is.append(R * S)
        Ds.append(D)
    return float(np.mean(Is)), float(np.mean(Ds))

# ============================================================
# NULL MODEL 2: CROSS-BAND SHUFFLE
# ============================================================
def extract_layers_band_shuffled(eeg_data, sfreq, rng, window_sec=5.0, step_sec=2.5):
    """Extract layers but randomly permute which band maps to which layer index."""
    full_layers = np.zeros((5, len(eeg_data)))
    for i, (low, high) in enumerate(BANDS):
        full_layers[i] = bandpass(eeg_data, low, high, sfreq)
    # Shuffle band assignments
    perm = rng.permutation(5)
    full_layers = full_layers[perm]
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    windows = []
    start = 0
    while start + window_samples <= full_layers.shape[1]:
        windows.append(full_layers[:, start:start + window_samples])
        start += step_samples
    return windows

def compute_epoch_I_band_shuffled(epoch_data, sfreq, rng):
    windows = extract_layers_band_shuffled(epoch_data, sfreq, rng)
    Is = []
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        Is.append(R * S)
    return float(np.mean(Is))

# ============================================================
# NULL MODEL 3: PHASE RANDOMIZATION
# ============================================================
def phase_randomize(data, rng):
    """Randomize phase while preserving power spectrum."""
    n = len(data)
    ft = np.fft.rfft(data)
    phases = rng.uniform(0, 2 * np.pi, len(ft))
    phases[0] = 0  # keep DC
    if n % 2 == 0:
        phases[-1] = 0  # keep Nyquist
    ft_rand = np.abs(ft) * np.exp(1j * phases)
    return np.fft.irfft(ft_rand, n=n)

def compute_epoch_I_phase_rand(epoch_data, sfreq, rng):
    """Compute I on phase-randomized data."""
    rand_data = phase_randomize(epoch_data, rng)
    return compute_epoch_I(rand_data, sfreq)

# ============================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================
def _worker_epoch_ID(args):
    data, sfreq = args
    return compute_epoch_ID(data, sfreq)

def _worker_epoch_I_band_shuf(args):
    data, sfreq, seed = args
    rng = np.random.RandomState(seed)
    return compute_epoch_I_band_shuffled(data, sfreq, rng)

def _worker_epoch_I_phase(args):
    data, sfreq, seed = args
    rng = np.random.RandomState(seed)
    return compute_epoch_I_phase_rand(data, sfreq, rng)

# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    N_SUBJECTS = 50
    N_PERM = 1000

    print("=" * 60)
    print("  NULL MODEL v2: THREE TESTS")
    print("  1. Label permutation (is classification real?)")
    print("  2. Cross-band shuffle (does band identity matter?)")
    print("  3. Phase randomization (do temporal dynamics matter?)")
    print("=" * 60)

    epochs = load_dataset(N_SUBJECTS)

    # Filter to Wake and N3
    epochs_wn3 = [e for e in epochs if e['stage'] in ('W', 'N3')]
    labels = np.array([1 if e['stage'] == 'W' else 0 for e in epochs_wn3])
    print(f"\nWake+N3: {len(epochs_wn3)} epochs (W={labels.sum()}, N3={(1-labels).sum()})")

    # --- COMPUTE REAL I and D ---
    print(f"\n[Step 1] Computing REAL I and D ({N_WORKERS} workers)...", flush=True)
    args_list = [(ep['data'], ep['sfreq']) for ep in epochs_wn3]
    with Pool(N_WORKERS) as pool:
        results_id = pool.map(_worker_epoch_ID, args_list, chunksize=100)
    real_I = np.array([r[0] for r in results_id])
    real_D = np.array([r[1] for r in results_id])

    real_auc = roc_auc_score(labels, real_I)
    real_corr, _ = spearmanr(real_I, real_D)
    print(f"  REAL AUC(I): {real_auc:.4f}")
    print(f"  REAL Spearman(I,D): {real_corr:.4f}")

    # --- NULL 1: LABEL PERMUTATION ---
    print(f"\n[Step 2] Label permutation ({N_PERM} rounds)...", flush=True)
    rng = np.random.RandomState(42)
    perm_aucs = []
    for p in range(N_PERM):
        shuffled_labels = rng.permutation(labels)
        perm_aucs.append(roc_auc_score(shuffled_labels, real_I))
    perm_aucs = np.array(perm_aucs)
    p_value = np.mean(perm_aucs >= real_auc)
    print(f"  Permutation AUC: {np.mean(perm_aucs):.4f} ± {np.std(perm_aucs):.4f}")
    print(f"  Real AUC: {real_auc:.4f}")
    print(f"  p-value: {p_value:.6f} ({'SIGNIFICANT' if p_value < 0.001 else 'NOT SIGNIFICANT'})")

    # --- NULL 2: CROSS-BAND SHUFFLE ---
    print(f"\n[Step 3] Cross-band shuffle (5 rounds, {N_WORKERS} workers)...", flush=True)
    band_shuf_aucs = []
    for rnd in range(5):
        base_seed = 100 + rnd * len(epochs_wn3)
        args_band = [(ep['data'], ep['sfreq'], base_seed + i) for i, ep in enumerate(epochs_wn3)]
        with Pool(N_WORKERS) as pool:
            shuf_I = np.array(pool.map(_worker_epoch_I_band_shuf, args_band, chunksize=100))
        auc = roc_auc_score(labels, shuf_I)
        band_shuf_aucs.append(auc)
        print(f"  Round {rnd+1}: AUC = {auc:.4f}", flush=True)
    band_shuf_mean = np.mean(band_shuf_aucs)
    print(f"  Mean cross-band AUC: {band_shuf_mean:.4f}")
    print(f"  Real AUC: {real_auc:.4f}")
    print(f"  Drop: {real_auc - band_shuf_mean:.4f}")
    band_matters = real_auc - band_shuf_mean > 0.01
    print(f"  Band identity matters: {'YES' if band_matters else 'NO (R×S is band-order invariant)'}")

    # --- NULL 3: PHASE RANDOMIZATION ---
    print(f"\n[Step 4] Phase randomization (5 rounds, {N_WORKERS} workers)...", flush=True)
    phase_aucs = []
    for rnd in range(5):
        base_seed = 200 + rnd * len(epochs_wn3)
        args_phase = [(ep['data'], ep['sfreq'], base_seed + i) for i, ep in enumerate(epochs_wn3)]
        with Pool(N_WORKERS) as pool:
            phase_I = np.array(pool.map(_worker_epoch_I_phase, args_phase, chunksize=100))
        auc = roc_auc_score(labels, phase_I)
        phase_aucs.append(auc)
        print(f"  Round {rnd+1}: AUC = {auc:.4f}", flush=True)
    phase_mean = np.mean(phase_aucs)
    print(f"  Mean phase-rand AUC: {phase_mean:.4f}")
    print(f"  Real AUC: {real_auc:.4f}")
    print(f"  Drop: {real_auc - phase_mean:.4f}")

    # --- EXPLANATION OF I-D CORRELATION ---
    print(f"\n[Step 5] I-D correlation analysis...", flush=True)
    # R and D share the same input: power distribution across bands.
    # R = f(entropy(powers)), D = JSD(powers, uniform)
    # When entropy is high → R is high, D is low → anticorrelation is mechanical.
    # This is NOT a bug — it's the mathematical structure of the I-D space.
    # The question is: do conscious states preferentially occupy the high-R, low-D region?
    print(f"  I-D correlation: {real_corr:.4f} (expected: strongly negative)")
    print(f"  This is a MATHEMATICAL PROPERTY of R and D sharing the power distribution.")
    print(f"  R ∝ exp(H(p)) and D = JSD(p, uniform) are complementary views of the same quantity.")
    print(f"  The paper's claim is about WHERE states sit in this space, not about the correlation itself.")

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  1. Label permutation:  p < {max(p_value, 0.001):.3f}  → Classification is {'REAL' if p_value < 0.001 else 'NOT SIGNIFICANT'}")
    print(f"  2. Cross-band shuffle: Δ AUC = {real_auc - band_shuf_mean:+.4f} → Band identity {'MATTERS' if band_matters else 'does NOT matter'}")
    print(f"  3. Phase randomization: Δ AUC = {real_auc - phase_mean:+.4f} → Temporal dynamics {'MATTER' if real_auc - phase_mean > 0.02 else 'do NOT matter'}")
    print(f"  4. I-D anticorrelation: MATHEMATICAL (not neural) — R and D are complementary functions of spectral power distribution")
    print(f"")

    if p_value < 0.001:
        if abs(real_auc - phase_mean) < 0.02:
            verdict = "CLASSIFICATION REAL, DRIVEN BY SPECTRAL POWER (not temporal dynamics)"
        else:
            verdict = "CLASSIFICATION REAL, PARTIALLY TEMPORAL"
    else:
        verdict = "CLASSIFICATION NOT SIGNIFICANT"
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    # --- SAVE ---
    results = {
        'n_subjects': N_SUBJECTS,
        'n_epochs_wn3': len(epochs_wn3),
        'n_wake': int(labels.sum()),
        'n_n3': int((1 - labels).sum()),
        'real': {
            'auc_I': float(real_auc),
            'id_correlation': float(real_corr),
        },
        'null_1_label_permutation': {
            'n_permutations': N_PERM,
            'perm_auc_mean': float(np.mean(perm_aucs)),
            'perm_auc_std': float(np.std(perm_aucs)),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.001),
        },
        'null_2_cross_band': {
            'aucs': [float(a) for a in band_shuf_aucs],
            'mean_auc': float(band_shuf_mean),
            'delta': float(real_auc - band_shuf_mean),
            'band_matters': bool(band_matters),
        },
        'null_3_phase_randomization': {
            'aucs': [float(a) for a in phase_aucs],
            'mean_auc': float(phase_mean),
            'delta': float(real_auc - phase_mean),
            'temporal_matters': bool(real_auc - phase_mean > 0.02),
        },
        'id_correlation_explanation': 'Mathematical: R and D are complementary functions of the spectral power distribution. Anticorrelation is structural, not neural.',
        'verdict': verdict,
    }

    with open(os.path.join(RESULTS, 'null_model_v2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.join(RESULTS, "null_model_v2_results.json")}")

    # --- PLOT ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Label permutation distribution
    ax = axes[0, 0]
    ax.hist(perm_aucs, bins=50, color='gray', alpha=0.7, label=f'Permuted (n={N_PERM})')
    ax.axvline(real_auc, color='red', linewidth=2, linestyle='-', label=f'Real AUC={real_auc:.3f}')
    ax.set_xlabel('AUC'); ax.set_ylabel('Count')
    ax.set_title(f'Null 1: Label Permutation (p<{max(p_value, 0.001):.3f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Cross-band shuffle comparison
    ax = axes[0, 1]
    ax.bar(['Real', 'Band-shuffled\n(mean±std)'],
           [real_auc, band_shuf_mean],
           yerr=[0, np.std(band_shuf_aucs)],
           color=['steelblue', 'coral'], alpha=0.8, capsize=10)
    ax.set_ylabel('AUC')
    ax.set_title(f'Null 2: Cross-Band Shuffle (Δ={real_auc - band_shuf_mean:+.4f})')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Phase randomization comparison
    ax = axes[1, 0]
    ax.bar(['Real', 'Phase-randomized\n(mean±std)'],
           [real_auc, phase_mean],
           yerr=[0, np.std(phase_aucs)],
           color=['steelblue', 'orange'], alpha=0.8, capsize=10)
    ax.set_ylabel('AUC')
    ax.set_title(f'Null 3: Phase Randomization (Δ={real_auc - phase_mean:+.4f})')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: I vs D scatter with explanation
    ax = axes[1, 1]
    wake_mask = labels == 1
    n3_mask = labels == 0
    ax.scatter(real_I[wake_mask], real_D[wake_mask], c='green', alpha=0.1, s=3, label='Wake')
    ax.scatter(real_I[n3_mask], real_D[n3_mask], c='red', alpha=0.3, s=3, label='N3')
    ax.set_xlabel('I (R×S)'); ax.set_ylabel('D')
    ax.set_title(f'I-D Space (ρ={real_corr:.3f}, mathematical)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Null Model Results — {verdict}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'null_model_v2_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(RESULTS, "null_model_v2_plot.png")}")

    elapsed = time.time() - t_start
    print(f"\nTOTAL TIME: {elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
