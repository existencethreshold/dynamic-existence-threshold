#!/usr/bin/env python3
"""
Layer Sensitivity Test — EEG (Sleep-EDF)
Tests DET framework at N ∈ {3, 4, 5, 6, 8} frequency band decompositions.
Wake vs N3 classification — same as Phase 7b Test 5.
"""
import numpy as np
import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from scipy.signal import butter, sosfilt

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')


def flush(msg):
    print(msg, flush=True)

# ============================================================
# FREQUENCY BAND DEFINITIONS FOR EACH N
# ============================================================
BAND_CONFIGS = {
    3: [
        (0.5,  8.0,  'low'),       # Delta+Theta
        (8.0,  30.0, 'mid'),       # Alpha+Beta
        (30.0, 45.0, 'high'),      # Gamma
    ],
    4: [
        (0.5,  4.0,  'delta'),
        (4.0,  13.0, 'theta_alpha'),  # Theta+Alpha merged
        (13.0, 30.0, 'beta'),
        (30.0, 45.0, 'gamma'),
    ],
    5: [
        (0.5,  4.0,  'delta'),
        (4.0,  8.0,  'theta'),
        (8.0,  13.0, 'alpha'),
        (13.0, 30.0, 'beta'),
        (30.0, 45.0, 'gamma'),
    ],
    6: [
        (0.5,  2.0,  'delta_slow'),   # Split delta
        (2.0,  4.0,  'delta_fast'),
        (4.0,  8.0,  'theta'),
        (8.0,  13.0, 'alpha'),
        (13.0, 30.0, 'beta'),
        (30.0, 45.0, 'gamma'),
    ],
    8: [
        (0.5,  2.0,  'delta_low'),
        (2.0,  4.0,  'delta_high'),
        (4.0,  6.0,  'theta_low'),
        (6.0,  8.0,  'theta_high'),
        (8.0,  13.0, 'alpha'),
        (13.0, 20.0, 'beta_low'),
        (20.0, 30.0, 'beta_high'),
        (30.0, 45.0, 'gamma'),
    ],
}

def bandpass(data, low, high, sfreq, order=4):
    nyq = sfreq / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return np.zeros_like(data)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos, data)

def extract_layers_n(eeg_data, sfreq, n_layers):
    """Extract N frequency band layers from raw EEG."""
    bands = BAND_CONFIGS[n_layers]
    layers = np.zeros((n_layers, len(eeg_data)), dtype=np.float64)
    for i, (low, high, name) in enumerate(bands):
        layers[i] = bandpass(eeg_data, low, high, sfreq)
    return layers

def extract_windows(layers, sfreq, window_sec=5.0, step_sec=2.5):
    """Sliding window over bandpass-filtered layers."""
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    n_samples = layers.shape[1]
    windows = []
    start = 0
    while start + window_samples <= n_samples:
        windows.append(layers[:, start:start + window_samples])
        start += step_samples
    return windows

# ============================================================
# PHI FUNCTIONS (self-contained, works for any N)
# ============================================================
def compute_R(layers):
    n_layers = layers.shape[0]
    if n_layers < 2: return 0.0
    powers = np.array([np.var(layers[i]) for i in range(n_layers)])
    total = np.sum(powers)
    if total == 0: return 0.0
    p = powers / total
    H = -np.sum(p * np.log(p + 1e-15))
    eff = float(np.exp(H))
    if eff < 2.0: return 0.0
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
    if eff < 2.0: return 0.0
    coverage = eff / n_layers
    H_max = np.log(n_layers)
    evenness = H / H_max if H_max > 0 else 0.0
    n_sub = 8
    sub_size = window_size // n_sub
    if sub_size < 2:
        pattern_diversity = 0.5
    else:
        dominant = []
        for k in range(n_sub):
            s0 = k * sub_size
            s1 = s0 + sub_size
            sub_pow = np.array([np.var(layers[i, s0:s1]) for i in range(n_layers)])
            dominant.append(int(np.argmax(sub_pow)))
        pattern_diversity = len(set(dominant)) / n_sub
    return float(np.clip(coverage * (0.5 * evenness + 0.5 * pattern_diversity), 0.0, 1.0))

def compute_D(layers):
    from scipy.spatial.distance import jensenshannon
    n_layers = layers.shape[0]
    if n_layers < 2: return 0.0
    powers = np.array([np.var(layers[i]) for i in range(n_layers)])
    total = np.sum(powers)
    if total == 0: return 0.0
    p = powers / total
    q = np.ones(n_layers) / n_layers
    return float(jensenshannon(p, q, base=2))

def compute_epoch_id(epoch_data, sfreq, n_layers):
    """Compute I and D for one EEG epoch at given N."""
    layers = extract_layers_n(epoch_data, sfreq, n_layers)
    windows = extract_windows(layers, sfreq)
    Is, Ds, Sums = [], [], []
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        D = compute_D(w)
        Is.append(R * S)
        Ds.append(D)
        Sums.append(np.sum([np.var(w[i]) for i in range(w.shape[0])]))
    return float(np.mean(Is)), float(np.mean(Ds)), float(np.mean(Sums))

# ============================================================
# MAIN
# ============================================================
def main():
    flush("=" * 60)
    flush("  LAYER SENSITIVITY TEST — EEG (SLEEP-EDF)")
    flush("=" * 60)

    outdir = RESULTS
    _neural_dir = os.environ.get("PHI_NEURAL_DIR", os.path.expanduser("./external/phi-neural-test"))
    if not os.path.isdir(_neural_dir):
        sys.exit(f"ERROR: PHI_NEURAL_DIR not found: {_neural_dir}. Set PHI_NEURAL_DIR env var.")
    sys.path.insert(0, _neural_dir)
    from eeg_loader import load_dataset

    n_subjects = 5
    flush(f"\nLoading Sleep-EDF data ({n_subjects} subjects)...")
    epochs, sfreq, metadata = load_dataset(n_subjects=n_subjects)
    flush(f"  {metadata['n_epochs']} epochs, sfreq={sfreq} Hz")

    N_VALUES = [3, 4, 5, 6, 8]
    all_results = {}

    for N in N_VALUES:
        flush(f"\n>>> EEG N={N} ({len(BAND_CONFIGS[N])} bands) <<<")
        flush(f"  Bands: {[(b[0],b[1],b[2]) for b in BAND_CONFIGS[N]]}")

        all_I, all_D, all_sum = [], [], []
        stages = []
        for i, ep in enumerate(epochs):
            if i % 200 == 0:
                flush(f"  Processing epoch {i}/{len(epochs)}...")
            I, D, S = compute_epoch_id(ep['data'], sfreq, N)
            all_I.append(I)
            all_D.append(D)
            all_sum.append(S)
            stages.append(ep['stage'])

        all_I = np.array(all_I)
        all_D = np.array(all_D)
        all_sum = np.array(all_sum)
        stages = np.array(stages)

        # Wake vs N3
        w_mask = stages == 'W'
        n3_mask = stages == 'N3'
        primary = w_mask | n3_mask
        I_p = all_I[primary]
        D_p = all_D[primary]
        sum_p = all_sum[primary]
        y_p = w_mask[primary].astype(int)

        results = {'n_layers': N, 'n_wake': int(w_mask.sum()), 'n_n3': int(n3_mask.sum())}

        if y_p.sum() > 0 and (y_p == 0).sum() > 0:
            results['auc_i_rs'] = float(roc_auc_score(y_p, I_p))
            results['auc_sum'] = float(roc_auc_score(y_p, sum_p))

            # Balance zone occupancy
            I_z = (I_p - np.mean(I_p)) / (np.std(I_p) + 1e-10)
            D_z = (D_p - np.mean(D_p)) / (np.std(D_p) + 1e-10)
            balance = I_z - D_z
            abs_balance = np.abs(balance)

            for thresh in [0.5, 1.0, 1.5]:
                in_bal = abs_balance < thresh
                wake_in = in_bal[y_p == 1].mean()
                n3_in = in_bal[y_p == 0].mean()
                results[f'balance_{thresh}_wake'] = float(wake_in)
                results[f'balance_{thresh}_n3'] = float(n3_in)
                results[f'balance_{thresh}_ratio'] = float(wake_in / (n3_in + 1e-10))

            # Exit velocity
            exit_vel = np.abs(np.diff(balance))
            y_ev = y_p[1:]
            if y_ev.sum() > 0 and (y_ev == 0).sum() > 0:
                results['auc_exit_velocity'] = float(roc_auc_score(y_ev, exit_vel))
                wake_ev = exit_vel[y_ev == 1]
                n3_ev = exit_vel[y_ev == 0]
                u, p = mannwhitneyu(wake_ev, n3_ev, alternative='two-sided')
                results['ev_p'] = float(p)
                results['ev_ratio'] = float(np.median(wake_ev) / (np.median(n3_ev) + 1e-10))

            # Mean I and D by stage
            results['wake_mean_I'] = float(np.mean(all_I[w_mask]))
            results['wake_mean_D'] = float(np.mean(all_D[w_mask]))
            results['n3_mean_I'] = float(np.mean(all_I[n3_mask]))
            results['n3_mean_D'] = float(np.mean(all_D[n3_mask]))

        all_results[f'eeg_N{N}'] = results
        flush(f"  AUC(I): {results.get('auc_i_rs', '?')}, AUC(Sum): {results.get('auc_sum', '?')}, AUC(EV): {results.get('auc_exit_velocity', '?')}")

    # ---- SAVE ----
    outfile = os.path.join(outdir, 'layer_sensitivity_eeg.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    flush(f"\nSaved: {outfile}")

    # ---- PLOT ----
    flush("\nGenerating EEG sensitivity plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: AUC vs N
    ax = axes[0]
    ns = []
    auc_i = []
    auc_sum = []
    auc_ev = []
    for N in N_VALUES:
        key = f'eeg_N{N}'
        if key in all_results:
            ns.append(N)
            auc_i.append(all_results[key].get('auc_i_rs', 0.5))
            auc_sum.append(all_results[key].get('auc_sum', 0.5))
            auc_ev.append(all_results[key].get('auc_exit_velocity', 0.5))
    ax.plot(ns, auc_i, 'o-', color='steelblue', linewidth=2, markersize=8, label='I (R×S)')
    ax.plot(ns, auc_sum, 's--', color='coral', linewidth=2, markersize=8, label='Sum')
    ax.plot(ns, auc_ev, '^:', color='green', linewidth=2, markersize=8, label='Exit Velocity')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Layers (N)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('EEG (Wake vs N3): AUC vs Layer Count', fontsize=13)
    ax.set_xticks(N_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)
    ax.axvspan(4.5, 5.5, alpha=0.1, color='blue')

    # Panel 2: Balance zone occupancy vs N
    ax = axes[1]
    for thresh in [0.5, 1.0, 1.5]:
        ratios = [all_results.get(f'eeg_N{N}', {}).get(f'balance_{thresh}_ratio', 0) for N in N_VALUES]
        ax.plot(N_VALUES, ratios, 'o-', label=f'±{thresh}σ', linewidth=2, markersize=7)
    ax.set_xlabel('Number of Layers (N)', fontsize=12)
    ax.set_ylabel('Wake/N3 Balance Ratio', fontsize=12)
    ax.set_title('Balance Zone Occupancy Ratio vs N', fontsize=13)
    ax.set_xticks(N_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Panel 3: Summary table
    ax = axes[2]
    ax.axis('off')
    table_data = []
    for N in N_VALUES:
        r = all_results.get(f'eeg_N{N}', {})
        table_data.append([
            str(N),
            f"{r.get('auc_i_rs', 0):.3f}",
            f"{r.get('auc_sum', 0):.3f}",
            f"{r.get('auc_exit_velocity', 0):.3f}",
            f"{r.get('balance_0.5_ratio', 0):.1f}x",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=['N', 'I AUC', 'Sum AUC', 'EV AUC', '±0.5σ ratio'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax.set_title('EEG Layer Sensitivity Summary', fontsize=13, pad=20)

    plt.tight_layout()
    outplot = os.path.join(outdir, 'layer_sensitivity_eeg.png')
    plt.savefig(outplot, dpi=200, bbox_inches='tight')
    plt.close()
    flush(f"Saved: {outplot}")

    # ---- PRINT SUMMARY ----
    flush("\n" + "=" * 60)
    flush("  EEG LAYER SENSITIVITY SUMMARY")
    flush("=" * 60)
    flush(f"\n  {'N':>3} | {'AUC(I)':>8} | {'AUC(Sum)':>8} | {'AUC(EV)':>8} | {'±0.5σ ratio':>11} | {'EV ratio':>8}")
    flush(f"  {'---':>3} | {'--------':>8} | {'--------':>8} | {'--------':>8} | {'-----------':>11} | {'--------':>8}")
    for N in N_VALUES:
        r = all_results.get(f'eeg_N{N}', {})
        flush(f"  {N:>3} | {r.get('auc_i_rs', 0):>8.4f} | {r.get('auc_sum', 0):>8.4f} | {r.get('auc_exit_velocity', 0):>8.4f} | {r.get('balance_0.5_ratio', 0):>10.1f}x | {r.get('ev_ratio', 0):>8.2f}")

    flush("\n  ROBUSTNESS CHECK:")
    auc_5 = all_results.get('eeg_N5', {}).get('auc_i_rs', 0.5)
    for N in [4, 6]:
        auc_n = all_results.get(f'eeg_N{N}', {}).get('auc_i_rs', 0.5)
        pct_diff = abs(auc_n - auc_5) / auc_5 * 100
        status = "PASS" if pct_diff < 10 else "NEEDS DISCUSSION"
        flush(f"    N={N} vs N=5: {pct_diff:.1f}% difference — {status}")

    flush("\n  KEY RESULT: Sum AUC should anti-predict (< 0.5) at ALL layer counts")
    for N in N_VALUES:
        sum_auc = all_results.get(f'eeg_N{N}', {}).get('auc_sum', 0.5)
        status = "ANTI-PREDICTS" if sum_auc < 0.5 else "WARNING: Sum works at N={N}"
        flush(f"    N={N}: Sum AUC={sum_auc:.3f} — {status}")

    flush("\n" + "=" * 60)
    flush("  DONE")
    flush("=" * 60)

if __name__ == '__main__':
    main()
