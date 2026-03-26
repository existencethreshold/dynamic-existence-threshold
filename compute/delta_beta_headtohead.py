#!/usr/bin/env python3
"""
Run 2: Delta/Beta Ratio vs R×S Head-to-Head
Comparison of delta/beta power ratio with R×S metric for Wake vs N3 classification.

Computes delta/beta power ratio for all subjects and compares AUC with R×S
for Wake vs N3 classification.
"""
import numpy as np
import os, sys, json, time
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu, spearmanr
from scipy.signal import butter, sosfilt, welch
from scipy.spatial.distance import jensenshannon
from eeg_loader_direct import load_dataset

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')

N_WORKERS = max(1, cpu_count() - 2)

# ============================================================
# PHI FUNCTIONS (R×S)
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

# ============================================================
# COMBINED METRIC COMPUTATION (per epoch)
# ============================================================
def compute_epoch_all(args):
    """Compute R×S and all spectral ratios for one epoch. Multiprocessing worker."""
    data, sfreq = args

    # R×S
    windows = extract_layers(data, sfreq)
    Is = []
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        Is.append(R * S)
    rxs = float(np.mean(Is))

    # Spectral ratios via Welch
    freqs, psd = welch(data, fs=sfreq, nperseg=min(len(data), int(4 * sfreq)))

    def band_power(lo, hi):
        mask = (freqs >= lo) & (freqs <= hi)
        return np.trapezoid(psd[mask], freqs[mask])

    delta = band_power(0.5, 4.0)
    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 13.0)
    beta = band_power(13.0, 30.0)
    gamma = band_power(30.0, 45.0)
    total = delta + theta + alpha + beta + gamma + 1e-20

    return {
        'rxs': rxs,
        'delta_beta': delta / (beta + 1e-20),
        'delta_alpha': delta / (alpha + 1e-20),
        'delta_rel': delta / total,
        'theta_beta': theta / (beta + 1e-20),
        'alpha_rel': alpha / total,
        'total_power': total,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    N_SUBJECTS = 50

    print("=" * 60)
    print("  HEAD-TO-HEAD: Delta/Beta Ratio vs R×S")
    print(f"  {N_WORKERS} workers")
    print("=" * 60)

    epochs = load_dataset(N_SUBJECTS)

    print(f"\nComputing metrics for {len(epochs)} epochs ({N_WORKERS} workers)...", flush=True)
    args_list = [(ep['data'], ep['sfreq']) for ep in epochs]
    with Pool(N_WORKERS) as pool:
        all_results = pool.map(compute_epoch_all, args_list, chunksize=100)

    all_I = np.array([r['rxs'] for r in all_results])
    all_db = np.array([r['delta_beta'] for r in all_results])
    all_stages = np.array([ep['stage'] for ep in epochs])
    ratio_keys = ['delta_beta', 'delta_alpha', 'delta_rel', 'theta_beta', 'alpha_rel', 'total_power']

    elapsed_compute = time.time() - t_start
    print(f"  Computed in {elapsed_compute/60:.1f} min", flush=True)

    # --- Primary: Wake vs N3 ---
    w_mask = all_stages == 'W'
    n3_mask = all_stages == 'N3'
    primary = w_mask | n3_mask
    y = w_mask[primary].astype(int)

    I_p = all_I[primary]
    db_p = all_db[primary]

    auc_rxs = roc_auc_score(y, I_p)
    auc_db = roc_auc_score(y, -db_p)

    # All spectral ratios
    ratio_aucs = {}
    for key in ratio_keys:
        vals = np.array([r[key] for r in all_results])[primary]
        if key in ('delta_beta', 'delta_alpha', 'delta_rel', 'theta_beta'):
            auc = roc_auc_score(y, -vals)
        else:
            auc = roc_auc_score(y, vals)
        ratio_aucs[key] = float(auc)

    print("\n" + "=" * 60)
    print("  WAKE vs N3 CLASSIFICATION (AUC)")
    print("=" * 60)
    print(f"  R×S (our framework):  {auc_rxs:.4f}")
    print(f"  Delta/Beta ratio:     {auc_db:.4f}")
    for key, auc in sorted(ratio_aucs.items(), key=lambda x: -x[1]):
        marker = " <<<" if auc > auc_rxs else ""
        print(f"  {key:20s}:  {auc:.4f}{marker}")

    winner = "R×S WINS" if auc_rxs > auc_db else "DELTA/BETA WINS"
    margin = abs(auc_rxs - auc_db)
    print(f"\n  {winner} by {margin:.4f}")

    # Correlation
    corr, pval = spearmanr(I_p, -db_p)
    print(f"\n  Spearman(R×S, -δ/β): {corr:.4f} (p={pval:.2e})")
    if corr > 0.9:
        print("  WARNING: R×S may be a proxy for delta/beta ratio")
    elif corr > 0.7:
        print("  NOTE: Moderate correlation — R×S captures spectral info + more")
    else:
        print("  GOOD: R×S captures different information than delta/beta")

    # Per-stage
    print("\n  Per-stage mean values:")
    for stage in ['W', 'N1', 'N2', 'N3', 'R']:
        mask = all_stages == stage
        if mask.sum() == 0: continue
        print(f"    {stage}: R×S={np.mean(all_I[mask]):.4f}, δ/β={np.mean(all_db[mask]):.1f} (n={mask.sum()})")

    # Adjacent stage discrimination
    print("\n  Adjacent stage AUCs:")
    stage_pairs = [('W', 'N1'), ('N1', 'N2'), ('N2', 'N3'), ('W', 'N2'), ('W', 'R')]
    adjacent_results = {}
    for s1, s2 in stage_pairs:
        m1 = all_stages == s1
        m2 = all_stages == s2
        if m1.sum() < 10 or m2.sum() < 10:
            continue
        mask = m1 | m2
        y_adj = m1[mask].astype(int)
        I_adj = all_I[mask]
        db_adj = all_db[mask]
        try:
            a_rxs = roc_auc_score(y_adj, I_adj)
            a_db = roc_auc_score(y_adj, -db_adj)
            w = "R×S" if a_rxs > a_db else "δ/β"
            print(f"    {s1} vs {s2}: R×S={a_rxs:.4f}, δ/β={a_db:.4f}  [{w}]")
            adjacent_results[f'{s1}_vs_{s2}'] = {'rxs': float(a_rxs), 'db': float(a_db)}
        except Exception as e:
            print(f"    {s1} vs {s2}: skipped ({e})")

    # Bootstrap CI
    print("\n  Bootstrap 95% CI (1000 resamples)...")
    rng = np.random.RandomState(42)
    boot_rxs, boot_db, boot_diff = [], [], []
    for _ in range(1000):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if y[idx].sum() > 0 and (y[idx] == 0).sum() > 0:
            a_r = roc_auc_score(y[idx], I_p[idx])
            a_d = roc_auc_score(y[idx], -db_p[idx])
            boot_rxs.append(a_r)
            boot_db.append(a_d)
            boot_diff.append(a_r - a_d)

    boot_rxs = np.array(boot_rxs)
    boot_db = np.array(boot_db)
    boot_diff = np.array(boot_diff)

    print(f"    R×S AUC:   [{np.percentile(boot_rxs, 2.5):.4f}, {np.percentile(boot_rxs, 97.5):.4f}]")
    print(f"    δ/β AUC:   [{np.percentile(boot_db, 2.5):.4f}, {np.percentile(boot_db, 97.5):.4f}]")
    print(f"    Difference: [{np.percentile(boot_diff, 2.5):.4f}, {np.percentile(boot_diff, 97.5):.4f}]")

    sig = "YES" if np.percentile(boot_diff, 2.5) > 0 or np.percentile(boot_diff, 97.5) < 0 else "NO"
    print(f"    Significant difference (CI excludes 0): {sig}")

    # --- SAVE ---
    results = {
        'n_subjects': N_SUBJECTS,
        'n_epochs': len(epochs),
        'primary_wake_vs_n3': {
            'n_wake': int(w_mask.sum()),
            'n_n3': int(n3_mask.sum()),
            'auc_rxs': float(auc_rxs),
            'auc_delta_beta': float(auc_db),
            'winner': winner,
            'margin': float(margin),
            'correlation_rxs_db': float(corr),
            'correlation_pval': float(pval),
        },
        'all_spectral_aucs': ratio_aucs,
        'adjacent_stages': adjacent_results,
        'bootstrap': {
            'rxs_ci': [float(np.percentile(boot_rxs, 2.5)), float(np.percentile(boot_rxs, 97.5))],
            'db_ci': [float(np.percentile(boot_db, 2.5)), float(np.percentile(boot_db, 97.5))],
            'diff_ci': [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
            'significant': sig == "YES",
        }
    }

    with open(os.path.join(RESULTS, 'delta_beta_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.join(RESULTS, "delta_beta_results.json")}")

    # --- PLOT ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: ROC curves
    ax = axes[0, 0]
    fpr_r, tpr_r, _ = roc_curve(y, I_p)
    fpr_d, tpr_d, _ = roc_curve(y, -db_p)
    ax.plot(fpr_r, tpr_r, 'b-', linewidth=2, label=f'R×S (AUC={auc_rxs:.3f})')
    ax.plot(fpr_d, tpr_d, 'r--', linewidth=2, label=f'δ/β (AUC={auc_db:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('Wake vs N3: ROC Comparison')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    # Panel 2: R×S vs delta/beta scatter
    ax = axes[0, 1]
    ax.scatter(I_p[y == 1], -db_p[y == 1], c='green', alpha=0.1, s=3, label='Wake')
    ax.scatter(I_p[y == 0], -db_p[y == 0], c='red', alpha=0.3, s=3, label='N3')
    ax.set_xlabel('R×S'); ax.set_ylabel('-δ/β')
    ax.set_title(f'R×S vs -δ/β (ρ={corr:.3f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: Bootstrap difference
    ax = axes[0, 2]
    ax.hist(boot_diff, bins=50, color='steelblue', alpha=0.7)
    ax.axvline(0, color='red', linewidth=2, linestyle='--', label='No difference')
    ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
    ax.axvline(ci_lo, color='orange', linestyle='--')
    ax.axvline(ci_hi, color='orange', linestyle='--')
    ax.set_xlabel('AUC(R×S) - AUC(δ/β)'); ax.set_ylabel('Count')
    ax.set_title(f'Bootstrap: R×S - δ/β = {np.mean(boot_diff):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4: All metrics bar chart
    ax = axes[1, 0]
    names = ['R×S'] + list(ratio_aucs.keys())
    vals = [float(auc_rxs)] + [ratio_aucs[k] for k in ratio_aucs]
    colors = ['steelblue'] + ['coral'] * len(ratio_aucs)
    bars = ax.bar(names, vals, color=colors, alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('AUC'); ax.set_title('All Metrics: Wake vs N3')
    ax.set_ylim(0.4, 1.0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: Adjacent stage comparison
    ax = axes[1, 1]
    if adjacent_results:
        pairs = list(adjacent_results.keys())
        rxs_vals = [adjacent_results[p]['rxs'] for p in pairs]
        db_vals = [adjacent_results[p]['db'] for p in pairs]
        x = np.arange(len(pairs))
        ax.bar(x - 0.15, rxs_vals, 0.3, label='R×S', color='steelblue', alpha=0.8)
        ax.bar(x + 0.15, db_vals, 0.3, label='δ/β', color='coral', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([p.replace('_vs_', ' vs ') for p in pairs])
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('AUC'); ax.set_title('Adjacent Stage Discrimination')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: Per-stage R×S distribution
    ax = axes[1, 2]
    stage_order = ['W', 'R', 'N1', 'N2', 'N3']
    stage_colors = {'W': 'green', 'R': 'lime', 'N1': 'gold', 'N2': 'orange', 'N3': 'red'}
    for si, stage in enumerate(stage_order):
        mask = all_stages == stage
        if mask.sum() == 0: continue
        bp = ax.boxplot([all_I[mask]], positions=[si], widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black'))
        bp['boxes'][0].set_facecolor(stage_colors[stage])
        bp['boxes'][0].set_alpha(0.6)
    ax.set_xticks(range(len(stage_order))); ax.set_xticklabels(stage_order)
    ax.set_ylabel('R×S'); ax.set_title('R×S Distribution by Sleep Stage')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'delta_beta_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(RESULTS, "delta_beta_plot.png")}")

    elapsed = time.time() - t_start
    print(f"\nTOTAL TIME: {elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
