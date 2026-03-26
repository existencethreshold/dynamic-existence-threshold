#!/usr/bin/env python3
"""
Phase 7b Test 5: EEG Sleep Stage Classification — I-D Balance Residence

Applies the I-D balance framework to EEG data (Sleep-EDF):
- Wake (conscious) should occupy the I-D balance zone
- N3 deep sleep (unconscious) should exit the balance zone
- Exit velocity should discriminate conscious vs unconscious states

Uses the five-layer DET measurement architecture:
  eeg_loader.py — Sleep-EDF data loader
  layer_mapping.py — EEG → 5 frequency band layers
  phi_functions.py — R, S, D computation
"""
import numpy as np
import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')


def flush(msg):
    print(msg, flush=True)

# Import from phi-neural-test
_neural_dir = os.environ.get("PHI_NEURAL_DIR", os.path.expanduser("./external/phi-neural-test"))
    if not os.path.isdir(_neural_dir):
        sys.exit(f"ERROR: PHI_NEURAL_DIR not found: {_neural_dir}. Set PHI_NEURAL_DIR env var.")
    sys.path.insert(0, _neural_dir)
from eeg_loader import load_dataset
from layer_mapping import extract_layers_windowed
from phi_functions import compute_R, compute_S, compute_D

def compute_epoch_id(epoch_data, sfreq):
    """
    Compute I (integration) and D (differentiation) for one EEG epoch.
    
    Uses windowed approach: extract 5-second windows with 2.5s overlap,
    compute R, S, D for each, then aggregate.
    Returns: mean_I (R*S), mean_D, mean_sum, all_I, all_D
    """
    windows, times = extract_layers_windowed(epoch_data, sfreq, window_sec=5.0, step_sec=2.5)
    
    Is = []
    Ds = []
    Sums = []
    
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        D = compute_D(w)
        Is.append(R * S)
        Ds.append(D)
        Sums.append(np.sum([np.var(w[i]) for i in range(w.shape[0])]))
    
    return {
        'mean_I': float(np.mean(Is)),
        'mean_D': float(np.mean(Ds)),
        'mean_sum': float(np.mean(Sums)),
        'all_I': Is,
        'all_D': Ds,
    }

def main():
    flush("=" * 60)
    flush("  PHASE 7b TEST 5: EEG I-D BALANCE RESIDENCE")
    flush("=" * 60)
    
    outdir = RESULTS
    
    # Load EEG data — 5 subjects for decent sample
    flush("\nLoading Sleep-EDF data (5 subjects)...")
    epochs, sfreq, metadata = load_dataset(n_subjects=5)
    flush(f"  {metadata['n_epochs']} epochs, sfreq={sfreq} Hz")
    flush(f"  Conscious: {metadata['conscious_epochs']}, Unconscious: {metadata['unconscious_epochs']}")
    
    # Compute I and D for each epoch
    flush("\nComputing I-D measures for each epoch...")
    epoch_results = []
    for i, ep in enumerate(epochs):
        if i % 100 == 0:
            flush(f"  Processing epoch {i}/{len(epochs)}...")
        result = compute_epoch_id(ep['data'], sfreq)
        result['stage'] = ep['stage']
        result['conscious'] = ep['conscious']
        epoch_results.append(result)
    
    # Extract arrays
    all_I = np.array([r['mean_I'] for r in epoch_results])
    all_D = np.array([r['mean_D'] for r in epoch_results])
    all_sum = np.array([r['mean_sum'] for r in epoch_results])
    stages = np.array([r['stage'] for r in epoch_results])
    conscious = np.array([r['conscious'] for r in epoch_results])
    
    # Primary test: Wake (W) vs N3
    w_mask = stages == 'W'
    n3_mask = stages == 'N3'
    primary_mask = w_mask | n3_mask
    
    flush(f"\n  Wake epochs: {w_mask.sum()}, N3 epochs: {n3_mask.sum()}")
    
    # Z-score I and D (on primary subset)
    I_primary = all_I[primary_mask]
    D_primary = all_D[primary_mask]
    y_primary = w_mask[primary_mask].astype(int)  # 1 = Wake (conscious), 0 = N3 (unconscious)
    
    I_z = (I_primary - np.mean(I_primary)) / (np.std(I_primary) + 1e-10)
    D_z = (D_primary - np.mean(D_primary)) / (np.std(D_primary) + 1e-10)
    
    balance = I_z - D_z
    abs_balance = np.abs(balance)
    
    results = {}
    
    # ---- Test 5a: Do conscious states live in the balance zone? ----
    flush("\n[Test 5a: Balance Zone Occupancy]")
    for thresh in [0.5, 1.0, 1.5]:
        in_balance = abs_balance < thresh
        wake_in = in_balance[y_primary == 1].mean()
        n3_in = in_balance[y_primary == 0].mean()
        flush(f"  ±{thresh}σ: Wake in balance: {wake_in:.3f}, N3 in balance: {n3_in:.3f}, ratio: {wake_in/(n3_in+1e-10):.2f}")
        results[f'balance_zone_{thresh}'] = {
            'wake_pct_in_balance': float(wake_in),
            'n3_pct_in_balance': float(n3_in),
            'ratio': float(wake_in / (n3_in + 1e-10))
        }
    
    # ---- Test 5b: I and D values by stage ----
    flush("\n[Test 5b: I and D by Stage]")
    for stage in ['W', 'N1', 'N2', 'N3', 'R']:
        mask = stages == stage
        if mask.sum() == 0:
            continue
        mean_i = float(np.mean(all_I[mask]))
        mean_d = float(np.mean(all_D[mask]))
        flush(f"  {stage}: I={mean_i:.4f}, D={mean_d:.4f}, I/D={mean_i/(mean_d+1e-10):.3f} (n={mask.sum()})")
        results[f'stage_{stage}'] = {
            'mean_I': mean_i,
            'mean_D': mean_d,
            'n': int(mask.sum())
        }
    
    # ---- Test 5c: Exit velocity (temporal transitions) ----
    flush("\n[Test 5c: Exit Velocity]")
    # Only meaningful for consecutive epochs of the same type
    # Use the balance signal and compute velocity
    exit_velocity = np.abs(np.diff(balance))
    y_ev = y_primary[1:]
    
    if y_ev.sum() > 0 and (y_ev == 0).sum() > 0:
        wake_ev = exit_velocity[y_ev == 1]
        n3_ev = exit_velocity[y_ev == 0]
        u, p = mannwhitneyu(wake_ev, n3_ev, alternative='two-sided')
        flush(f"  Wake exit velocity: {np.median(wake_ev):.4f}")
        flush(f"  N3 exit velocity:   {np.median(n3_ev):.4f}")
        flush(f"  Ratio: {np.median(wake_ev)/(np.median(n3_ev)+1e-10):.2f}")
        flush(f"  Mann-Whitney p: {p}")
        results['exit_velocity'] = {
            'wake_median': float(np.median(wake_ev)),
            'n3_median': float(np.median(n3_ev)),
            'ratio': float(np.median(wake_ev) / (np.median(n3_ev) + 1e-10)),
            'mannwhitney_p': float(p)
        }
    
    # ---- Test 5d: AUC classifiers ----
    flush("\n[Test 5d: Classifier AUC (Wake vs N3)]")
    if y_primary.sum() > 0 and (y_primary == 0).sum() > 0:
        # I as classifier
        auc_i = roc_auc_score(y_primary, I_primary)
        flush(f"  I (R×S) AUC: {auc_i:.4f}")
        
        # D as classifier (inverted — conscious should have moderate D)
        auc_d = roc_auc_score(y_primary, D_primary)
        flush(f"  D AUC: {auc_d:.4f}")
        
        # |I-D| as classifier (inverted — conscious should be near balance)
        auc_abs = roc_auc_score(y_primary, -abs_balance)  # negative because lower distance = conscious
        flush(f"  |I-D| (balance distance) AUC: {auc_abs:.4f}")
        
        # Sum as classifier
        sum_primary = all_sum[primary_mask]
        auc_sum = roc_auc_score(y_primary, sum_primary)
        flush(f"  Sum AUC: {auc_sum:.4f}")
        
        # Exit velocity AUC
        if len(exit_velocity) == len(y_ev):
            auc_ev = roc_auc_score(y_ev, exit_velocity)
            flush(f"  Exit velocity AUC: {auc_ev:.4f}")
            results['auc_exit_velocity'] = float(auc_ev)
        
        results['auc_I'] = float(auc_i)
        results['auc_D'] = float(auc_d)
        results['auc_abs_balance'] = float(auc_abs)
        results['auc_sum'] = float(auc_sum)
    
    # ---- Test 5e: The I-D Phase Space (same as financial/space) ----
    flush("\n[Test 5e: Phase Space]")
    # Full stage analysis
    for stage in ['W', 'N1', 'N2', 'N3', 'R']:
        mask = stages == stage
        if mask.sum() < 10:
            continue
        I_s = all_I[mask]
        D_s = all_D[mask]
        # Where do they cluster relative to the I=D line?
        I_z_s = (I_s - np.mean(all_I)) / (np.std(all_I) + 1e-10)
        D_z_s = (D_s - np.mean(all_D)) / (np.std(all_D) + 1e-10)
        balance_s = I_z_s - D_z_s
        flush(f"  {stage}: mean(I-D)={np.mean(balance_s):.3f}, std(I-D)={np.std(balance_s):.3f}, median|I-D|={np.median(np.abs(balance_s)):.3f}")
    
    # ---- PLOT ----
    flush("\nGenerating EEG phase space plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: I vs D by stage
    ax = axes[0, 0]
    colors = {'W': 'green', 'R': 'lime', 'N1': 'gold', 'N2': 'orange', 'N3': 'red'}
    for stage in ['N3', 'N2', 'N1', 'R', 'W']:  # plot W last (on top)
        mask = stages == stage
        if mask.sum() == 0:
            continue
        ax.scatter(all_I[mask], all_D[mask], c=colors.get(stage, 'gray'), alpha=0.3, s=8, label=f'{stage} (n={mask.sum()})')
    ax.set_xlabel('Integration (R×S)', fontsize=12)
    ax.set_ylabel('Differentiation (D)', fontsize=12)
    ax.set_title('EEG I-D Phase Space by Sleep Stage', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Balance distance distribution by stage
    ax = axes[0, 1]
    I_all_z = (all_I - np.mean(all_I)) / (np.std(all_I) + 1e-10)
    D_all_z = (all_D - np.mean(all_D)) / (np.std(all_D) + 1e-10)
    bal_all = np.abs(I_all_z - D_all_z)
    for stage in ['W', 'R', 'N1', 'N2', 'N3']:
        mask = stages == stage
        if mask.sum() < 10:
            continue
        vals = bal_all[mask]
        ax.hist(vals, bins=50, alpha=0.4, color=colors.get(stage, 'gray'), label=f'{stage} (med={np.median(vals):.2f})', density=True)
    ax.set_xlabel('|I-D| (balance distance)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Balance Distance by Stage\n(Lower = more balanced)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: I-D balance over time for one recording
    ax = axes[1, 0]
    # Use first 500 epochs for visibility
    n_show = min(500, len(epoch_results))
    I_show_z = (all_I[:n_show] - np.mean(all_I)) / (np.std(all_I) + 1e-10)
    D_show_z = (all_D[:n_show] - np.mean(all_D)) / (np.std(all_D) + 1e-10)
    bal_show = I_show_z - D_show_z
    
    for stage, color in colors.items():
        mask = stages[:n_show] == stage
        if mask.sum() > 0:
            ax.scatter(np.where(mask)[0], bal_show[mask], c=color, s=4, alpha=0.6, label=stage)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.fill_between(range(n_show), -1.5, 1.5, alpha=0.05, color='green')
    ax.set_xlabel('Epoch index', fontsize=12)
    ax.set_ylabel('I-D Balance (z-scored)', fontsize=12)
    ax.set_title('I-D Balance Over Time (first 500 epochs)', fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: AUC comparison bar chart
    ax = axes[1, 1]
    if 'auc_I' in results:
        classifiers = ['auc_I', 'auc_D', 'auc_abs_balance', 'auc_sum']
        labels = ['I (R×S)', 'D', '|I-D| (balance)', 'Sum']
        if 'auc_exit_velocity' in results:
            classifiers.append('auc_exit_velocity')
            labels.append('Exit Vel')
        vals = [results.get(c, 0.5) for c in classifiers]
        bars = ax.bar(labels, vals, color=['steelblue', 'coral', 'green', 'gold', 'purple'][:len(vals)], alpha=0.8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Wake vs N3 Classifier Performance', fontsize=13)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    outfile = os.path.join(outdir, 'phase7b_eeg_results.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    flush(f"  Saved: {outfile}")
    
    # Save results
    outfile = os.path.join(outdir, 'phase7b_eeg_results.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    flush(f"  Saved: {outfile}")
    
    flush("\n" + "=" * 60)
    flush("  TEST 5 COMPLETE")
    flush("=" * 60)

if __name__ == '__main__':
    main()
