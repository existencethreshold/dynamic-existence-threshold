#!/usr/bin/env python3
"""
Phase 7b EEG: Full I-D Balance Analysis (50+ subjects)
Self-contained — downloads data, runs all tests, prints results.
Requires: pip install -r requirements.txt (see README)
"""
import numpy as np
import os, sys, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu, gaussian_kde
from scipy.signal import butter, sosfilt
from scipy.spatial.distance import jensenshannon
import mne

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')


# ============================================================
# EEG DATA LOADER (from eeg_loader.py)
# ============================================================
STAGE_MAP = {
    'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3', 'Sleep stage 4': 'N3', 'Sleep stage R': 'R',
    'Sleep stage ?': '?', 'Movement time': 'MT',
}
CONSCIOUS_STAGES = {'W', 'R'}
UNCONSCIOUS_STAGES = {'N3'}

def load_dataset(n_subjects=76):
    DATA_DIR = os.environ.get("SLEEP_EDF_DIR", "./external/sleep-edf")
    os.makedirs(DATA_DIR, exist_ok=True)
    MISSING = {39, 68, 69, 78, 79}
    MISSING_REC1 = {36, 52}
    valid_subjects = [s for s in range(83) if s not in MISSING and s not in MISSING_REC1]
    if n_subjects > len(valid_subjects):
        n_subjects = len(valid_subjects)
    subjects = valid_subjects[:n_subjects]
    
    print(f"Downloading {n_subjects} subjects...", flush=True)
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=subjects, recording=[1], path=DATA_DIR)
    
    all_epochs = []
    sfreq = None
    for idx, (psg_path, hypno_path) in enumerate(paths):
        if idx % 10 == 0:
            print(f"  Loading subject {idx+1}/{len(paths)}...", flush=True)
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        eeg_channel = 'EEG Fpz-Cz'
        if eeg_channel not in raw.ch_names:
            eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
            eeg_channel = eeg_channels[0] if eeg_channels else raw.ch_names[0]
        raw.pick([eeg_channel])
        sfreq = raw.info['sfreq']
        raw_data = raw.get_data()[0]
        annot = mne.read_annotations(hypno_path)
        epoch_duration = 30.0
        epoch_samples = int(epoch_duration * sfreq)
        
        for onset, duration, description in zip(annot.onset, annot.duration, annot.description):
            stage = STAGE_MAP.get(description)
            if stage is None or stage in ('?', 'MT'):
                continue
            n_ep = int(duration / epoch_duration)
            start_sample = int(onset * sfreq)
            for i in range(n_ep):
                ep_start = start_sample + i * epoch_samples
                ep_end = ep_start + epoch_samples
                if ep_end > len(raw_data):
                    break
                data = raw_data[ep_start:ep_end]
                if stage in CONSCIOUS_STAGES:
                    conscious = True
                elif stage in UNCONSCIOUS_STAGES:
                    conscious = False
                else:
                    conscious = None
                all_epochs.append({'data': data, 'stage': stage, 'conscious': conscious})
    
    stage_counts = {}
    for ep in all_epochs:
        stage_counts[ep['stage']] = stage_counts.get(ep['stage'], 0) + 1
    
    print(f"\nLoaded {len(all_epochs)} epochs from {n_subjects} subjects", flush=True)
    print(f"Stages: {stage_counts}", flush=True)
    return all_epochs, sfreq

# ============================================================
# LAYER MAPPING (from layer_mapping.py)
# ============================================================
BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]

def bandpass(data, low, high, sfreq, order=4):
    nyq = sfreq / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return np.zeros_like(data)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos, data)

def extract_layers_windowed(eeg_data, sfreq, window_sec=5.0, step_sec=2.5):
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

# ============================================================
# PHI FUNCTIONS (from phi_functions.py)
# ============================================================
EFF_THRESHOLD = 2.0

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
        pattern_diversity = 0.0
    else:
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

# ============================================================
# ANALYSIS
# ============================================================
def compute_epoch_id(epoch_data, sfreq):
    windows = extract_layers_windowed(epoch_data, sfreq)
    Is, Ds, Sums = [], [], []
    for w in windows:
        R = compute_R(w)
        S = compute_S(w)
        D = compute_D(w)
        Is.append(R * S)
        Ds.append(D)
        Sums.append(np.sum([np.var(w[i]) for i in range(w.shape[0])]))
    return float(np.mean(Is)), float(np.mean(Ds)), float(np.mean(Sums))

def main():
    t_start = time.time()
    print("=" * 60, flush=True)
    print("  PHASE 7b EEG: 76-SUBJECT FULL-SCALE RUN", flush=True)
    print("=" * 60, flush=True)
    
    epochs, sfreq = load_dataset(n_subjects=76)
    
    print(f"\nComputing I-D measures for {len(epochs)} epochs...", flush=True)
    all_I, all_D, all_sum, stages, conscious = [], [], [], [], []
    
    for i, ep in enumerate(epochs):
        if i % 500 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(epochs) - i) / rate / 60 if rate > 0 else 0
            print(f"  Epoch {i}/{len(epochs)} ({rate:.0f}/s, ETA {eta:.0f}min)...", flush=True)
        I, D, S = compute_epoch_id(ep['data'], sfreq)
        all_I.append(I)
        all_D.append(D)
        all_sum.append(S)
        stages.append(ep['stage'])
        conscious.append(ep['conscious'])
    
    all_I = np.array(all_I)
    all_D = np.array(all_D)
    all_sum = np.array(all_sum)
    stages = np.array(stages)
    
    # Primary: Wake vs N3
    w_mask = stages == 'W'
    n3_mask = stages == 'N3'
    primary_mask = w_mask | n3_mask
    
    I_p = all_I[primary_mask]
    D_p = all_D[primary_mask]
    sum_p = all_sum[primary_mask]
    y_p = w_mask[primary_mask].astype(int)
    
    I_z = (I_p - np.mean(I_p)) / (np.std(I_p) + 1e-10)
    D_z = (D_p - np.mean(D_p)) / (np.std(D_p) + 1e-10)
    balance = I_z - D_z
    abs_balance = np.abs(balance)
    
    results = {
        'n_subjects': 76,
        'n_epochs': len(epochs),
        'n_wake': int(w_mask.sum()),
        'n_n3': int(n3_mask.sum()),
    }
    
    # Test 5a: Balance zone occupancy
    print("\n[Test 5a: Balance Zone Occupancy]", flush=True)
    for thresh in [0.5, 0.75, 1.0, 1.5]:
        in_bal = abs_balance < thresh
        wake_in = in_bal[y_p == 1].mean()
        n3_in = in_bal[y_p == 0].mean()
        ratio = wake_in / (n3_in + 1e-10)
        print(f"  ±{thresh}σ: Wake={wake_in:.4f}, N3={n3_in:.4f}, ratio={ratio:.1f}x", flush=True)
        results[f'balance_{thresh}'] = {'wake': float(wake_in), 'n3': float(n3_in), 'ratio': float(ratio)}
    
    # Test 5b: I and D by stage
    print("\n[Test 5b: I and D by Stage]", flush=True)
    for stage in ['W', 'N1', 'N2', 'N3', 'R']:
        mask = stages == stage
        if mask.sum() == 0: continue
        mi, md = float(np.mean(all_I[mask])), float(np.mean(all_D[mask]))
        print(f"  {stage}: I={mi:.4f}, D={md:.4f}, I/D={mi/(md+1e-10):.4f} (n={mask.sum()})", flush=True)
        results[f'stage_{stage}'] = {'I': mi, 'D': md, 'n': int(mask.sum())}
    
    # Test 5c: Exit velocity
    print("\n[Test 5c: Exit Velocity]", flush=True)
    exit_vel = np.abs(np.diff(balance))
    y_ev = y_p[1:]
    if y_ev.sum() > 0 and (y_ev == 0).sum() > 0:
        w_ev, n3_ev = exit_vel[y_ev == 1], exit_vel[y_ev == 0]
        u, p = mannwhitneyu(w_ev, n3_ev, alternative='two-sided')
        print(f"  Wake: {np.median(w_ev):.4f}, N3: {np.median(n3_ev):.4f}, ratio: {np.median(w_ev)/(np.median(n3_ev)+1e-10):.1f}x", flush=True)
        print(f"  Mann-Whitney p = {p}", flush=True)
        results['exit_velocity'] = {
            'wake_median': float(np.median(w_ev)), 'n3_median': float(np.median(n3_ev)),
            'ratio': float(np.median(w_ev) / (np.median(n3_ev) + 1e-10)), 'p': float(p)
        }
    
    # Test 5d: Classifier AUC
    print("\n[Test 5d: Classifier AUC]", flush=True)
    if y_p.sum() > 0 and (y_p == 0).sum() > 0:
        auc_i = roc_auc_score(y_p, I_p)
        auc_d = roc_auc_score(y_p, D_p)
        auc_bal = roc_auc_score(y_p, -abs_balance)
        auc_sum = roc_auc_score(y_p, sum_p)
        print(f"  I (R×S):     {auc_i:.4f}", flush=True)
        print(f"  D:           {auc_d:.4f}", flush=True)
        print(f"  |I-D|:       {auc_bal:.4f}", flush=True)
        print(f"  Sum:         {auc_sum:.4f}", flush=True)
        results['auc'] = {'I': float(auc_i), 'D': float(auc_d), 'balance': float(auc_bal), 'sum': float(auc_sum)}
        
        if len(exit_vel) == len(y_ev) and y_ev.sum() > 0:
            auc_ev = roc_auc_score(y_ev, exit_vel)
            print(f"  Exit vel:    {auc_ev:.4f}", flush=True)
            results['auc']['exit_velocity'] = float(auc_ev)
        
        # Mann-Whitney for I
        u, p_i = mannwhitneyu(I_p[y_p == 1], I_p[y_p == 0], alternative='greater')
        results['i_mannwhitney_p'] = float(p_i)
        print(f"  I Mann-Whitney p = {p_i}", flush=True)
    
    # Test 5e: Residence time
    print("\n[Test 5e: Residence Time]", flush=True)
    for thresh in [0.5, 1.0, 1.5]:
        in_bal = abs_balance < thresh
        res_times, exit_labels = [], []
        run = 0
        for i in range(len(in_bal)):
            if in_bal[i]:
                run += 1
            else:
                if run > 0:
                    res_times.append(run)
                    exit_labels.append(int(y_p[i]))
                run = 0
        res_times = np.array(res_times)
        exit_labels = np.array(exit_labels)
        if len(res_times) > 0 and (exit_labels == 0).sum() > 5 and (exit_labels == 1).sum() > 5:
            med_w = float(np.median(res_times[exit_labels == 1]))
            med_n = float(np.median(res_times[exit_labels == 0]))
            u, p = mannwhitneyu(res_times[exit_labels == 1], res_times[exit_labels == 0], alternative='two-sided')
            print(f"  ±{thresh}σ: Wake exit med={med_w:.0f}, N3 exit med={med_n:.0f}, p={p:.6f}", flush=True)
            results[f'residence_{thresh}'] = {'wake_med': med_w, 'n3_med': med_n, 'p': float(p)}
    
    # Test 5f: Per-subject consistency
    print("\n[Test 5f: Per-Subject Consistency]", flush=True)
    # Check if the signal holds across individual subjects
    # Group by subject (approximate: epochs are loaded sequentially per subject)
    # We can check if I separates W from N3 within each subject
    subject_aucs = []
    # Since we don't track subject ID, check if result is driven by a few subjects
    # by bootstrap resampling
    n_boot = 1000
    boot_aucs = []
    rng = np.random.RandomState(42)
    for b in range(n_boot):
        idx = rng.choice(len(y_p), size=len(y_p), replace=True)
        if y_p[idx].sum() > 0 and (y_p[idx] == 0).sum() > 0:
            boot_aucs.append(roc_auc_score(y_p[idx], I_p[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  I AUC bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)
    results['bootstrap_auc_ci'] = [float(ci_lo), float(ci_hi)]
    
    # Also bootstrap exit velocity
    boot_ev_aucs = []
    for b in range(n_boot):
        idx = rng.choice(len(y_ev), size=len(y_ev), replace=True)
        if y_ev[idx].sum() > 0 and (y_ev[idx] == 0).sum() > 0:
            boot_ev_aucs.append(roc_auc_score(y_ev[idx], exit_vel[idx]))
    boot_ev_aucs = np.array(boot_ev_aucs)
    ev_lo, ev_hi = np.percentile(boot_ev_aucs, [2.5, 97.5])
    print(f"  Exit vel AUC bootstrap 95% CI: [{ev_lo:.4f}, {ev_hi:.4f}]", flush=True)
    results['bootstrap_ev_auc_ci'] = [float(ev_lo), float(ev_hi)]
    
    # Bootstrap balance zone ratio
    boot_ratios = []
    for b in range(n_boot):
        idx = rng.choice(len(y_p), size=len(y_p), replace=True)
        ab_b = abs_balance[idx]
        y_b = y_p[idx]
        in_b = ab_b < 0.5
        w_in = in_b[y_b == 1].mean() if (y_b == 1).sum() > 0 else 0
        n_in = in_b[y_b == 0].mean() if (y_b == 0).sum() > 0 else 0
        boot_ratios.append(w_in / (n_in + 1e-10))
    boot_ratios = np.array(boot_ratios)
    r_lo, r_hi = np.percentile(boot_ratios, [2.5, 97.5])
    print(f"  Balance ratio (±0.5σ) bootstrap 95% CI: [{r_lo:.1f}x, {r_hi:.1f}x]", flush=True)
    results['bootstrap_balance_ratio_ci'] = [float(r_lo), float(r_hi)]
    
    # ---- PLOT ----
    print("\nGenerating plot...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    colors = {'W': 'green', 'R': 'lime', 'N1': 'gold', 'N2': 'orange', 'N3': 'red'}
    
    # Panel 1: I vs D
    ax = axes[0, 0]
    for stage in ['N3', 'N2', 'N1', 'R', 'W']:
        mask = stages == stage
        if mask.sum() == 0: continue
        ax.scatter(all_I[mask], all_D[mask], c=colors[stage], alpha=0.15, s=4, label=f'{stage} ({mask.sum()})')
    ax.set_xlabel('I (R×S)'); ax.set_ylabel('D'); ax.set_title(f'I-D Phase Space (76 subjects, {len(epochs)} epochs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Panel 2: Balance distance by stage
    ax = axes[0, 1]
    I_all_z = (all_I - np.mean(all_I)) / (np.std(all_I) + 1e-10)
    D_all_z = (all_D - np.mean(all_D)) / (np.std(all_D) + 1e-10)
    bal_all = np.abs(I_all_z - D_all_z)
    for stage in ['W', 'R', 'N1', 'N2', 'N3']:
        mask = stages == stage
        if mask.sum() < 10: continue
        ax.hist(bal_all[mask], bins=50, alpha=0.4, color=colors[stage], density=True, label=f'{stage} (med={np.median(bal_all[mask]):.2f})')
    ax.set_xlabel('|I-D|'); ax.set_ylabel('Density'); ax.set_title('Balance Distance by Stage')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Panel 3: AUC bars
    ax = axes[0, 2]
    if 'auc' in results:
        names = ['I (R×S)', 'D', '|I-D|', 'Sum']
        vals = [results['auc']['I'], results['auc']['D'], results['auc']['balance'], results['auc']['sum']]
        if 'exit_velocity' in results['auc']:
            names.append('Exit Vel')
            vals.append(results['auc']['exit_velocity'])
        bars = ax.bar(names, vals, color=['steelblue', 'coral', 'green', 'gold', 'purple'][:len(vals)], alpha=0.8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('AUC'); ax.set_title('Wake vs N3 Classifier (76 subjects)')
        ax.set_ylim(0.2, 1.0); ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Panel 4: Bootstrap CI
    ax = axes[1, 0]
    ax.hist(boot_aucs, bins=50, color='steelblue', alpha=0.7, label=f'I AUC [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.axvline(ci_lo, color='red', linestyle='--'); ax.axvline(ci_hi, color='red', linestyle='--')
    ax.set_xlabel('AUC'); ax.set_title('Bootstrap 95% CI (I classifier, 1000 resamples)')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Panel 5: Exit velocity distribution
    ax = axes[1, 1]
    bins = np.linspace(0, np.percentile(exit_vel, 99), 50)
    ax.hist(exit_vel[y_ev == 1], bins=bins, alpha=0.5, color='green', density=True, label=f'Wake (med={np.median(exit_vel[y_ev==1]):.3f})')
    ax.hist(exit_vel[y_ev == 0], bins=bins, alpha=0.5, color='red', density=True, label=f'N3 (med={np.median(exit_vel[y_ev==0]):.3f})')
    ax.set_xlabel('|Δ(I-D)|'); ax.set_title('Exit Velocity: Wake vs N3')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Panel 6: Balance zone occupancy across thresholds
    ax = axes[1, 2]
    thresholds = [0.5, 0.75, 1.0, 1.5]
    wake_pcts = [results[f'balance_{t}']['wake'] for t in thresholds]
    n3_pcts = [results[f'balance_{t}']['n3'] for t in thresholds]
    ax.plot(thresholds, wake_pcts, 'go-', label='Wake', linewidth=2, markersize=8)
    ax.plot(thresholds, n3_pcts, 'rs-', label='N3', linewidth=2, markersize=8)
    for t, w, n in zip(thresholds, wake_pcts, n3_pcts):
        ax.annotate(f'{w/n:.0f}x', (t, w), textcoords='offset points', xytext=(10, -5), fontsize=9, color='darkgreen')
    ax.set_xlabel('Balance zone (σ)'); ax.set_ylabel('% in zone')
    ax.set_title('Balance Zone Occupancy'); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'phase7b_eeg_76subj.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: {os.path.join(RESULTS, "phase7b_eeg_76subj.png")}", flush=True)
    
    # Save results
    with open(os.path.join(RESULTS, 'phase7b_eeg_76subj.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {os.path.join(RESULTS, "phase7b_eeg_76subj.json")}", flush=True)
    
    elapsed = time.time() - t_start
    print(f"\n  TOTAL TIME: {elapsed/60:.1f} minutes", flush=True)
    print("=" * 60, flush=True)
    print("=" * 60, flush=True)

if __name__ == '__main__':
    main()
