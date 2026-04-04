#!/usr/bin/env python3
"""
Phi Conservation Test (Test 6): Component Redistribution During Critical Transitions

Tests whether Phi = R*S + D is conserved during critical transitions across
financial markets, space weather, and EEG consciousness states. Measures
component redistribution (integration vs differentiation exchange) while
total organizational information remains approximately constant.
"""

import numpy as np
import sys, os, json, time
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')


# ================================================================
# PHI FUNCTIONS
# ================================================================
def _effective_bands(powers):
    total = np.sum(powers)
    if total == 0:
        return 1.0
    p = powers / total
    H = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(H))

def compute_R_scalar(band_powers, n_layers=5, eff_threshold=2.0):
    band_powers = np.asarray(band_powers, dtype=float)
    if n_layers < 2:
        return 0.0
    eff = _effective_bands(band_powers)
    if eff < eff_threshold:
        return 0.0
    return float(np.clip((eff - 1) / (n_layers - 1), 0.0, 1.0))

def compute_S_scalar(band_powers, sub_window_dominants=None, n_layers=5, eff_threshold=2.0):
    band_powers = np.asarray(band_powers, dtype=float)
    eff = _effective_bands(band_powers)
    if eff < eff_threshold:
        return 0.0
    total_power = np.sum(band_powers)
    if total_power == 0:
        return 0.0
    coverage = eff / n_layers
    p = band_powers / total_power
    H = -np.sum(p * np.log(p + 1e-15))
    H_max = np.log(n_layers)
    evenness = H / H_max if H_max > 0 else 0.0
    if sub_window_dominants is not None and len(sub_window_dominants) > 0:
        n_sub = len(sub_window_dominants)
        n_distinct = len(set(int(d) for d in sub_window_dominants))
        pattern_diversity = n_distinct / n_sub
    else:
        pattern_diversity = 0.5
    S = coverage * (0.5 * evenness + 0.5 * pattern_diversity)
    return float(np.clip(S, 0.0, 1.0))

def compute_D_scalar(band_powers, n_layers=5):
    band_powers = np.asarray(band_powers, dtype=float)
    if n_layers < 2:
        return 0.0
    total = np.sum(band_powers)
    if total == 0:
        return 0.0
    p = band_powers / total
    q = np.ones(n_layers) / n_layers
    return float(jensenshannon(p, q, base=2))

def compute_rsd_timeseries(layer_data):
    n = len(layer_data)
    Rs, Ss, Ds = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        bp = layer_data[i]
        swd = np.argmax(layer_data[max(0,i-8):i], axis=1) if i >= 8 else None
        Rs[i] = compute_R_scalar(bp)
        Ss[i] = compute_S_scalar(bp, sub_window_dominants=swd)
        Ds[i] = compute_D_scalar(bp)
    return Rs, Ss, Ds


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================
def analyze_domain(name, Rs, Ss, Ds, labels):
    """Compute mean component values and conservation statistics for a domain."""
    I_rs = Rs * Ss
    Phi = I_rs + Ds
    event_mask = labels == 1
    normal_mask = labels == 0

    results = {
        'domain': name,
        'n_total': int(len(labels)),
        'n_event': int(event_mask.sum()),
        'n_normal': int(normal_mask.sum()),
    }

    for metric_name, vals in [('R', Rs), ('S', Ss), ('RS', I_rs), ('D', Ds), ('Phi', Phi)]:
        nm = float(np.mean(vals[normal_mask]))
        em = float(np.mean(vals[event_mask]))
        dp = (1 - em / nm) * 100 if nm > 0 else 0
        stat, pval = mannwhitneyu(vals[event_mask], vals[normal_mask], alternative='two-sided')
        results[f'{metric_name}_normal_mean'] = nm
        results[f'{metric_name}_event_mean'] = em
        results[f'{metric_name}_drop_pct'] = dp
        results[f'{metric_name}_p'] = float(pval)

    # Peak cascade analysis per event cluster
    peak_drops = {'Phi': [], 'RS': [], 'D': [], 'R': [], 'S': []}
    in_event = False
    event_start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_event:
            in_event = True
            event_start = i
        elif labels[i] == 0 and in_event:
            in_event = False
            if i - event_start > 0:
                pre_start = max(0, event_start - 10)
                for metric, vals in [('Phi', Phi), ('RS', I_rs), ('D', Ds), ('R', Rs), ('S', Ss)]:
                    pre = np.mean(vals[pre_start:event_start]) if event_start > pre_start else np.mean(vals[normal_mask])
                    if metric in ('Phi', 'RS', 'R', 'S'):
                        min_idx = np.argmin(vals[event_start:i]) + event_start
                    else:
                        min_idx = np.argmax(vals[event_start:i]) + event_start
                    if pre > 0:
                        peak_drops[metric].append((1 - vals[min_idx] / pre) * 100)

    if peak_drops['Phi']:
        results['peak_n_events'] = len(peak_drops['Phi'])
        for metric in ['Phi', 'RS', 'D', 'R', 'S']:
            vals = peak_drops[metric]
            results[f'peak_{metric}_drop_mean'] = float(np.mean(vals))
            results[f'peak_{metric}_drop_std'] = float(np.std(vals))
            results[f'peak_{metric}_drop_median'] = float(np.median(vals))

    return results


def print_domain(r):
    print(f"\n{'='*70}")
    print(f"  {r['domain'].upper()}")
    print(f"  {r['n_total']} total | {r['n_event']} event | {r['n_normal']} normal")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<10} {'Normal':>10} {'Event':>10} {'Drop %':>10} {'p-value':>12}")
    for m in ['R', 'S', 'RS', 'D', 'Phi']:
        nm = r.get(f'{m}_normal_mean', 0)
        em = r.get(f'{m}_event_mean', 0)
        dp = r.get(f'{m}_drop_pct', 0)
        pv = r.get(f'{m}_p', 1)
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'NS'
        print(f"  {m:<10} {nm:>10.4f} {em:>10.4f} {dp:>+9.1f}% {pv:>10.2e} {sig}")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    t0 = time.time()
    all_results = {}

    # ============================================================
    # FINANCIAL MARKETS
    # ============================================================
    print("=" * 70)
    print("  LOADING FINANCIAL DATA")
    print("=" * 70, flush=True)

    fin_dir = os.environ.get('PHI_FINANCIAL_DIR', os.path.join(BASEDIR, 'external', 'phi-financial-test'))
    if not os.path.isdir(fin_dir):
        sys.exit(f"PHI_FINANCIAL_DIR not found: {fin_dir}\nSet PHI_FINANCIAL_DIR environment variable.")
    sys.path.insert(0, fin_dir)
    os.chdir(fin_dir)
    from data_loader import download_all
    from market_layers import compute_all_days as fin_compute
    from ground_truth import label_days as fin_label

    sector_returns, vix, haven_returns = download_all()
    fin_layers = fin_compute(sector_returns, vix, haven_returns, baseline_correct_l1=False)
    fin_y = fin_label(fin_layers.index).values
    layer_data = fin_layers[['L0','L1','L2','L3','L4']].values.astype(float)
    print(f"  {len(layer_data)} trading days loaded", flush=True)

    Rs, Ss, Ds = compute_rsd_timeseries(layer_data)
    fin_results = analyze_domain("Financial Markets", Rs, Ss, Ds, fin_y)
    print_domain(fin_results)
    all_results['financial'] = fin_results

    # ============================================================
    # SPACE WEATHER
    # ============================================================
    print(f"\n{'='*70}")
    print("  LOADING SPACE WEATHER DATA")
    print("=" * 70, flush=True)

    for m in list(sys.modules.keys()):
        if m in ('data_loader', 'ground_truth', 'market_layers', 'space_layers'):
            del sys.modules[m]

    space_dir = os.environ.get('PHI_SPACE_DIR', os.path.join(BASEDIR, 'external', 'phi-space-test'))
    if not os.path.isdir(space_dir):
        sys.exit(f"PHI_SPACE_DIR not found: {space_dir}\nSet PHI_SPACE_DIR environment variable.")
    sys.path.insert(0, space_dir)
    os.chdir(space_dir)
    from data_loader import load_data
    from space_layers import aggregate_hourly_to_daily, compute_all_days as space_compute
    from ground_truth import label_days_handlabeled

    hourly = load_data()
    daily = aggregate_hourly_to_daily(hourly)
    space_layers = space_compute(daily)
    space_y = label_days_handlabeled(space_layers.index).values
    layer_data = space_layers[['L0','L1','L2','L3','L4']].values.astype(float)
    print(f"  {len(layer_data)} days loaded", flush=True)

    Rs, Ss, Ds = compute_rsd_timeseries(layer_data)
    space_results = analyze_domain("Space Weather", Rs, Ss, Ds, space_y)
    print_domain(space_results)
    all_results['space'] = space_results

    # ============================================================
    # EEG — 50 SUBJECTS (ALL SLEEP STAGES)
    # ============================================================
    print(f"\n{'='*70}")
    print("  LOADING EEG DATA (50 SUBJECTS)")
    print("=" * 70, flush=True)

    for m in list(sys.modules.keys()):
        if m in ('data_loader', 'ground_truth', 'space_layers'):
            del sys.modules[m]

    neural_dir = os.environ.get('PHI_NEURAL_DIR', os.path.join(BASEDIR, 'external', 'phi-neural-test'))
    if not os.path.isdir(neural_dir):
        sys.exit(f"PHI_NEURAL_DIR not found: {neural_dir}\nSet PHI_NEURAL_DIR environment variable.")
    sys.path.insert(0, neural_dir)
    os.chdir(neural_dir)

    sleep_edf_dir = os.environ.get('SLEEP_EDF_DIR', os.path.join(BASEDIR, 'external', 'sleep-edf'))
    os.environ['SLEEP_EDF_DIR'] = sleep_edf_dir
    from eeg_loader_direct import load_dataset
    from layer_mapping import extract_layers_windowed
    from phi_functions import compute_R, compute_S, compute_D

    all_epochs = load_dataset(n_subjects=50)

    print(f"\nProcessing {len(all_epochs)} epochs...", flush=True)
    results_by_stage = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'R': []}
    processed = 0

    for i, ep in enumerate(all_epochs):
        stage = ep['stage']
        if stage not in results_by_stage:
            continue
        sfreq = ep['sfreq']
        windows, _ = extract_layers_windowed(ep['data'], sfreq)
        if not windows:
            continue

        Rs_ep, Ss_ep, Ds_ep = [], [], []
        for w in windows:
            Rs_ep.append(compute_R(w))
            Ss_ep.append(compute_S(w))
            Ds_ep.append(compute_D(w))

        R = np.mean(Rs_ep)
        S = np.mean(Ss_ep)
        RS = R * S
        D = np.mean(Ds_ep)
        Phi = RS + D
        results_by_stage[stage].append({'R': R, 'S': S, 'RS': RS, 'D': D, 'Phi': Phi})
        processed += 1
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(all_epochs)} epochs...", flush=True)

    print(f"\n  Processed {processed} epochs total", flush=True)
    for stage in ['W', 'R', 'N1', 'N2', 'N3']:
        print(f"    {stage}: {len(results_by_stage[stage])} epochs")

    # Compute stage means
    stage_means = {}
    print(f"\n{'='*70}")
    print(f"  EEG RESULTS — {processed} EPOCHS, 50 SUBJECTS")
    print(f"{'='*70}")
    print(f"\n  {'Stage':<6} {'N':>6} {'R':>8} {'S':>8} {'R·S':>8} {'D':>8} {'Phi':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for stage in ['W', 'R', 'N1', 'N2', 'N3']:
        data = results_by_stage[stage]
        if not data:
            continue
        R = np.mean([d['R'] for d in data])
        S = np.mean([d['S'] for d in data])
        RS = np.mean([d['RS'] for d in data])
        D = np.mean([d['D'] for d in data])
        Phi = np.mean([d['Phi'] for d in data])
        stage_means[stage] = {'R': R, 'S': S, 'RS': RS, 'D': D, 'Phi': Phi, 'n': len(data)}
        print(f"  {stage:<6} {len(data):>6} {R:>8.4f} {S:>8.4f} {RS:>8.4f} {D:>8.4f} {Phi:>8.4f}")

    # Wake vs N3 statistics
    if 'W' in stage_means and 'N3' in stage_means:
        print(f"\n  --- Wake vs N3 Component Changes ---")
        print(f"  {'Component':<12} {'Wake':>10} {'N3':>10} {'Change':>12}")
        w, n3 = stage_means['W'], stage_means['N3']
        for metric in ['R', 'S', 'RS', 'D', 'Phi']:
            wv, nv = w[metric], n3[metric]
            ch = (1 - nv / wv) * 100 if wv > 0 else float('nan')
            print(f"  {metric:<12} {wv:>10.4f} {nv:>10.4f} {ch:>+11.1f}%")

        print(f"\n  --- Statistical Tests ---")
        for metric in ['RS', 'D', 'Phi']:
            wv = np.array([d[metric] for d in results_by_stage['W']])
            nv = np.array([d[metric] for d in results_by_stage['N3']])
            stat, pval = mannwhitneyu(wv, nv, alternative='two-sided')
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'NS'
            print(f"  {metric:<12} p = {pval:.2e} {sig}")

    # Stage progression and conservation check
    print(f"\n  --- Phi Progression Through Sleep Stages ---")
    print(f"  {'Transition':<15} {'dRS':>10} {'dD':>10} {'dPhi':>10} {'Conserved?':>12}")
    for s1, s2 in [('W','N1'), ('N1','N2'), ('N2','N3'), ('W','R')]:
        if s1 in stage_means and s2 in stage_means:
            m1, m2 = stage_means[s1], stage_means[s2]
            dRS = ((m2['RS'] - m1['RS']) / max(m1['RS'], 1e-10)) * 100
            dD = ((m2['D'] - m1['D']) / max(m1['D'], 1e-10)) * 100
            dPhi = ((m2['Phi'] - m1['Phi']) / max(m1['Phi'], 1e-10)) * 100
            conserved = "YES" if abs(dPhi) < 15 else "NO"
            print(f"  {s1}->{s2:<11} {dRS:>+9.1f}% {dD:>+9.1f}% {dPhi:>+9.1f}% {conserved:>12}")

    all_results['eeg'] = {
        'stage_means': {k: v for k, v in stage_means.items()},
        'n_epochs_by_stage': {stage: len(results_by_stage[stage]) for stage in results_by_stage},
        'n_subjects': 50,
        'n_total_epochs': processed,
    }

    # ============================================================
    # CROSS-DOMAIN SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Domain/Transition':<25} {'dRS':>10} {'dD':>10} {'dPhi':>10} {'Direction':>15}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")

    f = all_results['financial']
    dRS = -f['RS_drop_pct']
    dD = -f['D_drop_pct']
    dPhi = -f['Phi_drop_pct']
    print(f"  {'Financial cascade':<25} {dRS:>+9.1f}% {dD:>+9.1f}% {dPhi:>+9.1f}% {'CASCADE':>15}")

    s = all_results['space']
    dRS = -s['RS_drop_pct']
    dD = -s['D_drop_pct']
    dPhi = -s['Phi_drop_pct']
    print(f"  {'Space weather storm':<25} {dRS:>+9.1f}% {dD:>+9.1f}% {dPhi:>+9.1f}% {'CASCADE':>15}")

    if 'W' in stage_means and 'N3' in stage_means:
        w, n3 = stage_means['W'], stage_means['N3']
        dRS = ((n3['RS'] - w['RS']) / max(w['RS'], 1e-10)) * 100
        dD = ((n3['D'] - w['D']) / max(w['D'], 1e-10)) * 100
        dPhi = ((n3['Phi'] - w['Phi']) / max(w['Phi'], 1e-10)) * 100
        print(f"  {'EEG Wake->N3':<25} {dRS:>+9.1f}% {dD:>+9.1f}% {dPhi:>+9.1f}% {'DISSOLVE':>15}")

    # Save results
    elapsed = time.time() - t0
    all_results['metadata'] = {
        'run_date': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'elapsed_seconds': round(elapsed),
        'python_version': sys.version,
    }

    outfile = os.path.join(RESULTS, 'phi_conservation_results.json')
    with open(outfile, 'w') as f_out:
        json.dump(all_results, f_out, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")
    print(f"  Total runtime: {elapsed/60:.1f} minutes")
