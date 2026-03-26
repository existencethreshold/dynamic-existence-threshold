#!/usr/bin/env python3
"""
Phase 7b: I-D Balance Residence — Quantitative Tests
1. Fix residence time (±1σ + KDE-based boundary)
2. Exit velocity as standalone classifier (ROC/AUC vs Sum baseline)
3. Pre-event variance early warning (lead time sweep)
4. Cross-domain universality (same threshold across financial + space)
"""
import numpy as np
import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import gaussian_kde, mannwhitneyu

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')


def flush(msg):
    print(msg, flush=True)

def compute_rsd_timeseries(layers_df, phi_mod, eff_threshold=2.0, n_layers=5):
    data = layers_df[['L0','L1','L2','L3','L4']].values.astype(float)
    n = len(data)
    Rs = np.zeros(n)
    Ss = np.zeros(n)
    Ds = np.zeros(n)
    for i in range(n):
        bp = data[i]
        swd = np.argmax(data[max(0,i-8):i], axis=1) if i >= 8 else None
        Rs[i] = phi_mod.compute_R(bp, n_layers=n_layers, eff_threshold=eff_threshold)
        Ss[i] = phi_mod.compute_S(bp, sub_window_dominants=swd, n_layers=n_layers, eff_threshold=eff_threshold)
        Ds[i] = phi_mod.compute_D(bp, n_layers=n_layers)
    return Rs, Ss, Ds

def get_id_measures(I_vals, D_vals):
    """Compute z-scored I-D balance and derived measures."""
    I_z = (I_vals - np.mean(I_vals)) / (np.std(I_vals) + 1e-10)
    D_z = (D_vals - np.mean(D_vals)) / (np.std(D_vals) + 1e-10)
    balance = I_z - D_z
    exit_velocity = np.abs(np.diff(balance))
    abs_balance = np.abs(balance)
    return I_z, D_z, balance, exit_velocity, abs_balance

# ============================================================
# TEST 1: Residence time with wider thresholds + KDE
# ============================================================
def test_residence(I_vals, D_vals, y, label):
    """Test residence times at multiple thresholds + KDE-based zone."""
    I_z, D_z, balance, _, abs_balance = get_id_measures(I_vals, D_vals)

    results = {}

    # Multiple fixed thresholds
    for thresh in [0.5, 0.75, 1.0, 1.5]:
        in_balance = abs_balance < thresh
        res_times = []
        exit_labels = []
        run = 0
        for i in range(len(in_balance)):
            if in_balance[i]:
                run += 1
            else:
                if run > 0:
                    res_times.append(run)
                    exit_labels.append(int(y[i]) if i < len(y) else 0)
                run = 0
        res_times = np.array(res_times)
        exit_labels = np.array(exit_labels)

        stats = {'threshold': thresh, 'n_runs': len(res_times)}
        if len(res_times) > 0:
            stats['median_all'] = float(np.median(res_times))
            if (exit_labels == 0).sum() > 0:
                stats['median_normal_exit'] = float(np.median(res_times[exit_labels == 0]))
            if (exit_labels == 1).sum() > 0:
                stats['median_event_exit'] = float(np.median(res_times[exit_labels == 1]))
            # Mann-Whitney U test: do event exits have different residence?
            if (exit_labels == 0).sum() > 5 and (exit_labels == 1).sum() > 5:
                u, p = mannwhitneyu(res_times[exit_labels == 1], res_times[exit_labels == 0], alternative='two-sided')
                stats['mannwhitney_p'] = float(p)
        results[f'thresh_{thresh}'] = stats

    # KDE-based zone: define balance = region where normal-day density > median density
    normal_balance = abs_balance[y == 0]
    try:
        kde = gaussian_kde(normal_balance)
        x_grid = np.linspace(0, np.max(abs_balance), 500)
        density = kde(x_grid)
        # Balance zone = where density > 50th percentile of density curve
        normal_densities = kde(normal_balance)
        density_thresh = np.median(normal_densities)
        # Find the abs_balance cutoff where density drops below threshold
        # This gives us a data-driven boundary
        per_point_density = kde(abs_balance)
        in_balance_kde = per_point_density > density_thresh

        res_times = []
        exit_labels = []
        run = 0
        for i in range(len(in_balance_kde)):
            if in_balance_kde[i]:
                run += 1
            else:
                if run > 0:
                    res_times.append(run)
                    exit_labels.append(int(y[i]) if i < len(y) else 0)
                run = 0
        res_times = np.array(res_times)
        exit_labels = np.array(exit_labels)

        kde_stats = {'method': 'kde_median_density', 'n_runs': len(res_times)}
        if len(res_times) > 0:
            kde_stats['median_all'] = float(np.median(res_times))
            kde_stats['pct_in_zone'] = float(in_balance_kde.mean())
            if (exit_labels == 0).sum() > 0:
                kde_stats['median_normal_exit'] = float(np.median(res_times[exit_labels == 0]))
            if (exit_labels == 1).sum() > 0:
                kde_stats['median_event_exit'] = float(np.median(res_times[exit_labels == 1]))
            if (exit_labels == 0).sum() > 5 and (exit_labels == 1).sum() > 5:
                u, p = mannwhitneyu(res_times[exit_labels == 1], res_times[exit_labels == 0], alternative='two-sided')
                kde_stats['mannwhitney_p'] = float(p)
        results['kde'] = kde_stats
    except Exception as e:
        results['kde'] = {'error': str(e)}

    return results

# ============================================================
# TEST 2: Exit velocity as classifier (ROC/AUC)
# ============================================================
def test_exit_velocity_classifier(I_vals, D_vals, y, label):
    """Test exit velocity as a standalone event classifier."""
    I_z, D_z, balance, exit_velocity, abs_balance = get_id_measures(I_vals, D_vals)
    y_ev = y[1:]  # align with diff

    # Sum baseline: raw layer sum
    layer_sum = I_vals  # I_vals is already Sum for sum variants; for R×S, this tests R×S as classifier
    
    results = {}
    
    # Exit velocity AUC
    if y_ev.sum() > 0 and (y_ev == 0).sum() > 0:
        auc_ev = roc_auc_score(y_ev, exit_velocity)
        results['exit_velocity_auc'] = float(auc_ev)
        
        # |I-D| as classifier (distance from balance)
        auc_abs = roc_auc_score(y, abs_balance)
        results['abs_balance_auc'] = float(auc_abs)

        # Sum/I baseline AUC
        auc_sum = roc_auc_score(y, I_vals)
        results['i_proxy_auc'] = float(auc_sum)

        # Rolling variance as classifier
        for window in [15, 30, 60]:
            roll_var = np.array([np.var(abs_balance[max(0,i-window):i]) for i in range(window, len(abs_balance))])
            y_rv = y[window:]
            if y_rv.sum() > 0 and (y_rv == 0).sum() > 0:
                auc_rv = roc_auc_score(y_rv, roll_var)
                results[f'rolling_var_{window}d_auc'] = float(auc_rv)

        # Composite: exit velocity + rolling variance (30d)
        window = 30
        roll_var_30 = np.array([np.var(abs_balance[max(0,i-window):i]) for i in range(window, len(abs_balance))])
        ev_aligned = exit_velocity[window-1:]
        y_comp = y[window:]
        min_len = min(len(ev_aligned), len(roll_var_30), len(y_comp))
        ev_a = ev_aligned[:min_len]
        rv_a = roll_var_30[:min_len]
        y_c = y_comp[:min_len]
        # z-score both and average
        ev_z = (ev_a - ev_a.mean()) / (ev_a.std() + 1e-10)
        rv_z = (rv_a - rv_a.mean()) / (rv_a.std() + 1e-10)
        composite = ev_z + rv_z
        if y_c.sum() > 0 and (y_c == 0).sum() > 0:
            auc_comp = roc_auc_score(y_c, composite)
            results['composite_ev_var30_auc'] = float(auc_comp)

        # Mann-Whitney for exit velocity
        u, p = mannwhitneyu(exit_velocity[y_ev == 1], exit_velocity[y_ev == 0], alternative='greater')
        results['exit_velocity_mannwhitney_p'] = float(p)

    return results

# ============================================================
# TEST 3: Pre-event variance early warning (lead time sweep)
# ============================================================
def test_early_warning(I_vals, D_vals, y, label):
    """Does rising variance predict events N days ahead?"""
    _, _, balance, _, abs_balance = get_id_measures(I_vals, D_vals)

    results = {}
    event_indices = np.where(y == 1)[0]

    for window in [15, 30]:
        roll_var = np.array([np.var(abs_balance[max(0,i-window):i]) for i in range(window, len(abs_balance))])

        for lead in [5, 10, 15, 30]:
            # For each lead time: check variance N days BEFORE event
            pre_event_vars = []
            baseline_vars = []
            for ev_idx in event_indices:
                check_idx = ev_idx - lead - window  # index into roll_var
                if 0 <= check_idx < len(roll_var):
                    pre_event_vars.append(roll_var[check_idx])

            # Baseline: random sample of non-event days
            normal_indices = np.where(y[window:] == 0)[0]
            if len(normal_indices) > 0:
                baseline_vars = roll_var[normal_indices]

            if len(pre_event_vars) > 5 and len(baseline_vars) > 5:
                pre_arr = np.array(pre_event_vars)
                ratio = float(np.median(pre_arr) / (np.median(baseline_vars) + 1e-10))
                u, p = mannwhitneyu(pre_arr, baseline_vars, alternative='greater')
                results[f'w{window}_lead{lead}'] = {
                    'median_pre_event': float(np.median(pre_arr)),
                    'median_baseline': float(np.median(baseline_vars)),
                    'ratio': ratio,
                    'mannwhitney_p': float(p),
                    'n_events_tested': len(pre_event_vars)
                }

    return results

# ============================================================
# TEST 4: Cross-domain universality
# ============================================================
def test_universality(fin_results, space_results, fin_ev, space_ev, fin_y, space_y):
    """Check if same exit velocity threshold works across domains."""
    results = {}
    
    # Find optimal threshold for each domain
    for domain, ev, y_ev in [('financial', fin_ev, fin_y), ('space', space_ev, space_y)]:
        if y_ev.sum() == 0:
            continue
        y_ev_aligned = y_ev[1:len(ev)+1]
        fpr, tpr, thresholds = roc_curve(y_ev_aligned, ev)
        # Youden's J = optimal balance point
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        results[f'{domain}_optimal_threshold'] = float(thresholds[opt_idx])
        results[f'{domain}_optimal_tpr'] = float(tpr[opt_idx])
        results[f'{domain}_optimal_fpr'] = float(fpr[opt_idx])
    
    # Cross-apply: use financial threshold on space and vice versa
    if 'financial_optimal_threshold' in results and 'space_optimal_threshold' in results:
        fin_thresh = results['financial_optimal_threshold']
        space_thresh = results['space_optimal_threshold']
        
        # Financial threshold → space data
        space_pred = (space_ev >= fin_thresh).astype(int)
        space_y_al = space_y[1:]
        if len(space_pred) == len(space_y_al) and space_y_al.sum() > 0:
            tp = ((space_pred == 1) & (space_y_al == 1)).sum()
            fp = ((space_pred == 1) & (space_y_al == 0)).sum()
            fn = ((space_pred == 0) & (space_y_al == 1)).sum()
            tn = ((space_pred == 0) & (space_y_al == 0)).sum()
            results['fin_thresh_on_space'] = {
                'threshold': float(fin_thresh),
                'tpr': float(tp / (tp + fn + 1e-10)),
                'fpr': float(fp / (fp + tn + 1e-10)),
                'precision': float(tp / (tp + fp + 1e-10)),
                'f1': float(2*tp / (2*tp + fp + fn + 1e-10))
            }
        
        # Space threshold → financial data
        fin_pred = (fin_ev >= space_thresh).astype(int)
        fin_y_al = fin_y[1:]
        if len(fin_pred) == len(fin_y_al) and fin_y_al.sum() > 0:
            tp = ((fin_pred == 1) & (fin_y_al == 1)).sum()
            fp = ((fin_pred == 1) & (fin_y_al == 0)).sum()
            fn = ((fin_pred == 0) & (fin_y_al == 1)).sum()
            tn = ((fin_pred == 0) & (fin_y_al == 0)).sum()
            results['space_thresh_on_fin'] = {
                'threshold': float(space_thresh),
                'tpr': float(tp / (tp + fn + 1e-10)),
                'fpr': float(fp / (fp + tn + 1e-10)),
                'precision': float(tp / (tp + fp + 1e-10)),
                'f1': float(2*tp / (2*tp + fp + fn + 1e-10))
            }
        
        # Average threshold test
        avg_thresh = (fin_thresh + space_thresh) / 2
        results['avg_threshold'] = float(avg_thresh)
        results['threshold_ratio'] = float(max(fin_thresh, space_thresh) / (min(fin_thresh, space_thresh) + 1e-10))
    
    return results

# ============================================================
# PLOTTING
# ============================================================
def plot_phase7b(all_results, outdir):
    """Summary plots for Phase 7b."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Panel 1: Residence time vs threshold
    ax = axes[0, 0]
    for domain in ['financial_sum', 'space_sum']:
        if domain not in all_results['residence']:
            continue
        res = all_results['residence'][domain]
        thresholds = []
        medians_all = []
        medians_event = []
        medians_normal = []
        for key in sorted(res.keys()):
            if not key.startswith('thresh_'):
                continue
            t = res[key]
            thresholds.append(t['threshold'])
            medians_all.append(t.get('median_all', 0))
            medians_event.append(t.get('median_event_exit', 0))
            medians_normal.append(t.get('median_normal_exit', 0))
        label = domain.replace('_', ' ').title()
        ax.plot(thresholds, medians_normal, 'o-', label=f'{label} normal exit', alpha=0.8)
        ax.plot(thresholds, medians_event, 's--', label=f'{label} event exit', alpha=0.8)
    ax.set_xlabel('Balance zone threshold (σ)')
    ax.set_ylabel('Median residence (days)')
    ax.set_title('Test 1: Residence Time vs Zone Width')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Classifier AUC comparison
    ax = axes[0, 1]
    classifiers = ['exit_velocity_auc', 'abs_balance_auc', 'i_proxy_auc', 
                   'rolling_var_15d_auc', 'rolling_var_30d_auc', 'composite_ev_var30_auc']
    short_names = ['Exit Vel', '|I-D|', 'I proxy', 'Var 15d', 'Var 30d', 'EV+Var30']
    x_pos = np.arange(len(classifiers))
    width = 0.35
    for i, domain in enumerate(['financial_sum', 'space_sum']):
        if domain not in all_results['classifier']:
            continue
        vals = [all_results['classifier'][domain].get(c, 0) for c in classifiers]
        offset = -width/2 + i * width
        color = 'steelblue' if i == 0 else 'coral'
        label = 'Financial' if i == 0 else 'Space'
        ax.bar(x_pos + offset, vals, width, label=label, color=color, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.set_ylabel('AUC')
    ax.set_title('Test 2: Classifier Comparison (AUC)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.4, 1.0)
    
    # Panel 3: Early warning lead time
    ax = axes[1, 0]
    for domain in ['financial_sum', 'space_sum']:
        if domain not in all_results['early_warning']:
            continue
        ew = all_results['early_warning'][domain]
        leads = []
        ratios = []
        pvals = []
        for key in sorted(ew.keys()):
            if 'w30_lead' in key:
                leads.append(int(key.split('lead')[1]))
                ratios.append(ew[key]['ratio'])
                pvals.append(ew[key]['mannwhitney_p'])
        label = 'Financial' if 'financial' in domain else 'Space'
        color = 'steelblue' if 'financial' in domain else 'coral'
        ax.plot(leads, ratios, 'o-', color=color, label=f'{label} (30d window)', linewidth=2)
        # Mark significant ones
        for l, r, p in zip(leads, ratios, pvals):
            if p < 0.05:
                ax.annotate(f'p={p:.3f}', (l, r), textcoords='offset points',
                           xytext=(5, 5), fontsize=8, color=color)
    ax.set_xlabel('Lead time (days before event)')
    ax.set_ylabel('Variance ratio (pre-event / baseline)')
    ax.set_title('Test 3: Early Warning Signal')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No signal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Cross-domain universality
    ax = axes[1, 1]
    univ = all_results.get('universality', {})
    if univ:
        categories = []
        tprs = []
        fprs = []
        # Native performance
        for domain in ['financial', 'space']:
            t = univ.get(f'{domain}_optimal_tpr', 0)
            f = univ.get(f'{domain}_optimal_fpr', 0)
            categories.append(f'{domain.title()}\n(native)')
            tprs.append(t)
            fprs.append(f)
        # Cross-applied
        for key, label in [('fin_thresh_on_space', 'Fin→Space'), ('space_thresh_on_fin', 'Space→Fin')]:
            if key in univ:
                categories.append(label)
                tprs.append(univ[key]['tpr'])
                fprs.append(univ[key]['fpr'])
        
        x_pos = np.arange(len(categories))
        ax.bar(x_pos - 0.15, tprs, 0.3, label='TPR (sensitivity)', color='green', alpha=0.7)
        ax.bar(x_pos + 0.15, fprs, 0.3, label='FPR (false alarm)', color='red', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel('Rate')
        ax.set_title(f'Test 4: Cross-Domain Universality\nThreshold ratio: {univ.get("threshold_ratio", 0):.2f}x')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    outfile = os.path.join(outdir, 'phase7b_summary.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    flush(f"  Saved: {outfile}")

def main():
    flush("=" * 60)
    flush("  PHASE 7b: I-D BALANCE RESIDENCE — QUANTITATIVE TESTS")
    flush("=" * 60)

    outdir = RESULTS
    all_results = {'residence': {}, 'classifier': {}, 'early_warning': {}, 'universality': {}}

    # ---- FINANCIAL ----
    flush("\nLoading financial data...")
    _fin_dir = os.environ.get("PHI_FINANCIAL_DIR", os.path.expanduser("./external/phi-financial-test"))
    if not os.path.isdir(_fin_dir):
        sys.exit(f"ERROR: PHI_FINANCIAL_DIR not found: {_fin_dir}. Set PHI_FINANCIAL_DIR env var.")
    sys.path.insert(0, _fin_dir)
    os.chdir(_fin_dir)
    from data_loader import download_all
    from market_layers import compute_all_days as fin_compute
    from ground_truth import label_days as fin_label
    import phi_functions as fin_phi
    sector_returns, vix, haven_returns = download_all()
    fin_layers = fin_compute(sector_returns, vix, haven_returns, baseline_correct_l1=False)
    fin_y = fin_label(fin_layers.index).values

    Rs, Ss, Ds = compute_rsd_timeseries(fin_layers, fin_phi)
    fin_I_rs = Rs * Ss
    fin_I_sum = fin_layers[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    fin_D = Ds

    for i_label, I_vals in [('rs', fin_I_rs), ('sum', fin_I_sum)]:
        key = f'financial_{i_label}'
        flush(f"\n--- {key} ---")
        flush("  Test 1: Residence time...")
        all_results['residence'][key] = test_residence(I_vals, fin_D, fin_y, key)
        flush("  Test 2: Exit velocity classifier...")
        all_results['classifier'][key] = test_exit_velocity_classifier(I_vals, fin_D, fin_y, key)
        flush("  Test 3: Early warning...")
        all_results['early_warning'][key] = test_early_warning(I_vals, fin_D, fin_y, key)

    # ---- SPACE ----
    flush("\nLoading space weather data...")
    for m in list(sys.modules.keys()):
        if m in ('data_loader', 'ground_truth', 'phi_functions', 'market_layers', 'space_layers'):
            del sys.modules[m]
    _space_dir = os.environ.get("PHI_SPACE_DIR", os.path.expanduser("./external/phi-space-test"))
    if not os.path.isdir(_space_dir):
        sys.exit(f"ERROR: PHI_SPACE_DIR not found: {_space_dir}. Set PHI_SPACE_DIR env var.")
    sys.path.insert(0, _space_dir)
    os.chdir(_space_dir)
    from data_loader import load_data
    from space_layers import aggregate_hourly_to_daily, compute_all_days as space_compute
    from ground_truth import label_days_handlabeled
    import phi_functions as space_phi
    hourly = load_data()
    daily = aggregate_hourly_to_daily(hourly)
    space_layers = space_compute(daily)
    space_y = label_days_handlabeled(space_layers.index).values

    Rs, Ss, Ds = compute_rsd_timeseries(space_layers, space_phi)
    space_I_rs = Rs * Ss
    space_I_sum = space_layers[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    space_D = Ds

    for i_label, I_vals in [('rs', space_I_rs), ('sum', space_I_sum)]:
        key = f'space_{i_label}'
        flush(f"\n--- {key} ---")
        flush("  Test 1: Residence time...")
        all_results['residence'][key] = test_residence(I_vals, space_D, space_y, key)
        flush("  Test 2: Exit velocity classifier...")
        all_results['classifier'][key] = test_exit_velocity_classifier(I_vals, space_D, space_y, key)
        flush("  Test 3: Early warning...")
        all_results['early_warning'][key] = test_early_warning(I_vals, space_D, space_y, key)

    # ---- UNIVERSALITY (Test 4) — Sum variant ----
    flush("\n--- Test 4: Cross-domain universality ---")
    _, _, fin_bal, fin_ev, _ = get_id_measures(fin_I_sum, fin_D)
    _, _, space_bal, space_ev, _ = get_id_measures(space_I_sum, space_D)
    all_results['universality']['sum'] = test_universality(
        all_results['classifier']['financial_sum'],
        all_results['classifier']['space_sum'],
        fin_ev, space_ev, fin_y, space_y
    )
    # Also R×S
    _, _, fin_bal_rs, fin_ev_rs, _ = get_id_measures(fin_I_rs, fin_D)
    _, _, space_bal_rs, space_ev_rs, _ = get_id_measures(space_I_rs, space_D)
    all_results['universality']['rs'] = test_universality(
        all_results['classifier']['financial_rs'],
        all_results['classifier']['space_rs'],
        fin_ev_rs, space_ev_rs, fin_y, space_y
    )

    # ---- PLOTS ----
    flush("\nGenerating summary plot...")
    plot_phase7b(all_results, outdir)

    # ---- SAVE ----
    outfile = os.path.join(outdir, 'phase7b_results.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    flush(f"\nSaved: {outfile}")

    # ---- PRINT SUMMARY ----
    flush("\n" + "=" * 60)
    flush("  PHASE 7b SUMMARY")
    flush("=" * 60)
    
    flush("\n[Test 1: Residence Time]")
    for domain in ['financial_sum', 'space_sum']:
        r = all_results['residence'].get(domain, {})
        flush(f"  {domain}:")
        for t in ['thresh_0.5', 'thresh_1.0', 'thresh_1.5', 'kde']:
            if t in r:
                d = r[t]
                med_n = d.get('median_normal_exit', '?')
                med_e = d.get('median_event_exit', '?')
                p = d.get('mannwhitney_p', '?')
                flush(f"    {t}: normal={med_n}, event={med_e}, p={p}")

    flush("\n[Test 2: Classifier AUC]")
    for domain in ['financial_sum', 'space_sum', 'financial_rs', 'space_rs']:
        c = all_results['classifier'].get(domain, {})
        if c:
            flush(f"  {domain}:")
            flush(f"    Exit velocity: {c.get('exit_velocity_auc', '?'):.4f}")
            flush(f"    |I-D|:         {c.get('abs_balance_auc', '?'):.4f}")
            flush(f"    I proxy:       {c.get('i_proxy_auc', '?'):.4f}")
            flush(f"    Var 30d:       {c.get('rolling_var_30d_auc', '?'):.4f}")
            flush(f"    Composite:     {c.get('composite_ev_var30_auc', '?'):.4f}")
            flush(f"    EV p-value:    {c.get('exit_velocity_mannwhitney_p', '?')}")

    flush("\n[Test 3: Early Warning (30d window)]")
    for domain in ['financial_sum', 'space_sum']:
        ew = all_results['early_warning'].get(domain, {})
        flush(f"  {domain}:")
        for lead in [5, 10, 15, 30]:
            k = f'w30_lead{lead}'
            if k in ew:
                flush(f"    {lead}d ahead: ratio={ew[k]['ratio']:.2f}, p={ew[k]['mannwhitney_p']:.4f}")

    flush("\n[Test 4: Universality]")
    for variant in ['sum', 'rs']:
        u = all_results['universality'].get(variant, {})
        if u:
            flush(f"  {variant}:")
            flush(f"    Financial threshold: {u.get('financial_optimal_threshold', '?')}")
            flush(f"    Space threshold:     {u.get('space_optimal_threshold', '?')}")
            flush(f"    Threshold ratio:     {u.get('threshold_ratio', '?')}")
            for cross in ['fin_thresh_on_space', 'space_thresh_on_fin']:
                if cross in u:
                    flush(f"    {cross}: TPR={u[cross]['tpr']:.3f}, FPR={u[cross]['fpr']:.3f}, F1={u[cross]['f1']:.3f}")

    flush("\n" + "=" * 60)
    flush("  DONE")
    flush("=" * 60)

if __name__ == '__main__':
    main()
