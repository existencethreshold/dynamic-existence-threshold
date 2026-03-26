#!/usr/bin/env python3
"""
DET Out-of-Sample Validation
Financial: train pre-2015, test post-2015
Space: train pre-2010, test post-2010
Reports AUC on held-out test set only.
"""
import numpy as np
import sys, os, json, time
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')

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

def get_id_measures(I_vals, D_vals, train_mean_I=None, train_std_I=None,
                    train_mean_D=None, train_std_D=None):
    """z-score using train stats if provided, otherwise own stats."""
    mI = train_mean_I if train_mean_I is not None else np.mean(I_vals)
    sI = train_std_I if train_std_I is not None else np.std(I_vals)
    mD = train_mean_D if train_mean_D is not None else np.mean(D_vals)
    sD = train_std_D if train_std_D is not None else np.std(D_vals)
    I_z = (I_vals - mI) / (sI + 1e-10)
    D_z = (D_vals - mD) / (sD + 1e-10)
    balance = I_z - D_z
    exit_velocity = np.abs(np.diff(balance))
    abs_balance = np.abs(balance)
    return I_z, D_z, balance, exit_velocity, abs_balance

def oos_test(train_I, train_D, train_y, test_I, test_D, test_y, label):
    """Train on train set (learn z-score params + optimal threshold), evaluate on test set."""
    results = {'label': label}
    
    # Learn normalization from training data
    train_mI, train_sI = np.mean(train_I), np.std(train_I)
    train_mD, train_sD = np.mean(train_D), np.std(train_D)
    
    # Apply train normalization to BOTH sets
    _, _, train_bal, train_ev, train_abs = get_id_measures(
        train_I, train_D, train_mI, train_sI, train_mD, train_sD)
    _, _, test_bal, test_ev, test_abs = get_id_measures(
        test_I, test_D, train_mI, train_sI, train_mD, train_sD)
    
    results['train_n'] = len(train_I)
    results['test_n'] = len(test_I)
    results['train_events'] = int(train_y.sum())
    results['test_events'] = int(test_y.sum())
    
    # --- Exit velocity AUC ---
    # Train AUC
    if len(train_ev) == len(train_y) - 1:
        train_y_ev = train_y[1:]
    else:
        train_y_ev = train_y[:len(train_ev)]
    results['train_ev_auc'] = float(roc_auc_score(train_y_ev, train_ev))
    
    # Test AUC (THE KEY NUMBER)
    if len(test_ev) == len(test_y) - 1:
        test_y_ev = test_y[1:]
    else:
        test_y_ev = test_y[:len(test_ev)]
    results['test_ev_auc'] = float(roc_auc_score(test_y_ev, test_ev))
    
    # --- |I-D| balance AUC ---
    results['train_abs_auc'] = float(roc_auc_score(train_y, train_abs))
    results['test_abs_auc'] = float(roc_auc_score(test_y, test_abs))
    
    # --- Rolling 30d variance AUC ---
    for prefix, bal, y in [('train', train_bal, train_y), ('test', test_bal, test_y)]:
        abs_diff = np.abs(np.diff(bal, prepend=bal[0]))
        var30 = np.convolve(abs_diff**2, np.ones(30)/30, mode='same')
        results[f'{prefix}_var30_auc'] = float(roc_auc_score(y, var30))
    
    # --- I proxy (raw) AUC ---
    # Normalize test I using train stats
    test_I_z = (test_I - train_mI) / (train_sI + 1e-10)
    train_I_z = (train_I - train_mI) / (train_sI + 1e-10)
    results['train_i_auc'] = float(roc_auc_score(train_y, train_I_z))
    results['test_i_auc'] = float(roc_auc_score(test_y, test_I_z))
    
    # --- Mann-Whitney on test set ---
    ev_event = test_ev[test_y_ev == 1]
    ev_normal = test_ev[test_y_ev == 0]
    if len(ev_event) > 2 and len(ev_normal) > 2:
        u, p = mannwhitneyu(ev_event, ev_normal, alternative='greater')
        results['test_ev_mannwhitney_p'] = float(p)
    
    return results

def main():
    flush("=" * 60)
    flush("  OUT-OF-SAMPLE VALIDATION")
    flush("=" * 60)
    
    all_results = {}
    
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
    
    # Temporal split: pre-2015 / post-2015
    dates = fin_layers.index
    split_date = '2015-01-01'
    train_mask = dates < split_date
    test_mask = dates >= split_date
    
    flush(f"  Financial split: train {train_mask.sum()} days (pre-2015), test {test_mask.sum()} days (post-2015)")
    flush(f"  Train events: {fin_y[train_mask].sum()}, Test events: {fin_y[test_mask].sum()}")
    
    for variant, I_vals in [('rs', fin_I_rs), ('sum', fin_I_sum)]:
        key = f'financial_{variant}'
        flush(f"\n  --- {key} ---")
        r = oos_test(
            I_vals[train_mask], fin_D[train_mask], fin_y[train_mask],
            I_vals[test_mask], fin_D[test_mask], fin_y[test_mask],
            key
        )
        all_results[key] = r
        flush(f"    Train EV AUC: {r['train_ev_auc']:.4f}")
        flush(f"    TEST  EV AUC: {r['test_ev_auc']:.4f}  *** OUT-OF-SAMPLE ***")
        flush(f"    Train |I-D| AUC: {r['train_abs_auc']:.4f}")
        flush(f"    TEST  |I-D| AUC: {r['test_abs_auc']:.4f}")
        flush(f"    Train Var30 AUC: {r['train_var30_auc']:.4f}")
        flush(f"    TEST  Var30 AUC: {r['test_var30_auc']:.4f}")
    
    # ---- SPACE WEATHER ----
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
    
    # Temporal split: pre-2010 / post-2010
    dates = space_layers.index
    split_date = '2010-01-01'
    train_mask = dates < split_date
    test_mask = dates >= split_date
    
    flush(f"\n  Space split: train {train_mask.sum()} days (pre-2010), test {test_mask.sum()} days (post-2010)")
    flush(f"  Train events: {space_y[train_mask].sum()}, Test events: {space_y[test_mask].sum()}")
    
    for variant, I_vals in [('rs', space_I_rs), ('sum', space_I_sum)]:
        key = f'space_{variant}'
        flush(f"\n  --- {key} ---")
        r = oos_test(
            I_vals[train_mask], space_D[train_mask], space_y[train_mask],
            I_vals[test_mask], space_D[test_mask], space_y[test_mask],
            key
        )
        all_results[key] = r
        flush(f"    Train EV AUC: {r['train_ev_auc']:.4f}")
        flush(f"    TEST  EV AUC: {r['test_ev_auc']:.4f}  *** OUT-OF-SAMPLE ***")
        flush(f"    Train |I-D| AUC: {r['train_abs_auc']:.4f}")
        flush(f"    TEST  |I-D| AUC: {r['test_abs_auc']:.4f}")
        flush(f"    Train Var30 AUC: {r['train_var30_auc']:.4f}")
        flush(f"    TEST  Var30 AUC: {r['test_var30_auc']:.4f}")
    
    # ---- SUMMARY ----
    flush("\n" + "=" * 60)
    flush("  OUT-OF-SAMPLE SUMMARY")
    flush("=" * 60)
    flush(f"\n  {'Domain':<20} {'Train EV AUC':>14} {'Test EV AUC':>14} {'Δ':>8}")
    flush(f"  {'-'*56}")
    for key in ['financial_rs', 'financial_sum', 'space_rs', 'space_sum']:
        r = all_results[key]
        delta = r['test_ev_auc'] - r['train_ev_auc']
        flush(f"  {key:<20} {r['train_ev_auc']:>14.4f} {r['test_ev_auc']:>14.4f} {delta:>+8.4f}")
    
    flush(f"\n  {'Domain':<20} {'Train |I-D|':>14} {'Test |I-D|':>14} {'Δ':>8}")
    flush(f"  {'-'*56}")
    for key in ['financial_rs', 'financial_sum', 'space_rs', 'space_sum']:
        r = all_results[key]
        delta = r['test_abs_auc'] - r['train_abs_auc']
        flush(f"  {key:<20} {r['train_abs_auc']:>14.4f} {r['test_abs_auc']:>14.4f} {delta:>+8.4f}")
    
    # Save
    outfile = os.path.join(RESULTS, 'out_of_sample_validation.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    flush(f"\n  Saved: {outfile}")
    flush("\n  DONE")

if __name__ == '__main__':
    main()
