#!/usr/bin/env python3
"""
Phase 6d: Bootstrap CIs (10k iterations) + θ AUPRC sweep
for space weather and financial markets.
"""
import numpy as np
import sys, os, json

from sklearn.metrics import roc_auc_score, average_precision_score

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')



def bootstrap_ci(y_true, y_score, metric_fn, n_boot=10000, seed=42):
    """Bootstrap 95% CI (percentile method)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    real = metric_fn(y_true, y_score)
    boots = []
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boots.append(metric_fn(yt, ys))
        except:
            continue
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return real, lo, hi


def flush(msg):
    print(msg, flush=True)


def compute_rs_from_layers(layers_df, phi_mod, eff_threshold=2.0, n_layers=5):
    """Compute R*S scores from layer DataFrame."""
    data = layers_df[['L0','L1','L2','L3','L4']].values.astype(float)
    n = len(data)
    rs = np.zeros(n)
    for i in range(n):
        bp = data[i]
        swd = np.argmax(data[max(0,i-8):i], axis=1) if i >= 8 else None
        R = phi_mod.compute_R(bp, n_layers=n_layers, eff_threshold=eff_threshold)
        S = phi_mod.compute_S(bp, sub_window_dominants=swd, n_layers=n_layers, eff_threshold=eff_threshold)
        rs[i] = R * S
    return rs


def run_domain(name, layers_df, labels, phi_mod):
    flush(f"\n{'='*60}")
    flush(f"  {name}")
    flush(f"{'='*60}")
    
    results = {}
    y = np.array(labels)
    
    rs = compute_rs_from_layers(layers_df, phi_mod)
    sums = layers_df[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    
    # --- Bootstrap CIs ---
    flush(f"\n  Bootstrap CIs (10,000 iterations)...")
    for score_name, scores in [("RS", rs), ("Sum", sums)]:
        val, lo, hi = bootstrap_ci(y, scores, roc_auc_score)
        flush(f"    {score_name} AUC: {val:.4f} [{lo:.4f}, {hi:.4f}]")
        results[f'{score_name}_AUC'] = {'value': round(val,4), 'ci_lo': round(lo,4), 'ci_hi': round(hi,4)}
        
        val, lo, hi = bootstrap_ci(y, scores, average_precision_score)
        flush(f"    {score_name} AUPRC: {val:.4f} [{lo:.4f}, {hi:.4f}]")
        results[f'{score_name}_AUPRC'] = {'value': round(val,4), 'ci_lo': round(lo,4), 'ci_hi': round(hi,4)}
    
    # --- Layer ablation with CIs ---
    flush(f"\n  Ablation CIs (5,000 iterations per layer)...")
    full_auc = roc_auc_score(y, rs)
    abl = []
    for lname in ['L0','L1','L2','L3','L4']:
        ab = layers_df.copy()
        ab[lname] = 0.0
        rs_ab = compute_rs_from_layers(ab, phi_mod)
        val, lo, hi = bootstrap_ci(y, rs_ab, roc_auc_score, n_boot=5000)
        delta = val - full_auc
        flush(f"    Remove {lname}: AUC={val:.3f} [{lo:.3f},{hi:.3f}] delta={delta:+.3f}")
        abl.append({'layer': lname, 'auc': round(val,4), 'ci_lo': round(lo,4), 'ci_hi': round(hi,4), 'delta': round(delta,4)})
    results['ablation'] = abl
    
    # --- θ sweep with AUPRC ---
    flush(f"\n  θ sensitivity sweep...")
    flush(f"    {'θ':<6} {'AUC':<10} {'AUPRC':<10} {'Pos%':<10} {'Neg%':<10}")
    sweep = []
    for t in [1.5, 2.0, 2.5, 3.0, 3.5]:
        rs_t = compute_rs_from_layers(layers_df, phi_mod, eff_threshold=t)
        auc_t = roc_auc_score(y, rs_t) if np.std(rs_t) > 0 else 0.5
        auprc_t = average_precision_score(y, rs_t)
        g = (rs_t > 0).astype(int)
        pg = g[y==1].mean()*100
        ng = g[y==0].mean()*100
        flush(f"    {t:<6.1f} {auc_t:<10.3f} {auprc_t:<10.4f} {pg:<10.1f} {ng:<10.1f}")
        sweep.append({'theta': t, 'auc': round(auc_t,4), 'auprc': round(auprc_t,4), 'pos_gate': round(pg,1), 'neg_gate': round(ng,1)})
    results['theta_sweep'] = sweep
    
    return results


def main():
    flush("Phase 6d: Bootstrap CIs + θ AUPRC Sweep")
    flush("="*60)
    all_results = {}
    
    # --- FINANCIAL ---
    flush("\n--- FINANCIAL MARKETS ---")
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
    all_results['financial'] = run_domain("FINANCIAL MARKETS", fin_layers, fin_y, fin_phi)
    
    # --- SPACE WEATHER ---
    flush("\n--- SPACE WEATHER ---")
    # Clear financial modules
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
    all_results['space'] = run_domain("SPACE WEATHER", space_layers, space_y, space_phi)
    
    # Save
    out = os.path.join(RESULTS, 'bootstrap_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    flush(f"\nSaved: {out}")
    flush("DONE")


if __name__ == '__main__':
    main()
