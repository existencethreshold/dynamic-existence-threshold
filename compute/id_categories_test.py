#!/usr/bin/env python3
"""
Explicit I-D category test:
  CASCADE:     dI > 0 AND dD < 0 (integration rising, differentiation falling)
  COINCIDENCE: dI > 0 AND dD > 0 (both rising independently)
  DISSOLVING:  dI < 0 AND dD > 0 (integration falling, differentiation rising)
  COLLAPSING:  dI < 0 AND dD < 0 (both falling)
  BALANCE:     |dI| ≈ |dD| (within some tolerance)

Question: What % of events fall in each category vs normal days?
"""
import numpy as np
import sys, os, json

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

def run_domain(name, layers_df, labels, phi_mod):
    flush(f"\n{'='*60}")
    flush(f"  {name}")
    flush(f"{'='*60}")

    y = np.array(labels)
    Rs, Ss, Ds = compute_rsd_timeseries(layers_df, phi_mod)
    
    # Use both I proxies
    I_rs = Rs * Ss
    layer_sum = layers_df[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    
    results = {}
    
    for i_name, I_vals in [('R*S', I_rs), ('Sum', layer_sum)]:
        dI = np.diff(I_vals)
        dD = np.diff(Ds)
        y_d = y[1:]
        
        event_mask = y_d == 1
        normal_mask = y_d == 0
        n_events = event_mask.sum()
        n_normal = normal_mask.sum()
        
        flush(f"\n  --- Using I = {i_name} ---")
        flush(f"  Total: {len(y_d)} days ({n_events} events, {n_normal} normal)")
        
        # Define categories
        cascade   = (dI > 0) & (dD < 0)   # I up, D down
        coincide  = (dI > 0) & (dD > 0)   # both up
        dissolve  = (dI < 0) & (dD > 0)   # I down, D up (recovery)
        collapse  = (dI < 0) & (dD < 0)   # both down
        
        # Balance: |dI| and |dD| within 10% of each other (normalized)
        dI_norm = np.abs(dI) / (np.std(dI) + 1e-10)
        dD_norm = np.abs(dD) / (np.std(dD) + 1e-10)
        balance = np.abs(dI_norm - dD_norm) < 0.25
        
        categories = {
            'CASCADE  (dI>0, dD<0)': cascade,
            'COINCIDE (dI>0, dD>0)': coincide,
            'DISSOLVE (dI<0, dD>0)': dissolve,
            'COLLAPSE (dI<0, dD<0)': collapse,
            'BALANCE  (|dI|≈|dD|) ': balance,
        }
        
        flush(f"\n  {'Category':<28} {'Events':>8} {'Normal':>8} {'Gap':>8}  {'Event Rate':>10}")
        flush(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
        
        cat_results = {}
        for cat_name, mask in categories.items():
            evt_pct = mask[event_mask].mean() if n_events > 0 else 0
            nrm_pct = mask[normal_mask].mean() if n_normal > 0 else 0
            gap = evt_pct - nrm_pct
            
            # Event rate WITHIN this category
            if mask.sum() > 0:
                event_rate = y_d[mask].mean()
                n_in_cat = mask.sum()
            else:
                event_rate = 0
                n_in_cat = 0
            
            flush(f"  {cat_name}  {evt_pct:>7.1%}  {nrm_pct:>7.1%}  {gap:>+7.1%}   {event_rate:>7.1%} ({n_in_cat})")
            
            cat_results[cat_name.strip()] = {
                'event_pct': round(float(evt_pct), 4),
                'normal_pct': round(float(nrm_pct), 4),
                'gap': round(float(gap), 4),
                'event_rate_in_category': round(float(event_rate), 4),
                'n_days_in_category': int(n_in_cat),
            }
        
        # Key question: if we ONLY flag cascade category, what's precision/recall?
        cascade_events = y_d[cascade].sum() if cascade.sum() > 0 else 0
        cascade_total = cascade.sum()
        precision = cascade_events / cascade_total if cascade_total > 0 else 0
        recall = cascade_events / n_events if n_events > 0 else 0
        
        flush(f"\n  CASCADE as detector:")
        flush(f"    Precision: {precision:.1%} ({int(cascade_events)}/{cascade_total} days in CASCADE are events)")
        flush(f"    Recall:    {recall:.1%} ({int(cascade_events)}/{n_events} events caught)")
        
        # What about NOT-COINCIDENCE as detector?
        not_coincide = ~coincide
        nc_events = y_d[not_coincide].sum() if not_coincide.sum() > 0 else 0
        nc_precision = nc_events / not_coincide.sum() if not_coincide.sum() > 0 else 0
        nc_recall = nc_events / n_events if n_events > 0 else 0
        
        flush(f"\n  NOT-COINCIDENCE as detector:")
        flush(f"    Precision: {nc_precision:.1%}")
        flush(f"    Recall:    {nc_recall:.1%}")
        
        cat_results['cascade_detector'] = {
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
        }
        results[i_name] = cat_results
    
    return results

def main():
    flush("="*60)
    flush("  I-D CATEGORY TEST")
    flush("  CASCADE vs COINCIDENCE vs DISSOLVE vs COLLAPSE")
    flush("="*60)
    
    all_results = {}
    
    # --- FINANCIAL ---
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
    
    # --- SPACE ---
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
    
    out = os.path.join(RESULTS, 'id_categories_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    flush(f"\n  Saved: {out}")

if __name__ == '__main__':
    main()
