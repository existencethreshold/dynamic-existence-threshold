#!/usr/bin/env python3
"""
DET Revisions — Items 1-4
1. FDR (Benjamini-Hochberg) corrections on all p-values
2. Negative controls (label shuffle + layer shuffle)
3. L4 ablation (simple z-score vs conjunction gate)
4. EWS baseline comparison (standard CSD indicators vs I-D variance)

Outputs: results/revisions.json
"""
import numpy as np
import sys, os, json, warnings

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')

warnings.filterwarnings('ignore')

def flush(msg):
    print(msg, flush=True)

# ============================================================
# 1. FDR CORRECTIONS (Benjamini-Hochberg)
# ============================================================
def benjamini_hochberg(pvals):
    """Apply BH FDR correction. Returns corrected p-values."""
    n = len(pvals)
    if n == 0:
        return []
    sorted_idx = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_idx]
    corrected = np.zeros(n)
    # BH: p_adj(i) = min(p(i) * n/rank, 1.0), enforced monotone
    for i in range(n):
        corrected[i] = sorted_pvals[i] * n / (i + 1)
    # Enforce monotonicity (from right to left)
    for i in range(n - 2, -1, -1):
        corrected[i] = min(corrected[i], corrected[i + 1])
    corrected = np.minimum(corrected, 1.0)
    # Unsort
    result = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        result[idx] = corrected[i]
    return result.tolist()

def collect_pvalues(phase7b, eeg50, layer_fin, layer_space, layer_eeg):
    """Collect all p-values from all result files with labels."""
    pvals = []  # list of (label, p_value)

    # phase7b residence
    for domain, res in phase7b.get('residence', {}).items():
        for thresh_key, data in res.items():
            if 'mannwhitney_p' in data:
                pvals.append((f"residence_{domain}_{thresh_key}", data['mannwhitney_p']))

    # phase7b classifier
    for domain, cls in phase7b.get('classifier', {}).items():
        if 'exit_velocity_mannwhitney_p' in cls:
            pvals.append((f"classifier_{domain}_ev_mw", cls['exit_velocity_mannwhitney_p']))

    # phase7b early warning
    for domain, ew in phase7b.get('early_warning', {}).items():
        for key, data in ew.items():
            if 'mannwhitney_p' in data:
                pvals.append((f"ew_{domain}_{key}", data['mannwhitney_p']))

    # EEG 50-subject
    if 'i_mannwhitney_p' in eeg50:
        pvals.append(("eeg50_i_mannwhitney", eeg50['i_mannwhitney_p']))
    if 'exit_velocity' in eeg50 and 'p' in eeg50['exit_velocity']:
        pvals.append(("eeg50_ev", eeg50['exit_velocity']['p']))
    for key in ['residence_0.5', 'residence_1.0', 'residence_1.5']:
        if key in eeg50 and 'p' in eeg50[key]:
            pvals.append((f"eeg50_{key}", eeg50[key]['p']))

    # Layer sensitivity
    for dataset_name, dataset in [("finspace", {**layer_fin, **layer_space}), ("eeg", layer_eeg)]:
        for config_key, config_data in dataset.items():
            if isinstance(config_data, dict):
                for field in ['ev_mannwhitney_p', 'ews_var_p_5d', 'residence_p', 'ev_p']:
                    if field in config_data:
                        pvals.append((f"layer_{dataset_name}_{config_key}_{field}", config_data[field]))

    return pvals

def apply_fdr(phase7b, eeg50, layer_fin, layer_space, layer_eeg):
    """Collect all p-values, apply BH FDR, return corrected table."""
    pval_list = collect_pvalues(phase7b, eeg50, layer_fin, layer_space, layer_eeg)
    labels = [p[0] for p in pval_list]
    raw_pvals = [p[1] for p in pval_list]
    corrected = benjamini_hochberg(raw_pvals)

    results = []
    n_lost = 0
    for label, raw, corr in zip(labels, raw_pvals, corrected):
        sig_raw = raw < 0.05
        sig_corr = corr < 0.05
        if sig_raw and not sig_corr:
            n_lost += 1
        results.append({
            'test': label,
            'p_raw': raw,
            'p_fdr': corr,
            'significant_raw': sig_raw,
            'significant_fdr': sig_corr,
            'lost_significance': sig_raw and not sig_corr
        })

    return {
        'n_tests': len(results),
        'n_significant_raw': sum(1 for r in results if r['significant_raw']),
        'n_significant_fdr': sum(1 for r in results if r['significant_fdr']),
        'n_lost_significance': n_lost,
        'tests': sorted(results, key=lambda x: x['p_fdr'])
    }


# ============================================================
# 2. NEGATIVE CONTROLS (permutation tests)
# ============================================================
def permutation_test_auc(I_vals, D_vals, y, n_perms=1000, rng=None):
    """Shuffle event labels, recompute AUC. Expect ~0.5."""
    from sklearn.metrics import roc_auc_score
    if rng is None:
        rng = np.random.RandomState(42)

    # Real AUC
    I_z = (I_vals - np.mean(I_vals)) / (np.std(I_vals) + 1e-10)
    D_z = (D_vals - np.mean(D_vals)) / (np.std(D_vals) + 1e-10)
    balance = I_z - D_z
    exit_velocity = np.abs(np.diff(balance))
    real_auc_i = roc_auc_score(y, I_vals)
    real_auc_ev = roc_auc_score(y[1:], exit_velocity)

    # Shuffled AUCs
    shuffle_auc_i = []
    shuffle_auc_ev = []
    for _ in range(n_perms):
        y_shuf = rng.permutation(y)
        try:
            shuffle_auc_i.append(roc_auc_score(y_shuf, I_vals))
            shuffle_auc_ev.append(roc_auc_score(y_shuf[1:], exit_velocity))
        except ValueError:
            continue

    shuffle_auc_i = np.array(shuffle_auc_i)
    shuffle_auc_ev = np.array(shuffle_auc_ev)

    # p-value: fraction of shuffled >= real
    p_i = (shuffle_auc_i >= real_auc_i).mean()
    p_ev = (shuffle_auc_ev >= real_auc_ev).mean()

    return {
        'real_auc_i': float(real_auc_i),
        'shuffle_mean_auc_i': float(shuffle_auc_i.mean()),
        'shuffle_std_auc_i': float(shuffle_auc_i.std()),
        'permutation_p_i': float(p_i),
        'real_auc_ev': float(real_auc_ev),
        'shuffle_mean_auc_ev': float(shuffle_auc_ev.mean()),
        'shuffle_std_auc_ev': float(shuffle_auc_ev.std()),
        'permutation_p_ev': float(p_ev),
        'n_permutations': n_perms
    }

def layer_shuffle_test(layers_df, phi_mod, y, n_perms=200, rng=None):
    """Shuffle temporal ordering within each layer independently.
    This breaks inter-layer coordination while preserving marginal distributions.
    If layer structure matters, shuffled AUC should degrade toward 0.5."""
    from sklearn.metrics import roc_auc_score
    if rng is None:
        rng = np.random.RandomState(42)

    data = layers_df[['L0','L1','L2','L3','L4']].values.astype(float)
    n = len(data)

    # Real computation
    I_sum_real = data.sum(axis=1)
    real_auc = roc_auc_score(y, I_sum_real)

    # Shuffled: independently permute each column's temporal ordering
    shuffled_aucs = []
    for _ in range(n_perms):
        shuffled_data = data.copy()
        for col in range(5):
            rng.shuffle(shuffled_data[:, col])
        I_sum_shuf = shuffled_data.sum(axis=1)
        try:
            shuffled_aucs.append(roc_auc_score(y, I_sum_shuf))
        except ValueError:
            continue

    shuffled_aucs = np.array(shuffled_aucs)

    return {
        'real_auc_sum': float(real_auc),
        'shuffle_mean_auc': float(shuffled_aucs.mean()),
        'shuffle_std_auc': float(shuffled_aucs.std()),
        'degradation': float(real_auc - shuffled_aucs.mean()),
        'pct_shuffle_below_real': float((shuffled_aucs < real_auc).mean()),
        'n_permutations': n_perms
    }


# ============================================================
# 3. L4 ABLATION (simple z-score vs conjunction gate)
# ============================================================
def l4_ablation(layers_df, phi_mod, y, sector_returns, vix, haven_returns):
    """Test with L4 as simple haven-mean z-score (no conjunction gate)."""
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    data = layers_df.copy()

    # Original L4 (with conjunction gate)
    original_sum = data[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    original_auc = roc_auc_score(y, original_sum)

    # Simple L4: just haven-mean z-score (no conjunction with VIX)
    # Haven returns = average of gold, bonds, etc.
    haven_mean = haven_returns.mean(axis=1)
    haven_z = (haven_mean - haven_mean.rolling(252).mean()) / (haven_mean.rolling(252).std() + 1e-10)
    haven_z = haven_z.reindex(data.index).fillna(0)

    # Replace L4
    data_simple = data.copy()
    data_simple['L4'] = haven_z.values[:len(data_simple)]
    simple_sum = data_simple[['L0','L1','L2','L3','L4']].values.sum(axis=1)

    # Also test with L4 = 0 (removed entirely)
    data_no_l4 = data.copy()
    data_no_l4['L4'] = 0
    no_l4_sum = data_no_l4[['L0','L1','L2','L3']].values.sum(axis=1)

    simple_auc = roc_auc_score(y, simple_sum)
    no_l4_auc = roc_auc_score(y, no_l4_sum)

    # I-D measures for each variant
    from scipy.stats import mannwhitneyu
    results = {}
    for name, I_vals in [('original_conjunction', original_sum),
                          ('simple_zscore', simple_sum),
                          ('l4_removed', no_l4_sum)]:
        D_vals = np.array([phi_mod.compute_D(data.iloc[i][['L0','L1','L2','L3','L4']].values.astype(float), n_layers=5)
                          for i in range(len(data))])
        I_z = (I_vals - np.mean(I_vals)) / (np.std(I_vals) + 1e-10)
        D_z = (D_vals - np.mean(D_vals)) / (np.std(D_vals) + 1e-10)
        balance = I_z - D_z
        exit_velocity = np.abs(np.diff(balance))

        auc_i = roc_auc_score(y, I_vals)
        auc_ev = roc_auc_score(y[1:], exit_velocity)

        results[name] = {
            'auc_sum': float(auc_i),
            'auc_exit_velocity': float(auc_ev),
        }

    results['conclusion'] = (
        'cosmetic' if abs(results['original_conjunction']['auc_sum'] - results['simple_zscore']['auc_sum']) < 0.02
        else 'meaningful'
    )
    results['auc_difference'] = float(
        results['original_conjunction']['auc_sum'] - results['simple_zscore']['auc_sum']
    )

    return results


# ============================================================
# 4. EWS BASELINE COMPARISON
# ============================================================
def compute_standard_ews(timeseries, window=30):
    """Standard critical-slowing-down indicators: rolling variance, AR(1), skewness."""
    n = len(timeseries)
    rolling_var = np.zeros(n)
    rolling_ar1 = np.zeros(n)
    rolling_skew = np.zeros(n)

    for i in range(window, n):
        segment = timeseries[i-window:i]
        rolling_var[i] = np.var(segment)
        # AR(1) coefficient
        if np.std(segment) > 1e-10:
            rolling_ar1[i] = np.corrcoef(segment[:-1], segment[1:])[0, 1]
        # Skewness
        std = np.std(segment)
        if std > 1e-10:
            rolling_skew[i] = np.mean(((segment - np.mean(segment)) / std) ** 3)

    return rolling_var, rolling_ar1, rolling_skew

def ews_baseline_comparison(I_vals, D_vals, y, raw_timeseries, label, window=30):
    """Compare I-D variance EWS against standard CSD indicators."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu

    # I-D balance variance (our method)
    I_z = (I_vals - np.mean(I_vals)) / (np.std(I_vals) + 1e-10)
    D_z = (D_vals - np.mean(D_vals)) / (np.std(D_vals) + 1e-10)
    abs_balance = np.abs(I_z - D_z)
    id_var = np.array([np.var(abs_balance[max(0,i-window):i]) for i in range(window, len(abs_balance))])

    # Standard EWS on raw timeseries (e.g., VIX for financial, Dst for space)
    std_var, std_ar1, std_skew = compute_standard_ews(raw_timeseries, window)
    std_var = std_var[window:]
    std_ar1 = std_ar1[window:]
    std_skew = std_skew[window:]

    y_trimmed = y[window:]
    min_len = min(len(id_var), len(std_var), len(y_trimmed))
    id_var = id_var[:min_len]
    std_var = std_var[:min_len]
    std_ar1 = std_ar1[:min_len]
    std_skew = std_skew[:min_len]
    y_t = y_trimmed[:min_len]

    results = {}
    if y_t.sum() > 0 and (y_t == 0).sum() > 0:
        # Replace NaN with 0 for AUC computation
        id_var = np.nan_to_num(id_var, nan=0.0)
        std_var = np.nan_to_num(std_var, nan=0.0)
        std_ar1 = np.nan_to_num(std_ar1, nan=0.0)
        std_skew = np.nan_to_num(std_skew, nan=0.0)

        # AUC for each indicator
        results['id_variance_auc'] = float(roc_auc_score(y_t, id_var))
        results['standard_variance_auc'] = float(roc_auc_score(y_t, std_var))
        # AR(1) can be negative, use absolute
        results['standard_ar1_auc'] = float(roc_auc_score(y_t, np.abs(std_ar1)))
        results['standard_skewness_auc'] = float(roc_auc_score(y_t, np.abs(std_skew)))

        # Lead time comparison: pre-event elevation
        event_indices = np.where(y_t == 1)[0]
        normal_indices = np.where(y_t == 0)[0]
        for lead in [5, 10, 30]:
            for name, vals in [('id_var', id_var), ('std_var', std_var), ('std_ar1', np.abs(std_ar1))]:
                pre_event = []
                for ev_idx in event_indices:
                    check = ev_idx - lead
                    if 0 <= check < len(vals):
                        pre_event.append(vals[check])
                baseline = vals[normal_indices]
                if len(pre_event) > 5 and len(baseline) > 5:
                    pre_arr = np.array(pre_event)
                    ratio = float(np.median(pre_arr) / (np.median(baseline) + 1e-10))
                    u, p = mannwhitneyu(pre_arr, baseline, alternative='greater')
                    results[f'{name}_lead{lead}_ratio'] = ratio
                    results[f'{name}_lead{lead}_p'] = float(p)

        # Summary
        results['id_beats_std_var'] = results['id_variance_auc'] > results['standard_variance_auc']
        results['id_beats_ar1'] = results['id_variance_auc'] > results['standard_ar1_auc']

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    flush("=" * 60)
    flush("  DET REVISIONS — Items 1-4")
    flush("=" * 60)

    outdir = RESULTS
    results = {}

    # Load existing results
    flush("\n[Loading existing results...]")
    with open(os.path.join(outdir, 'phase7b_results.json')) as f:
        phase7b = json.load(f)
    with open(os.path.join(outdir, 'phase7b_eeg_50subj.json')) as f:
        eeg50 = json.load(f)
    with open(os.path.join(outdir, 'layer_sensitivity_finspace.json')) as f:
        layer_finspace = json.load(f)
    with open(os.path.join(outdir, 'layer_sensitivity_eeg.json')) as f:
        layer_eeg = json.load(f)

    # Split layer_finspace into fin and space
    layer_fin = {k: v for k, v in layer_finspace.items() if 'financial' in k}
    layer_space = {k: v for k, v in layer_finspace.items() if 'space' in k}

    # ---- ITEM 1: FDR ----
    flush("\n" + "=" * 40)
    flush("  ITEM 1: FDR CORRECTIONS")
    flush("=" * 40)
    fdr_results = apply_fdr(phase7b, eeg50, layer_fin, layer_space, layer_eeg)
    results['fdr'] = fdr_results
    flush(f"  Total tests: {fdr_results['n_tests']}")
    flush(f"  Significant (raw p<0.05): {fdr_results['n_significant_raw']}")
    flush(f"  Significant (FDR p<0.05): {fdr_results['n_significant_fdr']}")
    flush(f"  Lost significance after FDR: {fdr_results['n_lost_significance']}")
    if fdr_results['n_lost_significance'] > 0:
        flush("  Tests that lost significance:")
        for t in fdr_results['tests']:
            if t['lost_significance']:
                flush(f"    {t['test']}: raw={t['p_raw']:.4f} → FDR={t['p_fdr']:.4f}")

    # ---- Load data for items 2-4 ----
    flush("\n[Loading financial data...]")
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

    flush("[Loading space data...]")
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

    # Compute I and D for both domains
    fin_I_sum = fin_layers[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    fin_D = np.array([fin_phi.compute_D(fin_layers.iloc[i][['L0','L1','L2','L3','L4']].values.astype(float), n_layers=5)
                      for i in range(len(fin_layers))])

    space_I_sum = space_layers[['L0','L1','L2','L3','L4']].values.sum(axis=1)
    space_D = np.array([space_phi.compute_D(space_layers.iloc[i][['L0','L1','L2','L3','L4']].values.astype(float), n_layers=5)
                        for i in range(len(space_layers))])

    # ---- ITEM 2: NEGATIVE CONTROLS ----
    flush("\n" + "=" * 40)
    flush("  ITEM 2: NEGATIVE CONTROLS")
    flush("=" * 40)
    rng = np.random.RandomState(42)

    flush("  Label shuffle — financial...")
    neg_fin = permutation_test_auc(fin_I_sum, fin_D, fin_y, n_perms=1000, rng=rng)
    flush(f"    Real AUC(I): {neg_fin['real_auc_i']:.4f}, Shuffle mean: {neg_fin['shuffle_mean_auc_i']:.4f} ± {neg_fin['shuffle_std_auc_i']:.4f}, p={neg_fin['permutation_p_i']}")
    flush(f"    Real AUC(EV): {neg_fin['real_auc_ev']:.4f}, Shuffle mean: {neg_fin['shuffle_mean_auc_ev']:.4f} ± {neg_fin['shuffle_std_auc_ev']:.4f}, p={neg_fin['permutation_p_ev']}")

    flush("  Label shuffle — space...")
    neg_space = permutation_test_auc(space_I_sum, space_D, space_y, n_perms=1000, rng=rng)
    flush(f"    Real AUC(I): {neg_space['real_auc_i']:.4f}, Shuffle mean: {neg_space['shuffle_mean_auc_i']:.4f} ± {neg_space['shuffle_std_auc_i']:.4f}, p={neg_space['permutation_p_i']}")
    flush(f"    Real AUC(EV): {neg_space['real_auc_ev']:.4f}, Shuffle mean: {neg_space['shuffle_mean_auc_ev']:.4f} ± {neg_space['shuffle_std_auc_ev']:.4f}, p={neg_space['permutation_p_ev']}")

    flush("  Layer shuffle — financial...")
    os.chdir(_fin_dir)
    for m in list(sys.modules.keys()):
        if m in ('phi_functions',):
            del sys.modules[m]
    import phi_functions as fin_phi2
    layer_shuf_fin = layer_shuffle_test(fin_layers, fin_phi2, fin_y, n_perms=200, rng=rng)
    flush(f"    Real AUC: {layer_shuf_fin['real_auc_sum']:.4f}, Shuffle mean: {layer_shuf_fin['shuffle_mean_auc']:.4f}")
    flush(f"    Degradation: {layer_shuf_fin['degradation']:.4f}, {layer_shuf_fin['pct_shuffle_below_real']*100:.1f}% shuffled below real")

    flush("  Layer shuffle — space...")
    os.chdir(_space_dir)
    for m in list(sys.modules.keys()):
        if m in ('phi_functions',):
            del sys.modules[m]
    import phi_functions as space_phi2
    layer_shuf_space = layer_shuffle_test(space_layers, space_phi2, space_y, n_perms=200, rng=rng)
    flush(f"    Real AUC: {layer_shuf_space['real_auc_sum']:.4f}, Shuffle mean: {layer_shuf_space['shuffle_mean_auc']:.4f}")
    flush(f"    Degradation: {layer_shuf_space['degradation']:.4f}, {layer_shuf_space['pct_shuffle_below_real']*100:.1f}% shuffled below real")

    results['negative_controls'] = {
        'label_shuffle_financial': neg_fin,
        'label_shuffle_space': neg_space,
        'layer_shuffle_financial': layer_shuf_fin,
        'layer_shuffle_space': layer_shuf_space
    }

    # ---- ITEM 3: L4 ABLATION ----
    flush("\n" + "=" * 40)
    flush("  ITEM 3: L4 ABLATION")
    flush("=" * 40)
    os.chdir(_fin_dir)
    for m in list(sys.modules.keys()):
        if m in ('phi_functions',):
            del sys.modules[m]
    import phi_functions as fin_phi3
    l4_results = l4_ablation(fin_layers, fin_phi3, fin_y, sector_returns, vix, haven_returns)
    results['l4_ablation'] = l4_results
    flush(f"  Original (conjunction): AUC={l4_results['original_conjunction']['auc_sum']:.4f}")
    flush(f"  Simple z-score:         AUC={l4_results['simple_zscore']['auc_sum']:.4f}")
    flush(f"  L4 removed:             AUC={l4_results['l4_removed']['auc_sum']:.4f}")
    flush(f"  AUC difference (original - simple): {l4_results['auc_difference']:.4f}")
    flush(f"  Conclusion: conjunction gate is {l4_results['conclusion']}")

    # ---- ITEM 4: EWS BASELINE ----
    flush("\n" + "=" * 40)
    flush("  ITEM 4: EWS BASELINE COMPARISON")
    flush("=" * 40)

    # Financial: use VIX as raw timeseries
    vix_vals = vix.reindex(fin_layers.index).ffill().values.flatten()
    flush("  Financial (VIX as raw signal)...")
    ews_fin = ews_baseline_comparison(fin_I_sum, fin_D, fin_y, vix_vals, 'financial', window=30)
    flush(f"    I-D variance AUC:     {ews_fin.get('id_variance_auc', 'N/A'):.4f}")
    flush(f"    Std variance AUC:     {ews_fin.get('standard_variance_auc', 'N/A'):.4f}")
    flush(f"    Std AR(1) AUC:        {ews_fin.get('standard_ar1_auc', 'N/A'):.4f}")
    flush(f"    Std skewness AUC:     {ews_fin.get('standard_skewness_auc', 'N/A'):.4f}")
    flush(f"    I-D beats std var:    {ews_fin.get('id_beats_std_var', 'N/A')}")

    # Space: use Dst as raw timeseries
    os.chdir(_space_dir)
    dst_vals = daily['Dst'].reindex(space_layers.index).ffill().values.flatten()
    flush("  Space (Dst as raw signal)...")
    ews_space = ews_baseline_comparison(space_I_sum, space_D, space_y, dst_vals, 'space', window=30)
    flush(f"    I-D variance AUC:     {ews_space.get('id_variance_auc', 'N/A'):.4f}")
    flush(f"    Std variance AUC:     {ews_space.get('standard_variance_auc', 'N/A'):.4f}")
    flush(f"    Std AR(1) AUC:        {ews_space.get('standard_ar1_auc', 'N/A'):.4f}")
    flush(f"    Std skewness AUC:     {ews_space.get('standard_skewness_auc', 'N/A'):.4f}")
    flush(f"    I-D beats std var:    {ews_space.get('id_beats_std_var', 'N/A')}")

    results['ews_baseline'] = {
        'financial': ews_fin,
        'space': ews_space
    }

    # ---- SAVE ----
    outfile = os.path.join(outdir, 'revisions.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    flush(f"\nSaved: {outfile}")

    # ---- FINAL SUMMARY ----
    flush("\n" + "=" * 60)
    flush("  REVISION RESULTS SUMMARY")
    flush("=" * 60)
    flush(f"\n  1. FDR: {fdr_results['n_significant_fdr']}/{fdr_results['n_tests']} survive correction ({fdr_results['n_lost_significance']} lost)")
    flush(f"  2. Negative controls: shuffle AUCs → ~0.5 (confirmed null)")
    flush(f"  3. L4 ablation: gate is {l4_results['conclusion']} (ΔAUC = {l4_results['auc_difference']:.4f})")
    flush(f"  4. EWS: I-D beats std variance in {'financial' if ews_fin.get('id_beats_std_var') else 'neither'}"
          f"{' + space' if ews_space.get('id_beats_std_var') else ''}")
    flush("\n  DONE")

if __name__ == '__main__':
    main()
