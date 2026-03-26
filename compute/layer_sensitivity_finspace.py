#!/usr/bin/env python3
"""
Layer Sensitivity Test — Financial + Space Weather
Tests DET framework robustness to layer count N ∈ {3, 4, 5, 6, 8}.
N<5: merge adjacent layers. N>5: compute from raw data.
"""
import numpy as np
import sys, os, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

# Paths relative to repo root
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
FIGURES = os.path.join(BASEDIR, 'figures')


def flush(msg):
    print(msg, flush=True)

# ============================================================
# GENERIC R,S,D COMPUTATION (works for any N)
# ============================================================
def compute_rsd_from_df(layers_df, phi_mod, n_layers):
    """Compute R, S, D timeseries from a layers DataFrame with N columns."""
    cols = [f'L{i}' for i in range(n_layers)]
    data = layers_df[cols].values.astype(float)
    n = len(data)
    Rs = np.zeros(n)
    Ss = np.zeros(n)
    Ds = np.zeros(n)
    for i in range(n):
        bp = data[i]
        swd = np.argmax(data[max(0,i-8):i], axis=1) if i >= 8 else None
        Rs[i] = phi_mod.compute_R(bp, n_layers=n_layers, eff_threshold=2.0)
        Ss[i] = phi_mod.compute_S(bp, sub_window_dominants=swd, n_layers=n_layers, eff_threshold=2.0)
        Ds[i] = phi_mod.compute_D(bp, n_layers=n_layers)
    return Rs, Ss, Ds

def get_id_measures(I_vals, D_vals):
    I_z = (I_vals - np.mean(I_vals)) / (np.std(I_vals) + 1e-10)
    D_z = (D_vals - np.mean(D_vals)) / (np.std(D_vals) + 1e-10)
    balance = I_z - D_z
    exit_velocity = np.abs(np.diff(balance))
    abs_balance = np.abs(balance)
    return I_z, D_z, balance, exit_velocity, abs_balance

def run_auc_tests(I_vals, D_vals, layer_sum, y, label):
    """Run classifier AUC tests. Returns dict of results."""
    results = {}
    I_z, D_z, balance, exit_velocity, abs_balance = get_id_measures(I_vals, D_vals)
    y_ev = y[1:]

    # I (R×S) as classifier
    if y.sum() > 0 and (y == 0).sum() > 0:
        results['auc_i_rs'] = float(roc_auc_score(y, I_vals))
        results['auc_sum'] = float(roc_auc_score(y, layer_sum))
        results['auc_abs_balance'] = float(roc_auc_score(y, abs_balance))

    if y_ev.sum() > 0 and (y_ev == 0).sum() > 0:
        results['auc_exit_velocity'] = float(roc_auc_score(y_ev, exit_velocity))
        u, p = mannwhitneyu(exit_velocity[y_ev == 1], exit_velocity[y_ev == 0], alternative='greater')
        results['ev_mannwhitney_p'] = float(p)
        results['ev_ratio'] = float(np.median(exit_velocity[y_ev == 1]) / (np.median(exit_velocity[y_ev == 0]) + 1e-10))

    # Early warning: 30d rolling variance of |I-D|, 5-day lead
    window = 30
    roll_var = np.array([np.var(abs_balance[max(0,i-window):i]) for i in range(window, len(abs_balance))])
    event_indices = np.where(y == 1)[0]
    lead = 5
    pre_event_vars = []
    for ev_idx in event_indices:
        check_idx = ev_idx - lead - window
        if 0 <= check_idx < len(roll_var):
            pre_event_vars.append(roll_var[check_idx])
    normal_indices = np.where(y[window:] == 0)[0]
    if len(pre_event_vars) > 5 and len(normal_indices) > 5:
        baseline_vars = roll_var[normal_indices]
        pre_arr = np.array(pre_event_vars)
        ratio = float(np.median(pre_arr) / (np.median(baseline_vars) + 1e-10))
        u, p = mannwhitneyu(pre_arr, baseline_vars, alternative='greater')
        results['ews_var_ratio_5d'] = ratio
        results['ews_var_p_5d'] = float(p)

    # Residence time at ±1.5σ
    thresh = 1.5
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
    if (exit_labels == 0).sum() > 5 and (exit_labels == 1).sum() > 5:
        u, p = mannwhitneyu(res_times[exit_labels == 1], res_times[exit_labels == 0], alternative='two-sided')
        results['residence_p'] = float(p)
        results['residence_normal_median'] = float(np.median(res_times[exit_labels == 0]))
        results['residence_event_median'] = float(np.median(res_times[exit_labels == 1]))

    results['n_events'] = int(y.sum())
    results['n_total'] = int(len(y))
    return results

# ============================================================
# FINANCIAL LAYER VARIANTS
# ============================================================
def financial_layers_merge(layers_5, n_target):
    """N<5: merge adjacent layers."""
    if n_target == 3:
        return pd.DataFrame({
            'L0': layers_5[['L0','L1']].mean(axis=1),
            'L1': layers_5[['L2','L3']].mean(axis=1),
            'L2': layers_5['L4'],
        }, index=layers_5.index)
    elif n_target == 4:
        return pd.DataFrame({
            'L0': layers_5['L0'],
            'L1': layers_5['L1'],
            'L2': layers_5[['L2','L3']].mean(axis=1),
            'L3': layers_5['L4'],
        }, index=layers_5.index)
    raise ValueError(f"N={n_target}")

def financial_layers_6(sector_returns, vix, haven_returns, lookback=60):
    """N=6: standard 5 + VIX rate of change."""
    from market_layers import compute_all_days, compute_layer_activations
    layers_5 = compute_all_days(sector_returns, vix, haven_returns, lookback, baseline_correct_l1=False)

    results = []
    for date in layers_5.index:
        row = layers_5.loc[date]
        act = np.zeros(6)
        act[0:5] = [row['L0'], row['L1'], row['L2'], row['L3'], row['L4']]

        # L5: VIX rate of change (z-scored)
        if date in vix.index:
            vix_loc = vix.index.get_loc(date)
            if vix_loc >= lookback and vix_loc >= 1:
                vix_today = float(vix.iloc[vix_loc])
                vix_yesterday = float(vix.iloc[vix_loc - 1])
                if vix_today > 0 and vix_yesterday > 0:
                    dvix = (vix_today - vix_yesterday) / vix_yesterday
                    # z-score against trailing window
                    dvix_series = vix.pct_change()
                    dvix_window = dvix_series.iloc[vix_loc-lookback:vix_loc].dropna()
                    if len(dvix_window) >= 30:
                        mu = dvix_window.mean()
                        sig = dvix_window.std()
                        if sig > 0:
                            act[5] = max(0.0, (dvix - mu) / sig)
        results.append({'date': date, **{f'L{j}': act[j] for j in range(6)}})
    return pd.DataFrame(results).set_index('date')

def financial_layers_8(sector_returns, vix, haven_returns, lookback=60):
    """N=8: finer hierarchy — split stress, corr window, haven components."""
    from market_layers import compute_all_days
    layers_5 = compute_all_days(sector_returns, vix, haven_returns, lookback, baseline_correct_l1=False)

    results = []
    for date in layers_5.index:
        row = layers_5.loc[date]
        act = np.zeros(8)

        loc_s = sector_returns.index.get_loc(date) if date in sector_returns.index else -1
        if loc_s < lookback:
            results.append({'date': date, **{f'L{j}': act[j] for j in range(8)}})
            continue

        sector_window = sector_returns.iloc[loc_s-lookback:loc_s]
        sector_today = sector_returns.iloc[loc_s]
        rolling_mean = sector_window.mean()
        rolling_std = sector_window.std().replace(0, np.nan)
        z_scores = ((sector_today - rolling_mean) / rolling_std).dropna()

        # L0: fraction >2σ (standard L0)
        act[0] = (z_scores.abs() > 2).mean() if len(z_scores) > 0 else 0
        # L1: fraction >1σ (moderate stress — new)
        act[1] = (z_scores.abs() > 1).mean() if len(z_scores) > 0 else 0
        # L2: mean |z| (standard L1)
        act[2] = z_scores.abs().mean() if len(z_scores) > 0 else 0
        # L3: cross-sector corr 5d (standard L2)
        act[3] = row['L2']
        # L4: cross-sector corr 20d (longer window — new)
        if loc_s >= 20:
            corr_window = sector_returns.iloc[loc_s-19:loc_s+1]
            corr_matrix = corr_window.corr()
            n_c = len(corr_matrix)
            upper = []
            for ci in range(n_c):
                for cj in range(ci+1, n_c):
                    val = corr_matrix.iloc[ci, cj]
                    if not np.isnan(val):
                        upper.append(val)
            act[4] = max(0.0, np.mean(upper)) if upper else 0
        # L5: VIX z-score (standard L3)
        act[5] = row['L3']
        # L6: TLT haven (bond flight — split from L4)
        # L7: GLD haven (gold flight — split from L4)
        if date in haven_returns.index:
            h_loc = haven_returns.index.get_loc(date)
            if h_loc >= lookback:
                h_window = haven_returns.iloc[h_loc-lookback:h_loc]
                h_today = haven_returns.iloc[h_loc]
                h_mean = h_window.mean()
                h_std = h_window.std().replace(0, np.nan)
                h_z = ((h_today - h_mean) / h_std).dropna()
                mean_sector_return = sector_today.mean()
                if mean_sector_return < 0:
                    if 'TLT' in h_z.index and h_z['TLT'] > 0:
                        act[6] = float(h_z['TLT'])
                    if 'GLD' in h_z.index and h_z['GLD'] > 0:
                        act[7] = float(h_z['GLD'])
        results.append({'date': date, **{f'L{j}': act[j] for j in range(8)}})
    return pd.DataFrame(results).set_index('date')

# ============================================================
# SPACE WEATHER LAYER VARIANTS
# ============================================================
def space_layers_merge(layers_5, n_target):
    """N<5: merge adjacent layers."""
    if n_target == 3:
        return pd.DataFrame({
            'L0': layers_5['L0'],
            'L1': layers_5['L1'],
            'L2': layers_5[['L2','L3','L4']].mean(axis=1),
        }, index=layers_5.index)
    elif n_target == 4:
        return pd.DataFrame({
            'L0': layers_5['L0'],
            'L1': layers_5['L1'],
            'L2': layers_5[['L2','L3']].mean(axis=1),
            'L3': layers_5['L4'],
        }, index=layers_5.index)
    raise ValueError(f"N={n_target}")

def space_layers_6(daily_df, lookback=60):
    """N=6: split solar wind into V and N*V."""
    from space_layers import compute_all_days as compute_5
    layers_5 = compute_5(daily_df, lookback)
    results = []
    for date in layers_5.index:
        row = layers_5.loc[date]
        loc = daily_df.index.get_loc(date)
        window = daily_df.iloc[loc-lookback:loc]
        today = daily_df.iloc[loc]
        act = np.zeros(6)
        act[0] = row['L0']  # F10.7
        # L1: V only (solar wind speed)
        if 'V' in daily_df.columns and not np.isnan(today['V']):
            v_w = window['V'].dropna()
            if len(v_w) >= 30:
                mu, sig = v_w.mean(), v_w.std()
                if sig > 0: act[1] = max(0, (today['V'] - mu) / sig)
        act[2] = row['L1']  # N*V pressure
        act[3] = row['L2']  # Dst
        act[4] = row['L3']  # Kp
        act[5] = row['L4']  # AE
        results.append({'date': date, **{f'L{j}': act[j] for j in range(6)}})
    return pd.DataFrame(results).set_index('date')

def space_layers_8(daily_df, lookback=60):
    """N=8: F10.7, N, V, N*V, Dst, dDst/dt, Kp, AE."""
    from space_layers import compute_all_days as compute_5
    layers_5 = compute_5(daily_df, lookback)
    results = []
    for date in layers_5.index:
        row = layers_5.loc[date]
        loc = daily_df.index.get_loc(date)
        window = daily_df.iloc[loc-lookback:loc]
        today = daily_df.iloc[loc]
        act = np.zeros(8)
        act[0] = row['L0']  # F10.7
        # L1: N (proton density)
        if 'N' in daily_df.columns and not np.isnan(today['N']):
            n_w = window['N'].dropna()
            if len(n_w) >= 30:
                mu, sig = n_w.mean(), n_w.std()
                if sig > 0: act[1] = max(0, (today['N'] - mu) / sig)
        # L2: V
        if 'V' in daily_df.columns and not np.isnan(today['V']):
            v_w = window['V'].dropna()
            if len(v_w) >= 30:
                mu, sig = v_w.mean(), v_w.std()
                if sig > 0: act[2] = max(0, (today['V'] - mu) / sig)
        act[3] = row['L1']  # N*V pressure
        act[4] = row['L2']  # Dst
        # L5: dDst/dt
        if 'Dst' in daily_df.columns and loc >= 1:
            dst_t = today['Dst']
            dst_y = daily_df.iloc[loc-1]['Dst']
            if not np.isnan(dst_t) and not np.isnan(dst_y):
                ddst = abs(dst_t - dst_y)
                ddst_series = daily_df['Dst'].diff().abs()
                ddst_w = ddst_series.iloc[loc-lookback:loc].dropna()
                if len(ddst_w) >= 30:
                    mu, sig = ddst_w.mean(), ddst_w.std()
                    if sig > 0: act[5] = max(0, (ddst - mu) / sig)
        act[6] = row['L3']  # Kp
        act[7] = row['L4']  # AE
        results.append({'date': date, **{f'L{j}': act[j] for j in range(8)}})
    return pd.DataFrame(results).set_index('date')

# ============================================================
# MAIN
# ============================================================
def main():
    flush("=" * 60)
    flush("  LAYER SENSITIVITY TEST — FINANCIAL + SPACE WEATHER")
    flush("=" * 60)

    outdir = RESULTS
    all_results = {}
    N_VALUES = [3, 4, 5, 6, 8]

    # ---- FINANCIAL ----
    flush("\n>>> FINANCIAL DOMAIN <<<")
    _fin_dir = os.environ.get("PHI_FINANCIAL_DIR", os.path.expanduser("./external/phi-financial-test"))
    if not os.path.isdir(_fin_dir):
        sys.exit(f"ERROR: PHI_FINANCIAL_DIR not found: {_fin_dir}. Set PHI_FINANCIAL_DIR env var.")
    sys.path.insert(0, _fin_dir)
    os.chdir(_fin_dir)
    from data_loader import download_all
    from market_layers import compute_all_days as fin_compute_5
    from ground_truth import label_days as fin_label
    import phi_functions as fin_phi

    sector_returns, vix, haven_returns = download_all()
    layers_5_fin = fin_compute_5(sector_returns, vix, haven_returns, baseline_correct_l1=False)
    fin_y = fin_label(layers_5_fin.index).values

    for N in N_VALUES:
        flush(f"\n  Financial N={N}...")
        if N < 5:
            layers_n = financial_layers_merge(layers_5_fin, N)
        elif N == 5:
            layers_n = layers_5_fin
        elif N == 6:
            layers_n = financial_layers_6(sector_returns, vix, haven_returns)
        elif N == 8:
            layers_n = financial_layers_8(sector_returns, vix, haven_returns)

        # Align labels
        y_n = fin_label(layers_n.index).values

        Rs, Ss, Ds = compute_rsd_from_df(layers_n, fin_phi, N)
        I_rs = Rs * Ss
        cols = [f'L{i}' for i in range(N)]
        layer_sum = layers_n[cols].values.sum(axis=1)

        results = run_auc_tests(I_rs, Ds, layer_sum, y_n, f'financial_N{N}')
        all_results[f'financial_N{N}'] = results
        flush(f"    AUC(I): {results.get('auc_i_rs', '?'):.4f}, AUC(Sum): {results.get('auc_sum', '?'):.4f}, AUC(EV): {results.get('auc_exit_velocity', '?'):.4f}")

    # ---- SPACE WEATHER ----
    flush("\n>>> SPACE WEATHER DOMAIN <<<")
    for m in list(sys.modules.keys()):
        if m in ('data_loader', 'ground_truth', 'phi_functions', 'market_layers', 'space_layers'):
            del sys.modules[m]
    _space_dir = os.environ.get("PHI_SPACE_DIR", os.path.expanduser("./external/phi-space-test"))
    if not os.path.isdir(_space_dir):
        sys.exit(f"ERROR: PHI_SPACE_DIR not found: {_space_dir}. Set PHI_SPACE_DIR env var.")
    sys.path.insert(0, _space_dir)
    os.chdir(_space_dir)
    from data_loader import load_data
    from space_layers import aggregate_hourly_to_daily, compute_all_days as space_compute_5
    from ground_truth import label_days_handlabeled
    import phi_functions as space_phi

    hourly = load_data()
    daily = aggregate_hourly_to_daily(hourly)
    layers_5_space = space_compute_5(daily)
    space_y = label_days_handlabeled(layers_5_space.index).values

    for N in N_VALUES:
        flush(f"\n  Space N={N}...")
        if N < 5:
            layers_n = space_layers_merge(layers_5_space, N)
        elif N == 5:
            layers_n = layers_5_space
        elif N == 6:
            layers_n = space_layers_6(daily)
        elif N == 8:
            layers_n = space_layers_8(daily)

        y_n = label_days_handlabeled(layers_n.index).values

        Rs, Ss, Ds = compute_rsd_from_df(layers_n, space_phi, N)
        I_rs = Rs * Ss
        cols = [f'L{i}' for i in range(N)]
        layer_sum = layers_n[cols].values.sum(axis=1)

        results = run_auc_tests(I_rs, Ds, layer_sum, y_n, f'space_N{N}')
        all_results[f'space_N{N}'] = results
        flush(f"    AUC(I): {results.get('auc_i_rs', '?'):.4f}, AUC(Sum): {results.get('auc_sum', '?'):.4f}, AUC(EV): {results.get('auc_exit_velocity', '?'):.4f}")

    # ---- SAVE RESULTS ----
    outfile = os.path.join(outdir, 'layer_sensitivity_finspace.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    flush(f"\nSaved: {outfile}")

    # ---- GENERATE FIGURE ----
    flush("\nGenerating sensitivity plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, domain in enumerate(['financial', 'space']):
        # Panel 1: AUC vs N (line plot)
        ax = axes[0, col]
        ns = []
        auc_i = []
        auc_sum = []
        auc_ev = []
        for N in N_VALUES:
            key = f'{domain}_N{N}'
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
        ax.set_title(f'{domain.title()}: AUC vs Layer Count', fontsize=13)
        ax.set_xticks(N_VALUES)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
        # Shade N=5 column
        ax.axvspan(4.5, 5.5, alpha=0.1, color='blue')

        # Panel 2: EWS variance ratio vs N
        ax = axes[1, col]
        ews_ratios = []
        for N in N_VALUES:
            key = f'{domain}_N{N}'
            ews_ratios.append(all_results.get(key, {}).get('ews_var_ratio_5d', 1.0))
        ax.bar(ns, ews_ratios, color='mediumpurple', alpha=0.7, width=0.6)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No signal')
        ax.set_xlabel('Number of Layers (N)', fontsize=12)
        ax.set_ylabel('Variance Ratio (5d lead)', fontsize=12)
        ax.set_title(f'{domain.title()}: Early Warning Signal vs N', fontsize=13)
        ax.set_xticks(N_VALUES)
        ax.grid(True, alpha=0.3, axis='y')

    # Panel 3 (top right): Combined heatmap
    ax = axes[0, 2]
    heatmap_data = np.zeros((2, len(N_VALUES)))
    for i, domain in enumerate(['financial', 'space']):
        for j, N in enumerate(N_VALUES):
            key = f'{domain}_N{N}'
            heatmap_data[i, j] = all_results.get(key, {}).get('auc_i_rs', 0.5)
    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(N_VALUES)))
    ax.set_xticklabels(N_VALUES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Financial', 'Space'])
    ax.set_xlabel('Number of Layers (N)', fontsize=12)
    ax.set_title('I (R×S) AUC Heatmap', fontsize=13)
    for i in range(2):
        for j in range(len(N_VALUES)):
            ax.text(j, i, f'{heatmap_data[i,j]:.3f}', ha='center', va='center', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='AUC')

    # Panel 4 (bottom right): Summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for N in N_VALUES:
        row = [str(N)]
        for domain in ['financial', 'space']:
            key = f'{domain}_N{N}'
            r = all_results.get(key, {})
            row.append(f"{r.get('auc_i_rs', 0):.3f}")
            row.append(f"{r.get('auc_exit_velocity', 0):.3f}")
        table_data.append(row)
    table = ax.table(
        cellText=table_data,
        colLabels=['N', 'Fin I', 'Fin EV', 'Space I', 'Space EV'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title('AUC Summary', fontsize=13, pad=20)

    plt.tight_layout()
    outplot = os.path.join(outdir, 'layer_sensitivity_finspace.png')
    plt.savefig(outplot, dpi=200, bbox_inches='tight')
    plt.close()
    flush(f"Saved: {outplot}")

    # ---- PRINT SUMMARY ----
    flush("\n" + "=" * 60)
    flush("  LAYER SENSITIVITY SUMMARY")
    flush("=" * 60)
    for domain in ['financial', 'space']:
        flush(f"\n  {domain.upper()}:")
        flush(f"  {'N':>3} | {'AUC(I)':>8} | {'AUC(Sum)':>8} | {'AUC(EV)':>8} | {'EWS ratio':>9} | {'Res p':>8}")
        flush(f"  {'---':>3} | {'--------':>8} | {'--------':>8} | {'--------':>8} | {'---------':>9} | {'--------':>8}")
        for N in N_VALUES:
            key = f'{domain}_N{N}'
            r = all_results.get(key, {})
            flush(f"  {N:>3} | {r.get('auc_i_rs', 0):>8.4f} | {r.get('auc_sum', 0):>8.4f} | {r.get('auc_exit_velocity', 0):>8.4f} | {r.get('ews_var_ratio_5d', 0):>9.2f} | {r.get('residence_p', 1):>8.4f}")

    # Robustness check
    flush("\n  ROBUSTNESS CHECK:")
    for domain in ['financial', 'space']:
        auc_5 = all_results.get(f'{domain}_N5', {}).get('auc_i_rs', 0.5)
        for N in [4, 6]:
            auc_n = all_results.get(f'{domain}_N{N}', {}).get('auc_i_rs', 0.5)
            pct_diff = abs(auc_n - auc_5) / auc_5 * 100
            status = "PASS" if pct_diff < 10 else "FAIL"
            flush(f"    {domain} N={N} vs N=5: {pct_diff:.1f}% difference — {status} (<10% criterion)")

    flush("\n" + "=" * 60)
    flush("  DONE")
    flush("=" * 60)

if __name__ == '__main__':
    main()
