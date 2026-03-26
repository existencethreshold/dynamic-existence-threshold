#!/usr/bin/env python3
"""
Generate publication-quality figures for DET paper:
"The Dynamic Existence Threshold"
300 DPI, consistent styling, journal-ready.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Publication styling
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.family'] = 'sans-serif'

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
OUTDIR = os.path.join(BASEDIR, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

def load_json(fname):
    path = os.path.join(RESULTS, fname)
    with open(path) as f:
        return json.load(f)

def load_eeg():
    """Load EEG results: prefer 50-subject, fall back to 5-subject."""
    for fname in ['phase7b_eeg_50subj.json', 'phase7b_eeg_30subj.json', 'phase7b_eeg_results.json']:
        path = os.path.join(RESULTS, fname)
        if os.path.exists(path) and os.path.getsize(path) > 10:
            with open(path) as f:
                data = json.load(f)
            print(f"  [EEG] Loaded {fname} ({data.get('n_subjects', '?')} subjects)")
            return data
    raise FileNotFoundError("No EEG results file found")

def eeg_auc(eeg, metric):
    """Get AUC value from EEG data, handling both nested and flat key formats.
    50-subj: eeg['auc']['I'], eeg['auc']['sum'], etc.
    5-subj:  eeg['auc_I'], eeg['auc_sum'], etc.
    """
    # Try nested format first (50-subj)
    if 'auc' in eeg and isinstance(eeg['auc'], dict):
        return eeg['auc'].get(metric, 0)
    # Fall back to flat format (5-subj)
    return eeg.get(f'auc_{metric}', 0)

def eeg_stage(eeg, stage, field):
    """Get stage value from EEG data, handling both key formats.
    50-subj: eeg['stage_W']['I'], eeg['stage_W']['D']
    5-subj:  eeg['stage_W']['mean_I'], eeg['stage_W']['mean_D']
    """
    stage_data = eeg.get(f'stage_{stage}', {})
    # Try direct key first (50-subj), then mean_ prefix (5-subj)
    return stage_data.get(field, stage_data.get(f'mean_{field}', 0))

def eeg_balance(eeg, thresh, field):
    """Get balance zone value from EEG data, handling both key formats.
    50-subj: eeg['balance_0.5']['wake'], eeg['balance_0.5']['n3']
    5-subj:  eeg['balance_zone_0.5']['wake_pct_in_balance'], eeg['balance_zone_0.5']['n3_pct_in_balance']
    """
    # Try 50-subj key first
    key_new = f'balance_{thresh}'
    key_old = f'balance_zone_{thresh}'
    if key_new in eeg:
        return eeg[key_new].get(field, 0)
    if key_old in eeg:
        # Map field names: 'wake' -> 'wake_pct_in_balance', 'n3' -> 'n3_pct_in_balance'
        return eeg[key_old].get(f'{field}_pct_in_balance', eeg[key_old].get(field, 0))
    return 0

def eeg_has_balance(eeg, thresh):
    """Check if balance data exists for a given threshold."""
    return f'balance_{thresh}' in eeg or f'balance_zone_{thresh}' in eeg


# ============================================================
# FIGURE 1: Four-Quadrant I-D Model (Schematic)
# ============================================================
def fig1_quadrant_model():
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Quadrant labels
    ax.text(0.75, 0.75, 'CRITICAL\nBALANCE\n(Consciousness,\nHealthy Function)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#4CAF50', alpha=0.3))
    ax.text(0.75, 0.25, 'CASCADE\n(Seizure, Crash,\nPathological Sync)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F44336', alpha=0.3))
    ax.text(0.25, 0.75, 'NORMAL IDLE\n(Quiet, Healthy,\nFragmented)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2196F3', alpha=0.3))
    ax.text(0.25, 0.25, 'DEATH / OFF\n(System Down,\nNo Activity)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#9E9E9E', alpha=0.3))
    
    # Axes
    ax.axhline(0.5, color='black', linewidth=1.5, linestyle='-', alpha=0.5)
    ax.axvline(0.5, color='black', linewidth=1.5, linestyle='-', alpha=0.5)
    
    # Balance zone (diagonal band)
    x = np.linspace(0.1, 0.9, 100)
    ax.fill_between(x, x - 0.12, x + 0.12, alpha=0.15, color='gold',
                     label='I-D Balance Zone')
    ax.plot(x, x, '--', color='goldenrod', linewidth=1.5, alpha=0.7)
    
    # Arrow: cascade exit
    ax.annotate('', xy=(0.85, 0.2), xytext=(0.6, 0.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.78, 0.42, 'Event\nExit', fontsize=9, color='red', ha='center')
    
    ax.set_xlabel('Integration (I = R × S)', fontsize=13)
    ax.set_ylabel('Differentiation (D)', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('The Four-Quadrant I-D State Model', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig1_quadrant_model.png'), bbox_inches='tight')
    plt.close()
    print("  Fig 1: Quadrant model saved")


# ============================================================
# FIGURE 2: Early Warning Signal (Financial + Space)
# ============================================================
def fig2_early_warning():
    data = load_json('phase7b_results.json')
    ew = data.get('early_warning', {})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (domain, label) in enumerate([('financial_sum', 'Financial Markets'), ('space_sum', 'Space Weather')]):
        ax = axes[idx]
        ew_d = ew.get(domain, {})
        
        leads = [5, 10, 15, 30]
        ratios = []
        pvals = []
        for lead in leads:
            key = f'w30_lead{lead}'
            if key in ew_d:
                ratios.append(ew_d[key]['ratio'])
                pvals.append(ew_d[key]['mannwhitney_p'])
            else:
                key2 = f'w15_lead{lead}'
                if key2 in ew_d:
                    ratios.append(ew_d[key2]['ratio'])
                    pvals.append(ew_d[key2]['mannwhitney_p'])
        
        if ratios:
            bars = ax.bar(range(len(leads)), ratios, color=['#1976D2', '#42A5F5', '#90CAF9', '#BBDEFB'][:len(ratios)], alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No elevation')
            ax.set_xticks(range(len(leads)))
            ax.set_xticklabels([f'{l}d' for l in leads[:len(ratios)]])
            ax.set_xlabel('Lead Time Before Event')
            ax.set_ylabel('Variance Ratio (event / baseline)')
            ax.set_title(label)
            ax.set_ylim(0, max(ratios) * 1.3)
            
            for bar, r, p in zip(bars, ratios, pvals):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(bar.get_x() + bar.get_width()/2, r + 0.05, f'{r:.2f}\n{sig}',
                        ha='center', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis='y')
    
    plt.suptitle('Early Warning: Rising |I-D| Variance Before Events', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig2_early_warning.png'), bbox_inches='tight')
    plt.close()
    print("  Fig 2: Early warning saved")


# ============================================================
# FIGURE 3: Classifier AUC Comparison (3 domains)
# ============================================================
def fig3_classifier_comparison():
    data = load_json('phase7b_results.json')
    cl = data.get('classifier', {})

    # Load EEG results (50-subj preferred)
    eeg = load_eeg()

    fig, ax = plt.subplots(figsize=(10, 6))

    domains = ['Financial', 'Space Weather', 'EEG']
    metrics = {
        'Sum': [cl.get('financial_sum',{}).get('i_proxy_auc',0),
                cl.get('space_sum',{}).get('i_proxy_auc',0),
                eeg_auc(eeg, 'sum')],
        'I (R×S)': [cl.get('financial_rs',{}).get('i_proxy_auc',0),
                    cl.get('space_rs',{}).get('i_proxy_auc',0),
                    eeg_auc(eeg, 'I')],
        'Exit Velocity': [cl.get('financial_sum',{}).get('exit_velocity_auc',0),
                         cl.get('space_sum',{}).get('exit_velocity_auc',0),
                         eeg_auc(eeg, 'exit_velocity')],
    }
    
    x = np.arange(len(domains))
    width = 0.25
    colors = {'Sum': '#FFC107', 'I (R×S)': '#1976D2', 'Exit Velocity': '#7B1FA2'}
    
    for i, (name, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i*width - width, vals, width, label=name, color=colors[name], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, max(v, 0.05) + 0.02, f'{v:.3f}',
                    ha='center', fontsize=8, fontweight='bold')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Classifier Performance Across Domains', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Highlight Sum anti-prediction in EEG
    auc_sum_val = eeg_auc(eeg, 'sum')
    ax.annotate('Sum\nANTI-PREDICTS', xy=(2 - width, auc_sum_val if auc_sum_val else 0.288),
                xytext=(2 - width, 0.12), fontsize=9, color='red', fontweight='bold',
                ha='center', arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig3_classifier_comparison.png'), bbox_inches='tight')
    plt.close()
    print("  Fig 3: Classifier comparison saved")


# ============================================================
# FIGURE 4: EEG I-D Stage Means
# ============================================================
def fig4_eeg_stages():
    # Load EEG results (50-subj preferred)
    eeg = load_eeg()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stages = ['W', 'N1', 'N2', 'N3', 'R']
    stage_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    colors_stage = ['#4CAF50', '#FFC107', '#FF9800', '#F44336', '#9C27B0']

    I_vals = [eeg_stage(eeg, s, 'I') for s in stages]
    D_vals = [eeg_stage(eeg, s, 'D') for s in stages]

    # Panel A: Bar chart of I and D
    ax = axes[0]
    x = np.arange(len(stages))
    ax.bar(x - 0.15, I_vals, 0.3, label='I (R×S)', color='#1976D2', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.15, D_vals, 0.3, label='D', color='#F44336', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel('Mean Value')
    ax.set_title('Integration and Differentiation by Sleep Stage')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # Panel B: Balance zone occupancy
    ax = axes[1]
    thresholds = [0.5, 1.0, 1.5]

    # Plot all thresholds
    wake_pcts = []
    n3_pcts = []
    thresh_labels = []
    for thresh in thresholds:
        if eeg_has_balance(eeg, thresh):
            wake_pcts.append(eeg_balance(eeg, thresh, 'wake') * 100)
            n3_pcts.append(eeg_balance(eeg, thresh, 'n3') * 100)
            thresh_labels.append(f'±{thresh}σ')
    
    if wake_pcts:
        x = np.arange(len(thresh_labels))
        ax.bar(x - 0.15, wake_pcts, 0.3, label='Wake', color='#4CAF50', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + 0.15, n3_pcts, 0.3, label='N3', color='#F44336', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(thresh_labels)
        ax.set_ylabel('% in Balance Zone')
        ax.set_title('Balance Zone Occupancy: Wake vs N3')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add ratio annotations
        for i, (w, n) in enumerate(zip(wake_pcts, n3_pcts)):
            if n > 0:
                ax.text(i, max(w, n) + 3, f'{w/n:.0f}×', ha='center', fontsize=10, fontweight='bold', color='darkgreen')
            else:
                ax.text(i, w + 3, '∞', ha='center', fontsize=12, fontweight='bold', color='darkgreen')
    
    plt.suptitle('EEG: The Crucial Experiment', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig4_eeg_stages.png'), bbox_inches='tight')
    plt.close()
    print("  Fig 4: EEG stages saved")


# ============================================================
# FIGURE 5: Residence Time (Financial + Space)
# ============================================================
def fig5_residence():
    data = load_json('phase7b_results.json')
    res = data.get('residence', {})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (domain_key, label) in enumerate([('financial_sum', 'Financial Markets'), ('space_sum', 'Space Weather')]):
        ax = axes[idx]
        res_d = res.get(domain_key, {})
        
        thresholds = ['0.5', '0.75', '1.0', '1.5']
        normal_res = []
        event_res = []
        pvals = []
        valid_thresh = []
        
        for t in thresholds:
            key = f'thresh_{t}'
            if key in res_d:
                normal_res.append(res_d[key].get('median_normal_exit', 0))
                event_res.append(res_d[key].get('median_event_exit', 0))
                pvals.append(res_d[key].get('mannwhitney_p', 1))
                valid_thresh.append(f'±{t}σ')
        
        if valid_thresh:
            x = np.arange(len(valid_thresh))
            ax.bar(x - 0.15, normal_res, 0.3, label='Normal', color='#4CAF50', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.bar(x + 0.15, event_res, 0.3, label='Event', color='#F44336', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(valid_thresh)
            ax.set_xlabel('Balance Zone Threshold')
            ax.set_ylabel('Median Residence (days)')
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.2, axis='y')
            
            for i, p in enumerate(pvals):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(i, max(normal_res[i], event_res[i]) + 0.15, sig,
                        ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Balance Zone Residence Time: Normal vs Events', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig5_residence.png'), bbox_inches='tight')
    plt.close()
    print("  Fig 5: Residence time saved")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Generating publication figures...")
    print(f"Output: {OUTDIR}")
    
    fig1_quadrant_model()
    fig2_early_warning()
    fig3_classifier_comparison()
    fig4_eeg_stages()
    fig5_residence()
    
    print(f"\nDone! All figures saved to {OUTDIR}")
