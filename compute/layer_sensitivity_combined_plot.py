#!/usr/bin/env python3
"""Combined layer sensitivity figure for DET paper — all 3 domains."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASEDIR, 'results')
outdir = BASEDIR

# Load results
with open(os.path.join(RESULTS, 'layer_sensitivity_finspace.json')) as f:
    finspace = json.load(f)
with open(os.path.join(RESULTS, 'layer_sensitivity_eeg.json')) as f:
    eeg = json.load(f)

N_VALUES = [3, 4, 5, 6, 8]

# Extract AUC values
def get_vals(results, domain_prefix, metric):
    return [results.get(f'{domain_prefix}_N{N}', {}).get(metric, 0.5) for N in N_VALUES]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ---- ROW 1: AUC vs N for each domain ----
domains = [
    ('Financial Markets', finspace, 'financial'),
    ('Space Weather', finspace, 'space'),
    ('EEG (Wake vs N3)', eeg, 'eeg'),
]

for col, (title, data, prefix) in enumerate(domains):
    ax = axes[0, col]
    auc_i = get_vals(data, prefix, 'auc_i_rs')
    auc_sum = get_vals(data, prefix, 'auc_sum')
    
    ax.plot(N_VALUES, auc_i, 'o-', color='#2166ac', linewidth=2.5, markersize=10, label='I (R×S)', zorder=3)
    ax.plot(N_VALUES, auc_sum, 's--', color='#b2182b', linewidth=2.5, markersize=10, label='Sum', zorder=3)
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvspan(4.5, 5.5, alpha=0.08, color='#2166ac', zorder=0)
    ax.text(5, ax.get_ylim()[0] + 0.02, 'N=5\n(standard)', ha='center', fontsize=8, color='#2166ac', alpha=0.7)
    
    ax.set_xlabel('Number of Layers (N)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(N_VALUES)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    if prefix == 'eeg':
        ax.set_ylim(0.2, 1.0)
    else:
        ax.set_ylim(0.4, 1.0)

# ---- ROW 2: Combined heatmap + robustness summary ----

# Panel 4: 3-domain heatmap
ax = axes[1, 0]
heatmap = np.zeros((3, len(N_VALUES)))
domain_labels = ['Financial', 'Space', 'EEG']
for i, (prefix, data) in enumerate([('financial', finspace), ('space', finspace), ('eeg', eeg)]):
    for j, N in enumerate(N_VALUES):
        heatmap[i, j] = data.get(f'{prefix}_N{N}', {}).get('auc_i_rs', 0.5)

im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(N_VALUES)))
ax.set_xticklabels(N_VALUES, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(domain_labels, fontsize=11)
ax.set_xlabel('Number of Layers (N)', fontsize=12)
ax.set_title('I (R×S) AUC Across Domains', fontsize=14, fontweight='bold')
for i in range(3):
    for j in range(len(N_VALUES)):
        color = 'white' if heatmap[i,j] > 0.75 else 'black'
        ax.text(j, i, f'{heatmap[i,j]:.3f}', ha='center', va='center', fontsize=12, fontweight='bold', color=color)
plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)

# Panel 5: EEG balance zone ratios
ax = axes[1, 1]
for thresh in [0.5, 1.0, 1.5]:
    ratios = [eeg.get(f'eeg_N{N}', {}).get(f'balance_{thresh}_ratio', 0) for N in N_VALUES]
    ax.plot(N_VALUES, ratios, 'o-', label=f'±{thresh}σ', linewidth=2, markersize=8)
ax.set_xlabel('Number of Layers (N)', fontsize=12)
ax.set_ylabel('Wake/N3 Balance Ratio', fontsize=12)
ax.set_title('EEG Balance Zone Ratio vs N', fontsize=14, fontweight='bold')
ax.set_xticks(N_VALUES)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Panel 6: Summary table
ax = axes[1, 2]
ax.axis('off')
table_data = []
for N in N_VALUES:
    fin_i = finspace.get(f'financial_N{N}', {}).get('auc_i_rs', 0)
    sp_i = finspace.get(f'space_N{N}', {}).get('auc_i_rs', 0)
    eeg_i = eeg.get(f'eeg_N{N}', {}).get('auc_i_rs', 0)
    eeg_s = eeg.get(f'eeg_N{N}', {}).get('auc_sum', 0)
    marker = ' *' if N == 5 else ''
    table_data.append([f'{N}{marker}', f'{fin_i:.3f}', f'{sp_i:.3f}', f'{eeg_i:.3f}', f'{eeg_s:.3f}'])
table = ax.table(
    cellText=table_data,
    colLabels=['N', 'Fin I', 'Space I', 'EEG I', 'EEG Sum'],
    loc='center', cellLoc='center',
    colWidths=[0.12, 0.18, 0.18, 0.18, 0.18]
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.0)
# Highlight N=5 row
for j in range(5):
    table[3, j].set_facecolor('#e6f0ff')
ax.set_title('Layer Sensitivity Summary\n(* = standard architecture)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(pad=2.0)
outplot = os.path.join(outdir, 'figures', 'fig6_layer_sensitivity.png')
os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
plt.savefig(outplot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {outplot}")

# Also save a lower-res version
outplot2 = os.path.join(outdir, 'layer_sensitivity_combined.png')
fig.savefig(outplot2, dpi=150, bbox_inches='tight')
print(f"Saved: {outplot2}")
