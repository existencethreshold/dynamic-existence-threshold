# Dynamic Existence Threshold (DET)

**The Dynamic Existence Threshold: Organizational Consciousness across Complex Systems**

Nathan M. Thornhill — Independent Researcher, Fort Wayne, Indiana, USA

## Overview

Analysis code, precomputed results, and publication figures for the Dynamic Existence Threshold (DET) framework. The DET introduces a two-dimensional coordinate system (Integration × Differentiation) that characterizes critical transitions across financial markets, space weather, and human EEG without parameter tuning between domains.

**Key result:** The coupling metric I = R×S achieves AUC 0.909 for discriminating wakefulness from deep sleep in 50-subject EEG, while simple summation of spectral power *anti-predicts* (AUC 0.416) — confirming the framework captures organizational structure rather than magnitude. Out-of-sample temporal validation confirms no overfitting (all metrics hold on held-out test periods). Component analysis reveals that total organizational information (Phi = I + D) is approximately conserved during critical transitions (within 1-14% across all domains), while components undergo inverse exchange — integration surges while differentiation drops in cascades, and the reverse during consciousness transitions.

## Repository Structure

```
├── Thornhill_2026_Dynamic_Existence_Threshold.pdf  # Paper (Zenodo)
├── compute/                                # Analysis pipelines
│   ├── phase7b_tests.py                    #   Financial + space weather (Tests 1-4)
│   ├── phase7b_eeg_50subj.py               #   EEG 50-subject analysis (Test 5)
│   ├── phase7b_eeg_test.py                 #   EEG 5-subject pilot
│   ├── revisions.py                        #   FDR, negative controls, L4 ablation, EWS baseline
│   ├── out_of_sample_validation.py         #   Temporal split OOS validation
│   ├── null_model_id_v2.py                 #   Null model for I-D classification
│   ├── delta_beta_headtohead.py            #   Delta/beta ratio vs R×S comparison
│   ├── eeg_loader_direct.py                #   EEG data loader (direct bandpass)
│   ├── layer_sensitivity_finspace.py       #   Layer count sensitivity (financial + space)
│   ├── layer_sensitivity_eeg.py            #   Layer count sensitivity (EEG)
│   ├── bootstrap_and_sweep.py              #   Bootstrap confidence intervals
│   ├── id_categories_test.py               #   I-D directional dynamics
│   ├── phi_conservation_test.py            #   Phi conservation analysis (Test 6)
│   ├── generate_paper_figures.py           #   Publication figures (300 DPI)
│   └── layer_sensitivity_combined_plot.py  #   Figure 6
├── results/                                # Precomputed results (JSON)
│   ├── phase7b_results.json                #   Financial + space weather
│   ├── phase7b_eeg_50subj.json             #   EEG 50-subject
│   ├── phase7b_eeg_results.json            #   EEG 5-subject pilot
│   ├── revisions.json                      #   FDR, negative controls, ablation, EWS
│   ├── out_of_sample_validation.json       #   Temporal split OOS results
│   ├── null_model_v2_results.json          #   Null model results
│   ├── delta_beta_results.json             #   Delta/beta comparison
│   ├── layer_sensitivity_finspace.json     #   Layer sensitivity (financial + space)
│   ├── layer_sensitivity_eeg.json          #   Layer sensitivity (EEG)
│   ├── bootstrap_results.json              #   Bootstrap CIs
│   ├── id_categories_results.json          #   Directional dynamics
│   └── phi_conservation_results.json       #   Phi conservation (Test 6)
└── figures/                                # Publication figures (300 DPI)
    ├── fig1_quadrant_model.png
    ├── fig2_early_warning.png
    ├── fig3_classifier_comparison.png
    ├── fig4_eeg_stages.png
    ├── fig5_residence.png
    └── fig6_layer_sensitivity.png
```

## Data Sources

| Domain | Records | Source |
|--------|---------|--------|
| Financial markets | 6,785 trading days | [yfinance](https://pypi.org/project/yfinance/) |
| Space weather | 9,802 days | [NASA OMNI2](https://omniweb.gsfc.nasa.gov/) |
| EEG | 50 subjects, 136,394 epochs | [Sleep-EDF, PhysioNet](https://physionet.org/content/sleep-edfx/) |

## Setup

The compute scripts import modules from separate project directories (phi-financial-test, phi-space-test, phi-neural-test). Set these environment variables before running:

```bash
# Required for financial + space weather scripts
export PHI_FINANCIAL_DIR=./external/phi-financial-test
export PHI_SPACE_DIR=./external/phi-space-test

# Required for EEG scripts (layer_sensitivity_eeg.py, phase7b_eeg_test.py)
export PHI_NEURAL_DIR=./external/phi-neural-test

# Required for EEG scripts that load Sleep-EDF data directly
# (eeg_loader_direct.py, phase7b_eeg_50subj.py)
export SLEEP_EDF_DIR=./external/sleep-edf
```

If not set, scripts fall back to default paths (`./external/`) and exit with an informative error if those defaults do not exist.

> **Note:** The `phi-*-test` directories contain shared data loading and metric computation functions used across related projects. These modules are not required to inspect precomputed results in `results/`. For full reproducibility, contact the author at research@nathanthornhill.com.

## Reproducing

```bash
pip install -r requirements.txt

# Financial + space weather (Tests 1-4)
python compute/phase7b_tests.py

# EEG (requires ~8GB RAM, ~30 min)
python compute/phase7b_eeg_50subj.py

# Robustness checks
python compute/revisions.py

# Out-of-sample validation
python compute/out_of_sample_validation.py

# Null model + delta/beta comparison
python compute/null_model_id_v2.py
python compute/delta_beta_headtohead.py

# Layer sensitivity
python compute/layer_sensitivity_finspace.py
python compute/layer_sensitivity_eeg.py

# Phi conservation analysis (Test 6, requires ~8GB RAM)
python compute/phi_conservation_test.py

# Figures
python compute/generate_paper_figures.py
python compute/layer_sensitivity_combined_plot.py
```

Precomputed results are in `results/` for verification without re-running.

## Prior Work

1. [The Existence Threshold](https://doi.org/10.5281/zenodo.18124074) (Thornhill, 2026a)
2. [Information Loss at Dimensional Boundaries](https://doi.org/10.5281/zenodo.18238485) (Thornhill, 2026b)
3. [The Dimensional Loss Theorem](https://doi.org/10.5281/zenodo.18319429) (Thornhill, 2026c)

## License

This repository is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse permitted with attribution — please cite the paper.

## Citation

> Thornhill, N. M. (2026). The Dynamic Existence Threshold: Organizational Consciousness across Complex Systems. *Submitted to Chaos (AIP).*
