# Goldilocks_CRI

Code, data, and figures for the Conscious Retroactive Intervention (CRI) Perspective manuscript and Supplementary Information (SI).

---

## Overview

This repository supports the Perspective submission:

> **"Conscious Retroactive Intervention: A Reversed-Time Quantum Framework for Predictive Cognition"**

It provides all code, synthetic data, simulations, figures, and numerical pipelines to transparently reproduce and audit the CRI framework and its supplementary analyses.

CRI (Goldilocks-CRI) proposes that neural systems can—under sharply tuned, “just-right” physiological conditions—be influenced by probabilistic information about future outcomes. This model combines predictive coding, quantum process tomography, and novel “Goldilocks” retrocausal gating, producing specific, testable signatures in behavior, EEG, and simulated neural dynamics.

---

## Repository Structure

```text
Goldilocks_CRI/
├── synthetic_EEG/              # Synthetic EEG generator and output
├── preprocessing/              # EEG artifact detection and cleaning
├── decay/                      # Retrocausal decay simulations & fits
├── logistic_gate/              # Logistic gating simulations & fits
├── qpt/                        # Quantum process tomography analysis
├── simulation_core/            # Core simulation scripts (toy_model_master_eq.py, retro_kernel_weight.py)
├── epochs_features/            # Epoch extraction and x(t) features
├── statistics/                 # Permutation tests and power analyses
├── figures/                    # Figure scripts and output (PDF)
├── stimulus_presentation/      # Psychopy cue schedule generator
├── utilities/                  # Shared utility functions
├── run_all.sh                  # Master pipeline script (recommended)
├── README.md
└── default_params.yml          # Master parameter file
```

## Key Features

- **Synthetic EEG Data:**  
  Generates realistic EEG with injected artifacts and spindles, for robust pipeline testing.

- **Automated Artifact Removal:**  
  Detects flat, noisy, and artifact-laden channels using peak-to-peak and variance metrics; robust even on synthetic datasets.

- **Reproducible Simulations:**  
  Full code for all simulations (decay, logistic gating, QPT) with output matching SI and manuscript figures.

- **Transparent SI Reproduction:**  
  Every table and figure in the SI can be reproduced from command line.

---

## Getting Started

### Requirements

- **Python** 3.9+ (Conda-managed)
- All core dependencies are pinned in `utilities/env.yml`
- **Picard** (for ICA)
- **PsychoPy** (optional: for cue schedule)


### Quick Environment Setup (Conda recommended)
1. **Create the Conda environment**

```
conda env create \
  --name goldilocks_cri \
  --file utilities/env.yml
```
2. **Activate the environment**
```
conda activate goldilocks_cri
```
3. **Install extra Python packages**
```
pip install picard      # for ICA
pip install psychopy    # optional: for PsychoPy demos
```
4. **Verify the installation**
```
python - <<EOF
import yaml, numpy, pandas, scipy, mne, matplotlib, h5py
from statsmodels.stats.power import TTestPower
print("All imports successful.")
EOF
```
## Code ocean Capsule
> A private Code Ocean capsule will be created and linked to this repository within a few days. Reviewers will then be able to execute all analyses with one click—no local setup required.

## Make the script executable
```
chmod +x run_all.sh
```

## Full Pipeline: One-Command Reproducibility

To reproduce all SI analyses and figures (using only synthetic data):

```
bash run_all.sh
```

All intermediate and final results are written to their respective output/ folders.
Figures (PDF) are generated in figures/output/.

## Manual Pipeline Steps (for development/testing)

### 1. Synthetic EEG Generation
```
python synthetic_EEG/make_synthetic_eeg.py
```

### 2. EEG Preprocessing & Artifact Removal

```
python preprocessing/artifact_pipeline.py
```
### 3. Core Simulations
```
# Run the toy-model equations
python simulation_core/toy_model_master_eq.py

# Generate/weight the retrocausal kernel
python simulation_core/retro_kernel_weight.py
```
### 4. Decay, Logistic, QPT Simulations

```
python decay/fit_decay.py
python logistic_gate/fit_logistic.py
python qpt/qpt_fit.py
```

### 5. Epoch Extraction & Feature Computation

```
python epochs_features/extract_epochs.py
python epochs_features/compute_x_t.py
```
### 6. Statistics & Figures

```
python statistics/permutation_tests.py
python statistics/power_analysis.py
python figures/make_decay_figure.py
python figures/make_logistic_figure.py
python figures/make_tomography_figure.py
```
---

## EEG Artifact Handling Details

- Bad channel detection is robust to both synthetic and real datasets.
- Peak-to-peak, flatness, and variance thresholds are set in `default_params.yml`.
- No digitization/headshape points:
  For synthetic data, interpolation sets bad channels to `NaN` (expected behavior).
  For real EEG, full spatial interpolation is used when digitization points are available.
- EOG‑channel handling tolerates missing or `NaN` values.

---

## Data and Figures

- **Synthetic data** (TierA, TierB) are generated to match pipeline requirements and test all artifact scenarios.
- **All figures** in the SI are reproducible from the generated outputs.
- **Statistical analyses** (permutation tests, power) use published parameters.

---

## Known Warnings & Their Interpretation

- **Interpolation failed**: No digitization points found (expected for synthetic data). Not an error.

- `interpolate_bads` was called with `method='nan'`: Intended behavior for synthetic pipelines.

- **Filename warnings**: This filename ... does not conform to MNE naming conventions.
Outputs cab be renamed to e.g., `_raw.fif` or `_eeg.fif` for full BIDS/MNE compatibility.

- `tight_layout` UserWarning: This figure includes Axes that are not compatible with `tight_layout`
Figure aesthetics only; no impact on results.

---

## Using with Real EEG Data

To apply the artifact pipeline to real EEG, place the `.fif` files in the appropriate input directory, ensure digitization/headshape points are present, and update `default_params.yml` for the channel naming and thresholds. See comments in `preprocessing/artifact_pipeline.py` for configuration.

---

## Contact

Maintained by: George Sopasakis
Conscious Retroactive Intervention Project, 2025
For reproducibility issues or questions, please contact via GitHub Issues.

---

## Citation

If you use this code or data, please cite:

> Sopasakis, G. (2025). Conscious Retroactive Intervention: A Reversed-Time Quantum Framework for Predictive Cognition. Manuscript in preparation.

---

## License

MIT License. See LICENSE for details.

---

## Disclaimer

This repository contains research code supporting a Perspective manuscript. Results and simulations are for demonstration, transparency, and reproducibility only. For clinical or commercial applications, independent validation is required.

---



