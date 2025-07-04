# Goldilocks_CRI

Code, data, and figures for the Conscious Retroactive Intervention (CRI) Perspective manuscript and Supplementary Information (SI).

---

## Overview

This repository supports the Perspective submission:

**"Conscious Retroactive Intervention: A Reversed-Time Framework for Predictive Cognition"**

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
├── epochs_features/            # Epoch extraction and x(t) features
├── statistics/                 # Permutation tests and power analyses
├── figures/                    # Figure scripts and output (PDF)
├── stimulus_presentation/      # Psychopy cue schedule generator
├── utilities/                  # Shared utility functions
├── run_all.sh                  # Master pipeline script (recommended)
├── README.md
└── default_params.yml          # Master parameter file
```










# Goldilocks_CRI

Code, data, and figures for the Conscious Retroactive Intervention (CRI) Perspective manuscript and Supplementary Information (SI).

---

## Overview

This repository supports the Perspective submission:

> **"Conscious Retroactive Intervention: A Reversed-Time Framework for Predictive Cognition"**

It provides all code, synthetic data, simulations, figures, and numerical pipelines to transparently reproduce and audit the CRI framework and its supplementary analyses.

CRI (Goldilocks-CRI) proposes that neural systems can—under sharply tuned, “just-right” physiological conditions—be influenced by probabilistic information about future outcomes. This model combines predictive coding, quantum process tomography, and novel “Goldilocks” retrocausal gating, producing specific, testable signatures in behavior, EEG, and simulated neural dynamics.

---

## Repository Structure

Goldilocks_CRI/
├── synthetic_EEG/ # Synthetic EEG generator and output
├── preprocessing/ # EEG artifact detection and cleaning
├── decay/ # Retrocausal decay simulations & fits
├── logistic_gate/ # Logistic gating simulations & fits
├── qpt/ # Quantum process tomography analysis
├── epochs_features/ # Epoch extraction and x(t) features
├── statistics/ # Permutation tests and power analyses
├── figures/ # Figure scripts and output (PDF)
├── stimulus_presentation/# Psychopy cue schedule generator
├── utilities/ # Shared utility functions
├── run_all.sh # Master pipeline script (recommended)
├── README.md
└── default_params.yml # Master parameter file

---

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

- Python 3.9+
- MNE-Python (≥ 1.3)
- numpy, scipy, pandas, matplotlib, pyyaml
- Picard (for ICA)
- psychopy (optional: for cue schedule)

> See `environment.yml` or `requirements.txt` (if provided)

### Quick Environment Setup (Conda recommended)






