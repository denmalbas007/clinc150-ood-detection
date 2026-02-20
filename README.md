# Out-of-Domain Detection for Intent Classification on CLINC150

This repository contains code for the NLP course project on OOD detection for intent classification using the CLINC150 dataset.

## Task

Given a user utterance, determine whether it belongs to one of the 150 known intent classes (in-domain) or is an out-of-domain (OOD) request.

## Methods Implemented

| Method | Description |
|--------|-------------|
| MSP | Maximum Softmax Probability (Hendrycks & Gimpel, 2017) |
| Energy | Energy-based OOD score (Liu et al., 2020) |
| Mahalanobis | Mahalanobis Distance in feature space (Lee et al., 2018) |
| KNN | k-Nearest Neighbors in embedding space |
| MC Dropout | Monte Carlo Dropout for uncertainty estimation |

## Metrics

- **AUROC** — Area Under the ROC Curve (higher is better)
- **FPR@95TPR** — False Positive Rate at 95% True Positive Rate (lower is better)
- **AUPR** — Area Under Precision-Recall Curve

## Dataset

[CLINC150](https://github.com/clinc/oos-eval) — 150 intent classes, 22,500 in-domain samples + 1,200 OOD samples.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download data
python scripts/download_data.py

# 2. Fine-tune BERT
python src/train.py --model bert-base-uncased --epochs 5

# 3. Evaluate all OOD methods
python src/evaluate_all.py --model_path checkpoints/best_model

# 4. Run notebooks for analysis
jupyter notebook notebooks/
```

## Results

See `report/report.pdf` for full results and analysis.

## Repository Structure

```
clinc150-ood-detection/
├── README.md
├── requirements.txt
├── data/                    # Dataset files
├── src/
│   ├── dataset.py           # Data loading and preprocessing
│   ├── models.py            # Model definitions
│   ├── train.py             # Training script
│   ├── evaluate_all.py      # Full evaluation pipeline
│   └── methods/
│       ├── msp.py           # Maximum Softmax Probability
│       ├── energy.py        # Energy Score
│       ├── mahalanobis.py   # Mahalanobis Distance
│       ├── knn.py           # KNN-based detection
│       └── mc_dropout.py    # MC Dropout
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   └── 02_results.ipynb     # Results visualization
├── scripts/
│   └── download_data.py     # Data download script
└── report/
    └── report.pdf
```
