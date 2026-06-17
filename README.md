# Credit Risk Modeling

A modular credit risk modeling system that implements the Basel framework for loan portfolio risk assessment. Computes PD, LGD, EAD, and Expected Loss, with model monitoring via Population Stability Index (PSI).

## Modules

| File | Purpose |
|------|---------|
| `preprocessing.py` | Data cleaning, feature engineering, WoE binning |
| `pd_model.py` | Probability of Default logistic regression model and credit scorecard |
| `lgd_ead.py` | Loss Given Default and Exposure at Default models |
| `expected_loss.py` | Combines PD × LGD × EAD to compute portfolio-level Expected Loss |
| `monitoring.py` | PSI-based model drift detection on new data |

## Input Data

Place the following files in `in/`:

- `loan_data.csv` — historical loan data used for training
- `loan_data_new.csv` — new period data used for monitoring

## Setup Before Running

### 1. Create the required folders

Create the following directory structure in your project root:
project/
├── in/
└── out/
├── pd/
├── prepr/
├── monitor/
├── lgd_ead/
└── el/

You can do this in one command:
```bash
mkdir -p in out/pd out/prepr out/monitor out/lgd_ead out/el
```

### 2. Download the input data

Download the two CSV files and place them in the `in/` folder:

- [`loan_data.csv`](https://drive.google.com/file/d/1jRY-0Ef_rZNbckqImtnD0vzx9axB9PDx/view?usp=share_link) — historical loan data for training
- [`loan_data_new.csv`](https://drive.google.com/file/d/12Pz0ff_3ACEkKL_PTsbsJCB3O2Ee_SGL/view?usp=share_link) — new period data for monitoring

After downloading, your `in/` folder should look like:
in/
├── loan_data.csv
└── loan_data_new.csv

## Project layout

The modelling code is now an importable package:

```
creditrisk/   # preprocessing, pd_model, lgd_ead, expected_loss, monitoring
training/     # train.py — fits models and saves an artifact bundle
models/       # generated: model_bundle.joblib + manifest.json per version
```

Install it once (editable):

```bash
pip install -e .
```

## Training

Fit all models and save a reusable bundle under `models/<version>/`:

```bash
python training/train.py --data in/loan_data.csv
```

Load it back for scoring:

```python
from creditrisk.artifacts import load_bundle, score_pd
bundle, meta = load_bundle()          # latest version
scores = score_pd(bundle, raw_applicant_df)
```

## Running the stages individually

Each module is still standalone and writes outputs to `out/`:

```bash
python -m creditrisk.preprocessing
python -m creditrisk.pd_model
python -m creditrisk.lgd_ead
python -m creditrisk.expected_loss
python -m creditrisk.monitoring
```

## Outputs

| Directory | Contents |
|-----------|---------|
| `out/prepr/` | Preprocessed train/test features and targets |
| `out/pd/` | Scorecard, model coefficients, applicant scores, ROC cutoffs |
| `out/lgd_ead/` | LGD/EAD model summaries, feature list, feature medians |
| `out/el/` | Per-loan expected loss values, portfolio summary |
| `out/monitor/` | PSI scores by feature, score distributions, stability report |

## Models

### PD — Probability of Default
- Logistic regression with custom p-value calculation via Fisher Information Matrix
- Outputs a credit scorecard scaled to the 300–850 range
- Evaluated with ROC-AUC, Gini coefficient, and KS statistic

### LGD — Loss Given Default
- Two-stage model:
  1. Logistic regression: P(recovery > 0)
  2. Linear regression: expected recovery rate given recovery > 0
- Target: recovery rate on charged-off loans

### EAD — Exposure at Default
- Linear regression on Credit Conversion Factor (CCF)
- Target: drawn amount relative to credit limit at time of default

### Expected Loss
- EL = PD × LGD × EAD
- Computed per loan, aggregated at portfolio level

## Monitoring

PSI thresholds used to classify model stability:

| PSI | Status |
|-----|--------|
| < 0.10 | Stable |
| 0.10 – 0.25 | Monitor |
| > 0.25 | Investigate |

## Dependencies

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
```
