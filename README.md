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

## Order of the Running 

Run each module in order:

```bash
python preprocessing.py
python pd_model.py
python lgd_ead.py
python expected_loss.py
python monitoring.py
```

Each script is standalone and writes outputs to the `out/` directory.

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
