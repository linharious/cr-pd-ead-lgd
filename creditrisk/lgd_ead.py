"""
Credit Risk Modeling - LGD & EAD Models (Pandas Approach)

LGD (Loss Given Default) is modelled in two stages:
  Stage 1 – Logistic regression : P(recovery > 0)   → binary 0/1
  Stage 2 – Linear regression   : Expected recovery rate | recovery > 0
  Final LGD = 1 - clip(Stage1_binary × Stage2_predicted, 0, 1)

EAD (Exposure at Default) is modelled via:
  Linear regression on CCF = (funded_amnt - total_rec_prncp) / funded_amnt
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sns.set()
warnings.filterwarnings('ignore')


class LogisticRegressionWithPValues:
    """
    Wrapper around sklearn LogisticRegression that also computes
    two-tailed p-values for each coefficient via the Fisher Information Matrix.
    """

    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        self.model.fit(X, y)
        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X)))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)
        cramer_rao = np.linalg.pinv(F_ij)
        sigma_estimates = np.sqrt(np.abs(np.diagonal(cramer_rao)))
        sigma_estimates = np.where(sigma_estimates == 0, np.nan, sigma_estimates)
        z_scores = self.model.coef_[0] / sigma_estimates
        self.p_values = [
            stat.norm.sf(abs(z)) * 2 if not np.isnan(z) else np.nan
            for z in z_scores
        ]
        self.coef_      = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        return self.model.predict(np.array(X, dtype=float))

    def predict_proba(self, X):
        return self.model.predict_proba(np.array(X, dtype=float))


class LinearRegressionWithPValues(linear_model.LinearRegression):
    """
    Extension of sklearn LinearRegression that also computes
    t-statistics and two-tailed p-values for each coefficient.
    """

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        super().fit(X, y)
        n, k = X.shape
        sse = np.sum((self.predict(X) - y) ** 2) / float(n - k)
        se  = np.sqrt(np.diagonal(sse * np.linalg.pinv(np.dot(X.T, X))))
        self.t = self.coef_ / se
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), n - k)))
        return self


features_all = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:MORTGAGE', 'home_ownership:NONE', 'home_ownership:OTHER',
    'home_ownership:OWN', 'home_ownership:RENT',
    'verification_status:Not Verified', 'verification_status:Source Verified',
    'verification_status:Verified',
    'purpose:car', 'purpose:credit_card', 'purpose:debt_consolidation',
    'purpose:educational', 'purpose:home_improvement', 'purpose:house',
    'purpose:major_purchase', 'purpose:medical', 'purpose:moving',
    'purpose:other', 'purpose:renewable_energy', 'purpose:small_business',
    'purpose:vacation', 'purpose:wedding',
    'initial_list_status:f', 'initial_list_status:w',
    'term_int', 'emp_length_int', 'mths_since_issue_d', 'mths_since_earliest_cr_line',
    'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
    'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim',
]

ref_categories = [
    'grade:G',
    'home_ownership:RENT',
    'verification_status:Verified',
    'purpose:credit_card',
    'initial_list_status:f',
]

charged_off_statuses = [
    'Charged Off',
    'Does not meet the credit policy. Status:Charged Off',
]


def select_lgd_ead_features(X, model_features, feature_medians=None):
    """
    Keep only model features present in X and apply median imputation.

    Parameters
    ----------
    X               : input DataFrame
    model_features  : list of feature columns to keep
    feature_medians : Series of medians fitted on training data (or None)

    Returns a cleaned feature DataFrame.
    """
    cols  = [f for f in model_features if f in X.columns]
    X_sel = X[cols].copy()
    if feature_medians is not None:
        fill_vals = {c: feature_medians.get(c, 0) for c in X_sel.columns}
        X_sel = X_sel.fillna(fill_vals)
    return X_sel


def build_model_summary(X, coefs, p_values):
    """Build a feature summary DataFrame with coefficients and p-values."""
    return pd.DataFrame({
        'Feature name': list(X.columns),
        'Coefficients': np.array(coefs).flatten(),
        'p_values':     np.array(p_values).flatten(),
    })


def load_and_prepare(csv_path, features=None, ref_cats=None,
                     test_size=0.2, random_state=42):
    """
    Load the full preprocessed CSV, filter to charged-off loans,
    compute target variables, and split into train / test sets.

    Parameters
    ----------
    csv_path     : path to loan_data_2007_2014_preprocessed.csv
    features     : list of feature columns (default: features_all)
    ref_cats     : reference categories to exclude (default: ref_categories)
    test_size    : fraction reserved for test set
    random_state : random seed

    Returns
    -------
    X_train, X_test
    y_lgd1_train, y_lgd1_test   – binary recovery indicator
    y_lgd2_train, y_lgd2_test   – continuous recovery rate
    y_ead_train,  y_ead_test    – credit conversion factor
    feature_medians             – Series of train-set medians for imputation
    """
    features  = features  or features_all
    ref_cats  = ref_cats  or ref_categories
    model_features = [f for f in features if f not in ref_cats]

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    defaults = df[df['loan_status'].isin(charged_off_statuses)].copy()
    print(f"  Charged-off loans: {len(defaults):,}")

    defaults['mths_since_last_delinq'].fillna(0, inplace=True)
    defaults['mths_since_last_record'].fillna(0, inplace=True)

    defaults['recovery_rate'] = defaults['recoveries'] / defaults['funded_amnt']
    defaults['recovery_rate'] = np.clip(defaults['recovery_rate'], 0, 1)
    defaults['recovery_rate_0_1'] = np.where(defaults['recovery_rate'] == 0, 0, 1)
    defaults['CCF'] = (
        (defaults['funded_amnt'] - defaults['total_rec_prncp'])
        / defaults['funded_amnt']
    )

    available = [f for f in model_features if f in defaults.columns]
    missing   = [f for f in model_features if f not in defaults.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not found — skipped: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    X      = defaults[available]
    y_lgd1 = defaults['recovery_rate_0_1']
    y_lgd2 = defaults['recovery_rate']
    y_ead  = defaults['CCF']

    (X_train, X_test,
     y_lgd1_train, y_lgd1_test,
     y_lgd2_train, y_lgd2_test,
     y_ead_train,  y_ead_test) = train_test_split(
        X, y_lgd1, y_lgd2, y_ead,
        test_size=test_size,
        random_state=random_state,
    )

    feature_medians = X_train.median()
    X_train = X_train.fillna(feature_medians)
    X_test  = X_test.fillna(feature_medians)

    nan_remaining = X_train.isna().sum().sum()
    if nan_remaining > 0:
        print(f"  Warning: {nan_remaining} NaNs still present — filled with 0.")
        X_train = X_train.fillna(0)
        X_test  = X_test.fillna(0)

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return (X_train, X_test,
            y_lgd1_train, y_lgd1_test,
            y_lgd2_train, y_lgd2_test,
            y_ead_train,  y_ead_test,
            feature_medians)


def fit_lgd_ead(X_train, y_lgd1_train, y_lgd2_train, y_ead_train,
                features=None, ref_cats=None):
    """
    Fit LGD Stage 1, LGD Stage 2, and EAD models.

    Parameters
    ----------
    X_train       : input features (train set)
    y_lgd1_train  : binary recovery indicator
    y_lgd2_train  : continuous recovery rate
    y_ead_train   : credit conversion factor
    features      : list of all feature columns (default: features_all)
    ref_cats      : reference categories to drop (default: ref_categories)

    Returns
    -------
    lgd_st1          : fitted LogisticRegressionWithPValues
    lgd_st2          : fitted LinearRegressionWithPValues
    ead_model        : fitted LinearRegressionWithPValues
    summary_lgd_st1  : DataFrame
    summary_lgd_st2  : DataFrame
    summary_ead      : DataFrame
    model_features   : list of feature columns used
    feature_medians  : Series (pass back to select_lgd_ead_features at predict time)
    """
    features = features or features_all
    ref_cats = ref_cats or ref_categories
    model_features = [f for f in features if f not in ref_cats]

    feature_medians = X_train.median()
    X_tr = select_lgd_ead_features(X_train, model_features, feature_medians)

    print("[1/3] Fitting LGD Stage 1 (logistic regression) …")
    lgd_st1 = LogisticRegressionWithPValues()
    lgd_st1.fit(X_tr, y_lgd1_train)
    summary_lgd_st1 = build_model_summary(X_tr, lgd_st1.coef_[0], lgd_st1.p_values)

    print("[2/3] Fitting LGD Stage 2 (linear regression, recovery > 0 only) …")
    mask = (np.array(y_lgd1_train) == 1)
    X_lgd2_tr = X_tr.iloc[mask]
    y_lgd2_nonzero = np.array(y_lgd2_train)[mask]
    lgd_st2 = LinearRegressionWithPValues()
    lgd_st2.fit(X_lgd2_tr, y_lgd2_nonzero)
    summary_lgd_st2 = build_model_summary(X_lgd2_tr, lgd_st2.coef_, lgd_st2.p)

    print("[3/3] Fitting EAD model (linear regression on CCF) …")
    ead_model = LinearRegressionWithPValues()
    ead_model.fit(X_tr, y_ead_train)
    summary_ead = build_model_summary(X_tr, ead_model.coef_, ead_model.p)

    print("All models fitted.")
    return (lgd_st1, lgd_st2, ead_model,
            summary_lgd_st1, summary_lgd_st2, summary_ead,
            model_features, feature_medians)


def predict_lgd(lgd_st1, lgd_st2, X, model_features, feature_medians=None):
    """
    Predict LGD for each observation.
    LGD = 1 - clip(Stage1_binary × Stage2_prediction, 0, 1)
    """
    X_sel = select_lgd_ead_features(X, model_features, feature_medians)
    st1_binary    = lgd_st1.predict(X_sel)
    st2_pred      = lgd_st2.predict(X_sel)
    recovery_rate = np.clip(st1_binary * st2_pred, 0, 1)
    return 1 - recovery_rate


def predict_ead(ead_model, X, model_features, feature_medians=None, funded_amnt=None):
    """
    Predict EAD (or CCF if funded_amnt is not provided).

    Parameters
    ----------
    funded_amnt : optional array/Series; if given, returns CCF × funded_amnt
    """
    X_sel = select_lgd_ead_features(X, model_features, feature_medians)
    ccf = ead_model.predict(X_sel)
    if funded_amnt is not None:
        return np.clip(ccf, 0, 1) * np.array(funded_amnt)
    return ccf


def evaluate_lgd_ead(lgd_st1, lgd_st2, ead_model,
                     X_test, y_lgd1_test, y_lgd2_test, y_ead_test,
                     model_features, feature_medians=None, plot=True):
    """
    Evaluate all three models on the test set.

    Returns
    -------
    dict with keys: lgd_st1, lgd_st2, lgd_combined, ead
    """
    X_te = select_lgd_ead_features(X_test, model_features, feature_medians)
    results = {}

    print("\n--- LGD Stage 1 Evaluation ---")
    y_proba_st1  = lgd_st1.predict_proba(X_te)[:, 1]
    y_pred_st1   = (y_proba_st1 >= 0.5).astype(int)
    y_actual_st1 = np.array(y_lgd1_test)

    cm       = pd.crosstab(pd.Series(y_actual_st1, name='Actual'),
                           pd.Series(y_pred_st1,   name='Predicted'))
    accuracy = (y_actual_st1 == y_pred_st1).mean()
    auroc    = roc_auc_score(y_actual_st1, y_proba_st1)

    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  AUROC    : {auroc:.4f}")
    print("  Confusion Matrix:\n", cm)
    results['lgd_st1'] = {'accuracy': accuracy, 'auroc': auroc, 'confusion_matrix': cm}

    if plot:
        fpr, tpr, _ = roc_curve(y_actual_st1, y_proba_st1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'LGD Stage 1  (AUROC = {auroc:.4f})', color='steelblue')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('LGD Stage 1 — ROC Curve')
        ax.legend()
        plt.tight_layout()
        plt.show()

    print("\n--- LGD Stage 2 Evaluation (recovery > 0 only) ---")
    mask_te       = (np.array(y_lgd1_test) == 1)
    X_lgd2_te     = X_te.iloc[mask_te]
    y_lgd2_actual = np.array(y_lgd2_test)[mask_te]
    y_lgd2_pred   = lgd_st2.predict(X_lgd2_te)

    corr_st2 = np.corrcoef(y_lgd2_actual, y_lgd2_pred)[0, 1]
    print(f"  Correlation (actual vs predicted): {corr_st2:.4f}")
    results['lgd_st2'] = {'correlation': corr_st2}

    print("\n--- Combined LGD Evaluation ---")
    y_st1_binary  = lgd_st1.predict(X_te)
    y_st2_all     = lgd_st2.predict(X_te)
    y_hat_lgd     = np.clip(y_st1_binary * y_st2_all, 0, 1)
    y_actual_rate = np.array(y_lgd2_test)

    corr_lgd = np.corrcoef(y_actual_rate, y_hat_lgd)[0, 1]
    print(f"  Correlation (actual recovery rate vs combined LGD): {corr_lgd:.4f}")
    results['lgd_combined'] = {'correlation': corr_lgd}

    print("\n--- EAD Evaluation ---")
    y_ead_pred   = ead_model.predict(X_te)
    y_ead_actual = np.array(y_ead_test)

    corr_ead = np.corrcoef(y_ead_actual, y_ead_pred)[0, 1]
    print(f"  Correlation (actual vs predicted CCF): {corr_ead:.4f}")
    results['ead'] = {'correlation': corr_ead}

    return results


if __name__ == '__main__':
    os.makedirs('out/lgd_ead', exist_ok=True)

    (X_train, X_test,
     y_lgd1_train, y_lgd1_test,
     y_lgd2_train, y_lgd2_test,
     y_ead_train,  y_ead_test,
     feature_medians) = load_and_prepare('in/loan_data.csv')

    (lgd_st1, lgd_st2, ead_model,
     summary_lgd_st1, summary_lgd_st2, summary_ead,
     model_features, feature_medians) = fit_lgd_ead(
        X_train, y_lgd1_train, y_lgd2_train, y_ead_train
    )

    results = evaluate_lgd_ead(
        lgd_st1, lgd_st2, ead_model,
        X_test, y_lgd1_test, y_lgd2_test, y_ead_test,
        model_features, feature_medians
    )

    summary_lgd_st1.to_csv('out/lgd_ead/lgd_summary_stage1.csv', index=False)
    summary_lgd_st2.to_csv('out/lgd_ead/lgd_summary_stage2.csv', index=False)
    summary_ead.to_csv('out/lgd_ead/ead_summary.csv', index=False)
    pd.DataFrame({'feature': model_features}).to_csv('out/lgd_ead/lgd_ead_features.csv', index=False)
    feature_medians.to_csv('out/lgd_ead/feature_medians.csv', header=True)

    print("\nAll outputs saved to: out/lgd_ead/")
