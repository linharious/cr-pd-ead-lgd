"""
Credit Risk Modeling - PD Model
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score

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
        Cramer_Rao = np.linalg.pinv(F_ij)
        sigma_estimates = np.sqrt(np.abs(np.diagonal(Cramer_Rao)))
        sigma_estimates = np.where(sigma_estimates == 0, np.nan, sigma_estimates)
        z_scores = self.model.coef_[0] / sigma_estimates
        self.p_values = [stat.norm.sf(abs(z)) * 2 if not np.isnan(z) else np.nan
                         for z in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


all_features = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:OWN', 'home_ownership:MORTGAGE',
    'addr_state:ND_NE_IA_NV_FL_HI_AL', 'addr_state:NM_VA', 'addr_state:NY',
    'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA', 'addr_state:UT_KY_AZ_NJ',
    'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN', 'addr_state:GA_WA_OR',
    'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT',
    'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID',
    'verification_status:Not Verified', 'verification_status:Source Verified',
    'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house', 'purpose:credit_card',
    'purpose:debt_consolidation', 'purpose:oth__med__vacation',
    'purpose:major_purch__car__home_impr',
    'initial_list_status:f', 'initial_list_status:w',
    'term:36', 'term:60',
    'emp_length:0', 'emp_length:1', 'emp_length:2-4', 'emp_length:5-6',
    'emp_length:7-9', 'emp_length:10',
    'mths_since_issue_d:<38', 'mths_since_issue_d:38-39', 'mths_since_issue_d:40-41',
    'mths_since_issue_d:42-48', 'mths_since_issue_d:49-52', 'mths_since_issue_d:53-64',
    'mths_since_issue_d:65-84', 'mths_since_issue_d:>84',
    'int_rate:<9.548', 'int_rate:9.548-12.025', 'int_rate:12.025-15.74',
    'int_rate:15.74-20.281', 'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140', 'mths_since_earliest_cr_line:141-164',
    'mths_since_earliest_cr_line:165-247', 'mths_since_earliest_cr_line:248-270',
    'mths_since_earliest_cr_line:271-352', 'mths_since_earliest_cr_line:>352',
    'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'inq_last_6mths:>6',
    'acc_now_delinq:0', 'acc_now_delinq:>=1',
    'annual_inc:<20K', 'annual_inc:20K-30K', 'annual_inc:30K-40K', 'annual_inc:40K-50K',
    'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K', 'annual_inc:80K-90K',
    'annual_inc:90K-100K', 'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K',
    'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5', 'dti:10.5-16.1',
    'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35', 'dti:>35',
    'mths_since_last_delinq:Missing', 'mths_since_last_delinq:0-3',
    'mths_since_last_delinq:4-30', 'mths_since_last_delinq:31-56',
    'mths_since_last_delinq:>=57',
    'mths_since_last_record:Missing', 'mths_since_last_record:0-2',
    'mths_since_last_record:3-20', 'mths_since_last_record:21-31',
    'mths_since_last_record:32-80', 'mths_since_last_record:81-86',
    'mths_since_last_record:>86',
]

ref_categories = [
    'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY',
    'addr_state:ND_NE_IA_NV_FL_HI_AL',
    'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house',
    'initial_list_status:f',
    'term:60',
    'emp_length:0',
    'mths_since_issue_d:>84',
    'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140',
    'inq_last_6mths:>6',
    'acc_now_delinq:0',
    'annual_inc:<20K',
    'dti:>35',
    'mths_since_last_delinq:0-3',
    'mths_since_last_record:0-2',
]


def select_features(X, model_features):
    """Select and cast model features from X, excluding reference categories."""
    missing = [c for c in model_features if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")
    return X[model_features].astype(float)


def build_summary_table(reg, inputs):
    """Build a coefficient + p-value DataFrame from a fitted model."""
    feature_name = inputs.columns.values
    tbl = pd.DataFrame({'Feature name': feature_name})
    tbl['Coefficients'] = np.transpose(reg.coef_)
    tbl.index = tbl.index + 1
    tbl.loc[0] = ['Intercept', reg.intercept_[0]]
    tbl = tbl.sort_index()
    tbl['p_values'] = np.append(np.nan, np.array(reg.p_values))
    return tbl


def build_scorecard(summary_table, ref_cats=None, min_score=300, max_score=850):
    """
    Build the full scorecard with final scores per feature.

    Returns
    -------
    scorecard    : DataFrame with 'Feature name' and 'Score - Final' columns
    min_sum_coef : float  (used for back-calculating PD from score)
    max_sum_coef : float
    """
    ref_cats = ref_cats or ref_categories

    df_ref = pd.DataFrame(ref_cats, columns=['Feature name'])
    df_ref['Coefficients'] = 0
    df_ref['p_values'] = np.nan

    sc = pd.concat([summary_table, df_ref]).reset_index(drop=True)
    sc['Original feature name'] = sc['Feature name'].str.split(':').str[0]

    min_sum_coef = sc.groupby('Original feature name')['Coefficients'].min().sum()
    max_sum_coef = sc.groupby('Original feature name')['Coefficients'].max().sum()

    scale = (max_score - min_score) / (max_sum_coef - min_sum_coef)
    sc['Score - Calculation'] = sc['Coefficients'] * scale

    intercept_idx = sc[sc['Feature name'] == 'Intercept'].index[0]
    sc.loc[intercept_idx, 'Score - Calculation'] = (
        (sc.loc[intercept_idx, 'Coefficients'] - min_sum_coef)
        / (max_sum_coef - min_sum_coef)
        * (max_score - min_score)
        + min_score
    )

    sc['Score - Preliminary'] = sc['Score - Calculation'].round()
    sc['Difference'] = sc['Score - Preliminary'] - sc['Score - Calculation']
    sc['Score - Final'] = sc['Score - Preliminary'].copy()

    max_sum = sc.groupby('Original feature name')['Score - Final'].max().sum()
    if max_sum != max_score:
        candidates = sc[
            (sc['Feature name'] != 'Intercept') &
            (~sc['Feature name'].isin(ref_cats))
        ]
        adjust_idx = candidates['Difference'].abs().idxmax()
        sc.loc[adjust_idx, 'Score - Final'] -= (max_sum - max_score)

    return sc, min_sum_coef, max_sum_coef


def fit_pd_model(X, y, features=None, ref_cats=None, min_score=300, max_score=850):
    """
    Fit the logistic regression and build the scorecard.

    Parameters
    ----------
    X         : preprocessed input DataFrame
    y         : binary target Series (good_bad)
    features  : list of all feature columns (default: all_features)
    ref_cats  : list of reference category columns (default: ref_categories)
    min_score : minimum credit score (default 300)
    max_score : maximum credit score (default 850)

    Returns
    -------
    reg           : fitted LogisticRegressionWithPValues
    summary_table : DataFrame with coefficients and p-values
    scorecard     : DataFrame with final scores per feature
    min_sum_coef  : float
    max_sum_coef  : float
    model_features: list of features used in the model (ref cats excluded)
    """
    features = features or all_features
    ref_cats = ref_cats or ref_categories
    model_features = [f for f in features if f not in ref_cats]

    inputs = select_features(X, model_features)

    reg = LogisticRegressionWithPValues()
    reg.fit(inputs, y)

    summary_table = build_summary_table(reg, inputs)
    scorecard, min_sum_coef, max_sum_coef = build_scorecard(
        summary_table, ref_cats, min_score, max_score
    )

    return reg, summary_table, scorecard, min_sum_coef, max_sum_coef, model_features


def predict_proba_pd(reg, X, model_features):
    """Return probability of being 'good' (class 1) for each row."""
    inputs = select_features(X, model_features)
    return reg.predict_proba(inputs)[:, 1]


def score_applicants(X, scorecard, features, min_sum_coef, max_sum_coef,
                     min_score=300, max_score=850):
    """
    Calculate credit scores (300–850) and PD estimates for each applicant.

    Returns a DataFrame with columns: ['credit_score', 'pd_estimate']
    """
    avail = [c for c in features if c in X.columns]
    X_w_intercept = X[avail].copy()
    X_w_intercept.insert(0, 'Intercept', 1)

    scorecard_scores = scorecard.set_index('Feature name')['Score - Final']
    cols = [c for c in scorecard_scores.index if c in X_w_intercept.columns]
    y_scores = X_w_intercept[cols].dot(scorecard_scores[cols])

    sum_coef = np.array(
        (y_scores - min_score)
        / (max_score - min_score)
        * (max_sum_coef - min_sum_coef)
        + min_sum_coef,
        dtype=float
    ).flatten()
    pd_from_score = np.exp(sum_coef) / (np.exp(sum_coef) + 1)

    return pd.DataFrame({
        'credit_score': np.array(y_scores, dtype=float).flatten(),
        'pd_estimate':  1 - pd_from_score,
    }, index=X.index)


def build_cutoffs_table(fpr, tpr, thresholds, proba_series, min_sum_coef, max_sum_coef,
                        min_score=300, max_score=850):
    """Build a threshold/score/approval-rate table from ROC curve data."""
    df_cut = pd.concat([
        pd.DataFrame(thresholds, columns=['thresholds']),
        pd.DataFrame(fpr,        columns=['fpr']),
        pd.DataFrame(tpr,        columns=['tpr']),
    ], axis=1)

    df_cut.loc[0, 'thresholds'] = 1 - 1 / np.power(10, 16)

    df_cut['Score'] = (
        (
            np.log(df_cut['thresholds'] / (1 - df_cut['thresholds']))
            - min_sum_coef
        )
        * ((max_score - min_score) / (max_sum_coef - min_sum_coef))
        + min_score
    ).round()
    df_cut.loc[0, 'Score'] = max_score

    df_cut['N Approved']     = df_cut['thresholds'].apply(
        lambda p: np.where(proba_series >= p, 1, 0).sum()
    )
    df_cut['N Rejected']     = len(proba_series) - df_cut['N Approved']
    df_cut['Approval Rate']  = df_cut['N Approved'] / len(proba_series)
    df_cut['Rejection Rate'] = 1 - df_cut['Approval Rate']
    return df_cut


def evaluate_pd_model(reg, X, y, model_features, min_sum_coef, max_sum_coef,
                      threshold=0.9, plot=True, min_score=300, max_score=850):
    """
    Evaluate model on a dataset. Prints and returns key metrics.

    Parameters
    ----------
    reg           : fitted LogisticRegressionWithPValues
    X             : preprocessed input DataFrame
    y             : binary target Series
    model_features: list of features used in the model (from fit_pd_model)
    min_sum_coef  : float (from fit_pd_model)
    max_sum_coef  : float (from fit_pd_model)
    threshold     : classification threshold (default 0.9)
    plot          : whether to show ROC, Gini, KS charts

    Returns
    -------
    dict with keys: auroc, gini, ks, accuracy, confusion_matrix,
                    confusion_matrix_rates, df_cutoffs
    """
    y_proba = predict_proba_pd(reg, X, model_features)

    df = pd.DataFrame({'actual': y.values, 'proba': y_proba})
    df['predicted'] = np.where(df['proba'] > threshold, 1, 0)

    cm       = pd.crosstab(df['actual'], df['predicted'],
                           rownames=['Actual'], colnames=['Predicted'])
    cm_rates = cm / len(df)
    accuracy = cm_rates.iloc[0, 0] + cm_rates.iloc[1, 1]

    fpr, tpr, thresholds_arr = roc_curve(df['actual'], df['proba'])
    auroc = roc_auc_score(df['actual'], df['proba'])
    gini  = auroc * 2 - 1

    df_sorted = df.sort_values('proba').reset_index(drop=True)
    df_sorted['cum_pop']  = (df_sorted.index + 1) / len(df_sorted)
    df_sorted['cum_good'] = df_sorted['actual'].cumsum() / df_sorted['actual'].sum()
    df_sorted['cum_bad']  = (
        (df_sorted.index + 1 - df_sorted['actual'].cumsum())
        / (len(df_sorted) - df_sorted['actual'].sum())
    )
    ks = (df_sorted['cum_bad'] - df_sorted['cum_good']).max()

    df_cutoffs = build_cutoffs_table(
        fpr, tpr, thresholds_arr, df['proba'],
        min_sum_coef, max_sum_coef, min_score, max_score
    )

    print("=" * 50)
    print(f"  AUROC    : {auroc:.4f}")
    print(f"  Gini     : {gini:.4f}")
    print(f"  KS       : {ks:.4f}")
    print(f"  Accuracy : {accuracy:.4f}  (threshold={threshold})")
    print("=" * 50)
    print("\nConfusion Matrix (rates):")
    print(cm_rates.round(4))

    if plot:
        plot_roc(fpr, tpr)
        plot_gini(df_sorted)
        plot_ks(df_sorted)

    return dict(
        auroc=auroc, gini=gini, ks=ks, accuracy=accuracy,
        confusion_matrix=cm, confusion_matrix_rates=cm_rates,
        df_cutoffs=df_cutoffs,
    )


def plot_roc(fpr, tpr):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Model')
    plt.plot(fpr, fpr, linestyle='--', color='k', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gini(df_sorted):
    plt.figure(figsize=(8, 6))
    plt.plot(df_sorted['cum_pop'], df_sorted['cum_bad'], label='Model')
    plt.plot(df_sorted['cum_pop'], df_sorted['cum_pop'],
             linestyle='--', color='k', label='Random')
    plt.xlabel('Cumulative % Population')
    plt.ylabel('Cumulative % Bad')
    plt.title('Gini (CAP Curve)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ks(df_sorted):
    plt.figure(figsize=(8, 6))
    plt.plot(df_sorted['proba'], df_sorted['cum_bad'],  color='r', label='% Bad')
    plt.plot(df_sorted['proba'], df_sorted['cum_good'], color='b', label='% Good')
    plt.xlabel('Estimated Probability of Being Good')
    plt.ylabel('Cumulative %')
    plt.title('Kolmogorov-Smirnov')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    os.makedirs('out/pd', exist_ok=True)

    loan_data_inputs_train  = pd.read_csv('out/prepr/loan_data_inputs_train.csv',  index_col=0)
    loan_data_inputs_test   = pd.read_csv('out/prepr/loan_data_inputs_test.csv',   index_col=0)
    loan_data_targets_train = pd.read_csv('out/prepr/loan_data_targets_train.csv', index_col=0).iloc[:, 0]
    loan_data_targets_test  = pd.read_csv('out/prepr/loan_data_targets_test.csv',  index_col=0).iloc[:, 0]
    loan_data_targets_train = loan_data_targets_train.loc[loan_data_inputs_train.index]
    loan_data_targets_test  = loan_data_targets_test.loc[loan_data_inputs_test.index]

    reg, summary_table, scorecard, min_sum_coef, max_sum_coef, model_features = fit_pd_model(
        loan_data_inputs_train, loan_data_targets_train
    )

    print("\n--- Summary Table (first 10 rows) ---")
    print(summary_table.head(10))

    print("\n--- Scorecard (first 10 rows) ---")
    print(scorecard[['Feature name', 'Coefficients', 'Score - Final']].head(10))

    print("\n--- Test Set Evaluation ---")
    results = evaluate_pd_model(
        reg, loan_data_inputs_test, loan_data_targets_test,
        model_features, min_sum_coef, max_sum_coef, threshold=0.9
    )

    applicant_scores = score_applicants(
        loan_data_inputs_test, scorecard, all_features, min_sum_coef, max_sum_coef
    )
    print("\n--- Credit Scores (first 5) ---")
    print(applicant_scores.head())

    scorecard.to_csv('out/pd/df_scorecard.csv')
    applicant_scores.to_csv('out/pd/applicant_scores.csv')
    results['df_cutoffs'].to_csv('out/pd/df_cutoffs.csv')

    pd.DataFrame({
        'min_sum_coef': [min_sum_coef],
        'max_sum_coef': [max_sum_coef],
    }).to_csv('out/pd/pd_model_params.csv', index=False)

    pd.DataFrame({'feature': model_features}).to_csv('out/pd/pd_model_features.csv', index=False)

    print("\nAll outputs saved to: out/pd/")