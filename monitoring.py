"""
Credit Risk Modeling - Model Monitoring (Pandas Approach)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set()


psi_stable  = 0.10
psi_monitor = 0.25

score_bands = [
    ('Score:300-350',  300, 350),
    ('Score:350-400',  350, 400),
    ('Score:400-450',  400, 450),
    ('Score:450-500',  450, 500),
    ('Score:500-550',  500, 550),
    ('Score:550-600',  550, 600),
    ('Score:600-650',  600, 650),
    ('Score:650-700',  650, 700),
    ('Score:700-750',  700, 750),
    ('Score:750-800',  750, 800),
    ('Score:800-850',  800, 851),
]

def _psi_contribution(p_expected, p_actual):
    """PSI contribution for a single bin. Returns 0 if either proportion is 0."""
    if p_expected == 0 or p_actual == 0:
        return 0.0
    return (p_actual - p_expected) * np.log(p_actual / p_expected)


def compute_psi(expected_props, actual_props):
    """
    Compute PSI given two Series of proportions indexed by bin name.
    Only bins present in both series are included.
    """
    common = expected_props.index.intersection(actual_props.index)
    return sum(_psi_contribution(expected_props[b], actual_props[b]) for b in common)


def psi_status(psi_val):
    if psi_val < psi_stable:
        return 'Stable'
    elif psi_val < psi_monitor:
        return 'Monitor'
    else:
        return 'Investigate'


def add_score_bands(df, score_col='Score'):
    """Append binary score-band columns to a DataFrame."""
    df = df.copy()
    for col, lo, hi in score_bands:
        df[col] = np.where((df[score_col] >= lo) & (df[score_col] < hi), 1, 0)
    return df


def preprocess_new_data(csv_path, reference_date, expected_cols):
    """
    Parameters
    ----------
    csv_path       : path to the new raw CSV
    reference_date : date string used for mths_since_* calculations
    expected_cols  : list of columns from fit_preprocessor (to align new data)

    Returns
    -------
    X : preprocessed feature DataFrame
    y : good_bad Series
    """
    from preprocessing import (general_preprocessing, make_dummies,
                                fill_missing, create_good_bad,
                                apply_feature_engineering, align_columns)

    raw = pd.read_csv(csv_path)
    df  = general_preprocessing(raw, reference_date)
    df  = make_dummies(df)
    df  = fill_missing(df)
    df  = create_good_bad(df)

    X = df.drop('good_bad', axis=1)
    y = df['good_bad']

    X = apply_feature_engineering(X)
    X = align_columns(X, expected_cols)

    return X, y


def prepare_for_scoring(X, scorecard):
    """
    Add Intercept column and keep only features present in the scorecard.

    Returns a DataFrame ready for score computation.
    """
    out = X.copy()
    if 'Intercept' not in out.columns:
        out.insert(0, 'Intercept', 1)
    feature_names = scorecard['Feature name'].values
    available = [f for f in feature_names if f in out.columns]
    return out[available]


def compute_scores(X_with_intercept, scorecard):
    """Dot product of features with scorecard 'Score - Final' values."""
    sc_indexed = scorecard.set_index('Feature name')
    scores_vec = sc_indexed.loc[X_with_intercept.columns, 'Score - Final'].values.reshape(-1, 1)
    result = X_with_intercept.dot(scores_vec)
    return result.iloc[:, 0]


def build_psi_table(props_train, props_new):
    """Build a full bin-level PSI contribution DataFrame."""
    psi = pd.concat([props_train, props_new], axis=1).reset_index()
    psi.columns = ['index', 'Proportions_Train', 'Proportions_New']
    psi['Original feature name'] = psi['index'].str.split(':').str[0]
    psi = psi[['index', 'Original feature name', 'Proportions_Train', 'Proportions_New']]
    psi = psi[~psi['index'].isin(['Intercept', 'Score'])].copy()
    psi['Contribution'] = psi.apply(
        lambda r: _psi_contribution(r['Proportions_Train'], r['Proportions_New']),
        axis=1,
    )
    return psi.reset_index(drop=True)


def print_psi_report(psi_by_feature, psi_score):
    """Print a formatted PSI report to stdout."""
    print(f"{'Feature':<45}  {'PSI':>8}  Status")
    print("-" * 70)
    for feat, psi_val in psi_by_feature.items():
        print(f"{feat:<45}  {psi_val:>8.4f}  {psi_status(psi_val)}")
    print("-" * 70)
    print(f"{'Score (overall)':<45}  {psi_score:>8.4f}  {psi_status(psi_score)}")
    print()
    print("Legend:  Stable (<0.10)    Monitor (0.10–0.25)   Investigate (>0.25)")


def plot_psi(psi_by_feature, title='PSI by Feature'):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [
        '#d62728' if v >= psi_monitor else
        '#ff7f0e' if v >= psi_stable  else
        '#2ca02c'
        for v in psi_by_feature.values
    ]
    ax.bar(psi_by_feature.index, psi_by_feature.values, color=colors)
    ax.axhline(psi_stable,  linestyle='--', color='#ff7f0e', linewidth=1,
               label=f'Monitor threshold ({psi_stable})')
    ax.axhline(psi_monitor, linestyle='--', color='#d62728', linewidth=1,
               label=f'Investigate threshold ({psi_monitor})')
    ax.set_title(title)
    ax.set_ylabel('PSI')
    ax.set_xticklabels(psi_by_feature.index, rotation=45, ha='right', fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_score_distribution(scores_train, scores_new):
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = range(300, 860, 50)
    ax.hist(scores_train, bins=bins, alpha=0.6, label='Train (expected)', color='steelblue')
    ax.hist(scores_new,   bins=bins, alpha=0.6, label='New (actual)',     color='darkorange')
    ax.set_title('Score Distribution: Train vs New Data')
    ax.set_xlabel('Credit Score')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_monitoring(
    train_inputs_path,
    new_raw_csv_path,
    scorecard,
    expected_cols,
    reference_date='2018-12-01',
    out_dir='.',
    plot=True,
):
    """
    Full monitoring run: preprocess new data, compute PSI, save outputs.

    Parameters
    ----------
    train_inputs_path : path to loan_data_inputs_train.csv (already preprocessed)
    new_raw_csv_path  : path to the new raw CSV (e.g. loan_data_2015.csv)
    scorecard         : scorecard DataFrame (from build_scorecard or loaded CSV)
    expected_cols     : list of columns from fit_preprocessor (to align new data)
    reference_date    : date string for mths_since_* on new data
    out_dir           : directory to save output CSVs
    plot              : whether to display PSI charts

    Returns
    -------
    dict with keys: psi_by_feature, psi_score, psi_detail, new_inputs, new_targets
    """
    print("=" * 60)
    print("  MODEL MONITORING REPORT")
    print("=" * 60)

    print("\n[1/5] Loading training inputs …")
    train_inputs = pd.read_csv(train_inputs_path, index_col=0)
    train_scored = prepare_for_scoring(train_inputs, scorecard)
    train_scored = train_scored.copy()
    train_scored['Score'] = compute_scores(train_scored, scorecard).values
    train_scored = add_score_bands(train_scored)

    print("[2/5] Preprocessing new data …")
    new_inputs, new_targets = preprocess_new_data(new_raw_csv_path, reference_date, expected_cols)
    new_scored = prepare_for_scoring(new_inputs, scorecard)
    new_scored = new_scored.copy()
    new_scored['Score'] = compute_scores(new_scored, scorecard).values
    new_scored = add_score_bands(new_scored)

    new_inputs.to_csv(os.path.join(out_dir, 'loan_data_inputs_new.csv'))
    new_targets.to_csv(os.path.join(out_dir, 'loan_data_targets_new.csv'))

    print("[3/5] Computing proportions …")
    props_train = train_scored.sum() / train_scored.shape[0]
    props_new   = new_scored.sum()   / new_scored.shape[0]

    print("[4/5] Calculating PSI …")
    psi_detail = build_psi_table(props_train, props_new)
    psi_detail.to_csv(os.path.join(out_dir, 'psi_detail.csv'), index=False)

    psi_by_feature = (
        psi_detail[psi_detail['index'] != 'Score']
        .groupby('Original feature name')['Contribution']
        .sum()
        .sort_values(ascending=False)
        .rename('PSI')
    )
    psi_by_feature.to_csv(os.path.join(out_dir, 'psi_by_feature.csv'))

    psi_score = psi_detail[
        psi_detail['Original feature name'] == 'Score'
    ]['Contribution'].sum()

    print("[5/5] Generating report …\n")
    print_psi_report(psi_by_feature, psi_score)

    if plot:
        plot_psi(psi_by_feature, title='PSI by Feature')
        plot_score_distribution(train_scored['Score'], new_scored['Score'])

    return dict(
        psi_by_feature=psi_by_feature,
        psi_score=psi_score,
        psi_detail=psi_detail,
        new_inputs=new_inputs,
        new_targets=new_targets,
    )


def psi_from_csvs(
    train_inputs_path,
    new_inputs_path,
    scorecard_path,
    out_dir='.',
    plot=True,
):
    """
    Calculate PSI directly from pre-processed CSV files and a scorecard CSV,
    without needing the preprocessor object.

    Parameters
    ----------
    train_inputs_path : path to loan_data_inputs_train.csv
    new_inputs_path   : path to loan_data_inputs_new.csv (already preprocessed)
    scorecard_path    : path to df_scorecard.csv
    out_dir           : directory to write psi_detail.csv
    plot              : show charts

    Returns
    -------
    DataFrame with PSI per feature
    """
    train = pd.read_csv(train_inputs_path, index_col=0)
    new   = pd.read_csv(new_inputs_path,   index_col=0)
    sc    = pd.read_csv(scorecard_path)

    feature_cols = sc['Feature name'].values

    def _score(df):
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).copy()
        df.insert(0, 'Intercept', 1)
        cols = [c for c in feature_cols if c in df.columns]
        scores_vec = sc.set_index('Feature name').loc[cols, 'Score - Final'].values.reshape(-1, 1)
        df['Score'] = df[cols].astype(float).dot(scores_vec)
        df = add_score_bands(df)
        keep = ['Intercept'] + list(cols) + ['Score'] + [b for b, _, _ in score_bands]
        return df[[c for c in keep if c in df.columns]]

    train_scored = _score(train)
    new_scored   = _score(new)

    props_train = train_scored.sum() / train_scored.shape[0]
    props_new   = new_scored.sum()   / new_scored.shape[0]

    psi_tbl = build_psi_table(props_train, props_new)
    psi_by_feature = (
        psi_tbl[psi_tbl['index'] != 'Score']
        .groupby('Original feature name')['Contribution']
        .sum()
        .sort_values(ascending=False)
        .rename('PSI')
    )

    psi_tbl.to_csv(os.path.join(out_dir, 'psi_detail.csv'), index=False)
    psi_by_feature.to_csv(os.path.join(out_dir, 'psi_by_feature.csv'))

    if plot:
        plot_psi(psi_by_feature)

    return psi_by_feature


if __name__ == '__main__':
    from preprocessing import load_and_split, fit_preprocessor, transform

    os.makedirs('out/monitor', exist_ok=True)

    print("Fitting preprocessor on training data …")
    X_train, X_test, y_train, y_test = load_and_split('in/loan_data.csv')
    expected_cols = fit_preprocessor(X_train)

    print("Loading scorecard …")
    scorecard = pd.read_csv('out/pd/df_scorecard.csv')

    report = run_monitoring(
        train_inputs_path = 'out/prepr/loan_data_inputs_train.csv',
        new_raw_csv_path  = 'in/loan_data_new.csv',
        scorecard         = scorecard,
        expected_cols     = expected_cols,
        reference_date    = '2018-12-01',
        out_dir           = 'out/monitor',
        plot              = True,
    )

    print("\nPSI by feature:")
    print(report['psi_by_feature'].to_string())
    print(f"\nScore PSI : {report['psi_score']:.4f}")
