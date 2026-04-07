"""
Credit Risk Modeling - Preprocessing
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def woe_discrete(df, col, target_series):
    """Compute Weight of Evidence table for a discrete variable."""
    tmp = pd.concat([df[col], target_series], axis=1)
    cnt = tmp.groupby(col, as_index=False)[target_series.name].count()
    avg = tmp.groupby(col, as_index=False)[target_series.name].mean()
    out = pd.concat([cnt, avg], axis=1).iloc[:, [0, 1, 3]]
    out.columns = [col, 'n_obs', 'prop_good']
    _add_woe_columns(out)
    out = out.sort_values('WoE').reset_index(drop=True)
    out['diff_prop_good'] = out['prop_good'].diff().abs()
    out['diff_WoE']       = out['WoE'].diff().abs()
    return out


def woe_ordered_continuous(df, col, target_series):
    """Compute Weight of Evidence table for an ordered discrete or continuous variable."""
    tmp = pd.concat([df[col], target_series], axis=1)
    cnt = tmp.groupby(col, as_index=False)[target_series.name].count()
    avg = tmp.groupby(col, as_index=False)[target_series.name].mean()
    out = pd.concat([cnt, avg], axis=1).iloc[:, [0, 1, 3]]
    out.columns = [col, 'n_obs', 'prop_good']
    _add_woe_columns(out)
    out['diff_prop_good'] = out['prop_good'].diff().abs()
    out['diff_WoE']       = out['WoE'].diff().abs()
    return out


def _add_woe_columns(df):
    """Add n_good, n_bad, prop_n_good, prop_n_bad, WoE, IV in-place."""
    df['prop_n_obs']  = df['n_obs'] / df['n_obs'].sum()
    df['n_good']      = df['prop_good'] * df['n_obs']
    df['n_bad']       = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad']  = df['n_bad']  / df['n_bad'].sum()
    df['WoE']         = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df['IV']          = ((df['prop_n_good'] - df['prop_n_bad']) * df['WoE']).sum()


def plot_by_woe(df_woe, rotation=0):
    """WoE plot."""
    x = df_woe.iloc[:, 0].astype(str).values
    y = df_woe['WoE'].values
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(f'Weight of Evidence by {df_woe.columns[0]}')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


# Constants
_discrete_cols = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'loan_status', 'purpose', 'addr_state', 'initial_list_status',
]

_bad_statuses = [
    'Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off',
    'Late (31-120 days)',
]

_required_raw_cols = [
    'emp_length', 'earliest_cr_line', 'term', 'issue_d',
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'loan_status', 'purpose', 'addr_state', 'initial_list_status',
    'total_rev_hi_lim', 'funded_amnt', 'annual_inc',
    'mths_since_last_delinq', 'mths_since_last_record',
    'int_rate', 'dti', 'delinq_2yrs', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq',
]


def general_preprocessing(df, reference_date='2017-12-01'):
    """
    Clean raw loan data and compute date-derived columns.

    Parameters
    ----------
    df             : raw loan DataFrame
    reference_date : string or datetime used to compute months-since columns

    Returns a new DataFrame (does not modify in place).
    """
    reference_date = pd.to_datetime(reference_date)
    df = df.copy()

    df['emp_length_int'] = (
        df['emp_length']
        .str.replace(r'\+ years', '', regex=True)
        .str.replace('< 1 year', '0')
        .str.replace('n/a', '0')
        .str.replace(' years', '')
        .str.replace(' year', '')
    )
    df['emp_length_int'] = pd.to_numeric(df['emp_length_int'])

    df['earliest_cr_line_date'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
    df['mths_since_earliest_cr_line'] = round(
        pd.to_numeric(
            (reference_date - df['earliest_cr_line_date']) / np.timedelta64(1, 'm')
        )
    )
    mask = df['mths_since_earliest_cr_line'] < 0
    df.loc[mask, 'mths_since_earliest_cr_line'] = df['mths_since_earliest_cr_line'].max()

    df['term_int'] = pd.to_numeric(df['term'].str.replace(' months', ''))

    df['issue_d_date'] = pd.to_datetime(df['issue_d'], format='%b-%y')
    df['mths_since_issue_d'] = round(
        pd.to_numeric(
            (reference_date - df['issue_d_date']) / np.timedelta64(1, 'm')
        )
    )

    return df


def make_dummies(df):
    """One-hot encode categorical columns."""
    dummies = [
        pd.get_dummies(df[col], prefix=col, prefix_sep=':')
        for col in _discrete_cols
    ]
    return pd.concat([df] + dummies, axis=1)


def fill_missing(df):
    """Fill missing values using domain-appropriate rules."""
    df = df.copy()
    df['total_rev_hi_lim'].fillna(df['funded_amnt'], inplace=True)
    df['annual_inc'].fillna(df['annual_inc'].mean(), inplace=True)
    for col in ['mths_since_earliest_cr_line', 'acc_now_delinq', 'total_acc',
                'pub_rec', 'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'emp_length_int']:
        df[col].fillna(0, inplace=True)
    return df


def create_good_bad(df):
    """Add good_bad column: 0 = bad (defaulted), 1 = good."""
    df = df.copy()
    df['good_bad'] = np.where(df['loan_status'].isin(_bad_statuses), 0, 1)
    return df


def _check_cols(df, cols, fill=0):
    """Add missing dummy columns with a default fill value."""
    for col in cols:
        if col not in df.columns:
            df[col] = fill


def fe_home_ownership(df):
    _check_cols(df, [
        'home_ownership:RENT', 'home_ownership:OTHER',
        'home_ownership:NONE', 'home_ownership:ANY',
    ])
    df['home_ownership:RENT_OTHER_NONE_ANY'] = (
        df['home_ownership:RENT'] + df['home_ownership:OTHER'] +
        df['home_ownership:NONE'] + df['home_ownership:ANY']
    )
    return df


def fe_addr_state(df):
    _check_cols(df, [f'addr_state:{s}' for s in [
        'ND','NE','IA','NV','FL','HI','AL','NM','VA','NY',
        'OK','TN','MO','LA','MD','NC','CA','UT','KY','AZ','NJ',
        'AR','MI','PA','OH','MN','RI','MA','DE','SD','IN',
        'GA','WA','OR','WI','MT','TX','IL','CT','KS','SC',
        'CO','VT','AK','MS','WV','NH','WY','DC','ME','ID',
    ]])
    combos = {
        'addr_state:ND_NE_IA_NV_FL_HI_AL': ['ND','NE','IA','NV','FL','HI','AL'],
        'addr_state:NM_VA':                 ['NM','VA'],
        'addr_state:OK_TN_MO_LA_MD_NC':     ['OK','TN','MO','LA','MD','NC'],
        'addr_state:UT_KY_AZ_NJ':           ['UT','KY','AZ','NJ'],
        'addr_state:AR_MI_PA_OH_MN':         ['AR','MI','PA','OH','MN'],
        'addr_state:RI_MA_DE_SD_IN':         ['RI','MA','DE','SD','IN'],
        'addr_state:GA_WA_OR':               ['GA','WA','OR'],
        'addr_state:WI_MT':                  ['WI','MT'],
        'addr_state:IL_CT':                  ['IL','CT'],
        'addr_state:KS_SC_CO_VT_AK_MS':     ['KS','SC','CO','VT','AK','MS'],
        'addr_state:WV_NH_WY_DC_ME_ID':     ['WV','NH','WY','DC','ME','ID'],
    }
    for new_col, states in combos.items():
        df[new_col] = sum(df[f'addr_state:{s}'] for s in states)
    return df


def fe_purpose(df):
    _check_cols(df, [f'purpose:{p}' for p in [
        'educational','small_business','wedding','renewable_energy','moving','house',
        'other','medical','vacation','major_purchase','car','home_improvement',
    ]])
    df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum(
        df[f'purpose:{p}'] for p in
        ['educational','small_business','wedding','renewable_energy','moving','house']
    )
    df['purpose:oth__med__vacation'] = sum(
        df[f'purpose:{p}'] for p in ['other','medical','vacation']
    )
    df['purpose:major_purch__car__home_impr'] = sum(
        df[f'purpose:{p}'] for p in ['major_purchase','car','home_improvement']
    )
    return df


def fe_term(df):
    df['term:36'] = np.where(df['term_int'] == 36, 1, 0)
    df['term:60'] = np.where(df['term_int'] == 60, 1, 0)
    return df


def fe_emp_length(df):
    df['emp_length:0']   = np.where(df['emp_length_int'].isin([0]), 1, 0)
    df['emp_length:1']   = np.where(df['emp_length_int'].isin([1]), 1, 0)
    df['emp_length:2-4'] = np.where(df['emp_length_int'].isin(range(2, 5)), 1, 0)
    df['emp_length:5-6'] = np.where(df['emp_length_int'].isin(range(5, 7)), 1, 0)
    df['emp_length:7-9'] = np.where(df['emp_length_int'].isin(range(7, 10)), 1, 0)
    df['emp_length:10']  = np.where(df['emp_length_int'].isin([10]), 1, 0)
    return df


def fe_mths_since_issue_d(df):
    bins = [
        ('mths_since_issue_d:<38',   range(38)),
        ('mths_since_issue_d:38-39', range(38, 40)),
        ('mths_since_issue_d:40-41', range(40, 42)),
        ('mths_since_issue_d:42-48', range(42, 49)),
        ('mths_since_issue_d:49-52', range(49, 53)),
        ('mths_since_issue_d:53-64', range(53, 65)),
        ('mths_since_issue_d:65-84', range(65, 85)),
        ('mths_since_issue_d:>84',   range(85, int(df['mths_since_issue_d'].max()) + 1)),
    ]
    for col, rng in bins:
        df[col] = np.where(df['mths_since_issue_d'].isin(rng), 1, 0)
    return df


def fe_int_rate(df):
    df['int_rate:<9.548']       = np.where(df['int_rate'] <= 9.548, 1, 0)
    df['int_rate:9.548-12.025'] = np.where((df['int_rate'] > 9.548)  & (df['int_rate'] <= 12.025), 1, 0)
    df['int_rate:12.025-15.74'] = np.where((df['int_rate'] > 12.025) & (df['int_rate'] <= 15.74),  1, 0)
    df['int_rate:15.74-20.281'] = np.where((df['int_rate'] > 15.74)  & (df['int_rate'] <= 20.281), 1, 0)
    df['int_rate:>20.281']      = np.where(df['int_rate'] > 20.281, 1, 0)
    return df


def fe_mths_since_earliest_cr_line(df):
    col = 'mths_since_earliest_cr_line'
    df[f'{col}:<140']    = np.where(df[col].isin(range(140)), 1, 0)
    df[f'{col}:141-164'] = np.where(df[col].isin(range(140, 165)), 1, 0)
    df[f'{col}:165-247'] = np.where(df[col].isin(range(165, 248)), 1, 0)
    df[f'{col}:248-270'] = np.where(df[col].isin(range(248, 271)), 1, 0)
    df[f'{col}:271-352'] = np.where(df[col].isin(range(271, 353)), 1, 0)
    df[f'{col}:>352']    = np.where(df[col].isin(range(353, int(df[col].max()) + 1)), 1, 0)
    return df


def fe_delinq_2yrs(df):
    df['delinq_2yrs:0']   = np.where(df['delinq_2yrs'] == 0, 1, 0)
    df['delinq_2yrs:1-3'] = np.where((df['delinq_2yrs'] >= 1) & (df['delinq_2yrs'] <= 3), 1, 0)
    df['delinq_2yrs:>=4'] = np.where(df['delinq_2yrs'] >= 4, 1, 0)
    return df


def fe_inq_last_6mths(df):
    df['inq_last_6mths:0']   = np.where(df['inq_last_6mths'] == 0, 1, 0)
    df['inq_last_6mths:1-2'] = np.where((df['inq_last_6mths'] >= 1) & (df['inq_last_6mths'] <= 2), 1, 0)
    df['inq_last_6mths:3-6'] = np.where((df['inq_last_6mths'] >= 3) & (df['inq_last_6mths'] <= 6), 1, 0)
    df['inq_last_6mths:>6']  = np.where(df['inq_last_6mths'] > 6, 1, 0)
    return df


def fe_open_acc(df):
    df['open_acc:0']     = np.where(df['open_acc'] == 0, 1, 0)
    df['open_acc:1-3']   = np.where((df['open_acc'] >= 1)  & (df['open_acc'] <= 3),  1, 0)
    df['open_acc:4-12']  = np.where((df['open_acc'] >= 4)  & (df['open_acc'] <= 12), 1, 0)
    df['open_acc:13-17'] = np.where((df['open_acc'] >= 13) & (df['open_acc'] <= 17), 1, 0)
    df['open_acc:18-22'] = np.where((df['open_acc'] >= 18) & (df['open_acc'] <= 22), 1, 0)
    df['open_acc:23-25'] = np.where((df['open_acc'] >= 23) & (df['open_acc'] <= 25), 1, 0)
    df['open_acc:26-30'] = np.where((df['open_acc'] >= 26) & (df['open_acc'] <= 30), 1, 0)
    df['open_acc:>=31']  = np.where(df['open_acc'] >= 31, 1, 0)
    return df


def fe_pub_rec(df):
    df['pub_rec:0-2'] = np.where((df['pub_rec'] >= 0) & (df['pub_rec'] <= 2), 1, 0)
    df['pub_rec:3-4'] = np.where((df['pub_rec'] >= 3) & (df['pub_rec'] <= 4), 1, 0)
    df['pub_rec:>=5'] = np.where(df['pub_rec'] >= 5, 1, 0)
    return df


def fe_total_acc(df):
    df['total_acc:<=27']  = np.where(df['total_acc'] <= 27, 1, 0)
    df['total_acc:28-51'] = np.where((df['total_acc'] >= 28) & (df['total_acc'] <= 51), 1, 0)
    df['total_acc:>=52']  = np.where(df['total_acc'] >= 52, 1, 0)
    return df


def fe_acc_now_delinq(df):
    df['acc_now_delinq:0']   = np.where(df['acc_now_delinq'] == 0, 1, 0)
    df['acc_now_delinq:>=1'] = np.where(df['acc_now_delinq'] >= 1, 1, 0)
    return df


def fe_total_rev_hi_lim(df):
    df['total_rev_hi_lim:<=5K']   = np.where(df['total_rev_hi_lim'] <= 5000, 1, 0)
    df['total_rev_hi_lim:5K-10K'] = np.where((df['total_rev_hi_lim'] > 5000)  & (df['total_rev_hi_lim'] <= 10000), 1, 0)
    df['total_rev_hi_lim:10K-20K']= np.where((df['total_rev_hi_lim'] > 10000) & (df['total_rev_hi_lim'] <= 20000), 1, 0)
    df['total_rev_hi_lim:20K-30K']= np.where((df['total_rev_hi_lim'] > 20000) & (df['total_rev_hi_lim'] <= 30000), 1, 0)
    df['total_rev_hi_lim:30K-40K']= np.where((df['total_rev_hi_lim'] > 30000) & (df['total_rev_hi_lim'] <= 40000), 1, 0)
    df['total_rev_hi_lim:40K-55K']= np.where((df['total_rev_hi_lim'] > 40000) & (df['total_rev_hi_lim'] <= 55000), 1, 0)
    df['total_rev_hi_lim:55K-95K']= np.where((df['total_rev_hi_lim'] > 55000) & (df['total_rev_hi_lim'] <= 95000), 1, 0)
    df['total_rev_hi_lim:>95K']   = np.where(df['total_rev_hi_lim'] > 95000, 1, 0)
    return df


def fe_annual_inc(df):
    df['annual_inc:<20K']     = np.where(df['annual_inc'] <= 20000, 1, 0)
    df['annual_inc:20K-30K']  = np.where((df['annual_inc'] > 20000)  & (df['annual_inc'] <= 30000),  1, 0)
    df['annual_inc:30K-40K']  = np.where((df['annual_inc'] > 30000)  & (df['annual_inc'] <= 40000),  1, 0)
    df['annual_inc:40K-50K']  = np.where((df['annual_inc'] > 40000)  & (df['annual_inc'] <= 50000),  1, 0)
    df['annual_inc:50K-60K']  = np.where((df['annual_inc'] > 50000)  & (df['annual_inc'] <= 60000),  1, 0)
    df['annual_inc:60K-70K']  = np.where((df['annual_inc'] > 60000)  & (df['annual_inc'] <= 70000),  1, 0)
    df['annual_inc:70K-80K']  = np.where((df['annual_inc'] > 70000)  & (df['annual_inc'] <= 80000),  1, 0)
    df['annual_inc:80K-90K']  = np.where((df['annual_inc'] > 80000)  & (df['annual_inc'] <= 90000),  1, 0)
    df['annual_inc:90K-100K'] = np.where((df['annual_inc'] > 90000)  & (df['annual_inc'] <= 100000), 1, 0)
    df['annual_inc:100K-120K']= np.where((df['annual_inc'] > 100000) & (df['annual_inc'] <= 120000), 1, 0)
    df['annual_inc:120K-140K']= np.where((df['annual_inc'] > 120000) & (df['annual_inc'] <= 140000), 1, 0)
    df['annual_inc:>140K']    = np.where(df['annual_inc'] > 140000, 1, 0)
    return df


def fe_mths_since_last_delinq(df):
    col = 'mths_since_last_delinq'
    df[f'{col}:Missing'] = np.where(df[col].isnull(), 1, 0)
    df[f'{col}:0-3']     = np.where((df[col] >= 0)  & (df[col] <= 3),  1, 0)
    df[f'{col}:4-30']    = np.where((df[col] >= 4)  & (df[col] <= 30), 1, 0)
    df[f'{col}:31-56']   = np.where((df[col] >= 31) & (df[col] <= 56), 1, 0)
    df[f'{col}:>=57']    = np.where(df[col] >= 57, 1, 0)
    return df


def fe_dti(df):
    df['dti:<=1.4']     = np.where(df['dti'] <= 1.4, 1, 0)
    df['dti:1.4-3.5']   = np.where((df['dti'] > 1.4)  & (df['dti'] <= 3.5),  1, 0)
    df['dti:3.5-7.7']   = np.where((df['dti'] > 3.5)  & (df['dti'] <= 7.7),  1, 0)
    df['dti:7.7-10.5']  = np.where((df['dti'] > 7.7)  & (df['dti'] <= 10.5), 1, 0)
    df['dti:10.5-16.1'] = np.where((df['dti'] > 10.5) & (df['dti'] <= 16.1), 1, 0)
    df['dti:16.1-20.3'] = np.where((df['dti'] > 16.1) & (df['dti'] <= 20.3), 1, 0)
    df['dti:20.3-21.7'] = np.where((df['dti'] > 20.3) & (df['dti'] <= 21.7), 1, 0)
    df['dti:21.7-22.4'] = np.where((df['dti'] > 21.7) & (df['dti'] <= 22.4), 1, 0)
    df['dti:22.4-35']   = np.where((df['dti'] > 22.4) & (df['dti'] <= 35),   1, 0)
    df['dti:>35']       = np.where(df['dti'] > 35, 1, 0)
    return df


def fe_mths_since_last_record(df):
    col = 'mths_since_last_record'
    df[f'{col}:Missing'] = np.where(df[col].isnull(), 1, 0)
    df[f'{col}:0-2']     = np.where((df[col] >= 0)  & (df[col] <= 2),  1, 0)
    df[f'{col}:3-20']    = np.where((df[col] >= 3)  & (df[col] <= 20), 1, 0)
    df[f'{col}:21-31']   = np.where((df[col] >= 21) & (df[col] <= 31), 1, 0)
    df[f'{col}:32-80']   = np.where((df[col] >= 32) & (df[col] <= 80), 1, 0)
    df[f'{col}:81-86']   = np.where((df[col] >= 81) & (df[col] <= 86), 1, 0)
    df[f'{col}:>86']     = np.where(df[col] > 86, 1, 0)
    return df


def apply_feature_engineering(df):
    """Apply all feature steps (bins + combined dummies)."""
    df = fe_home_ownership(df)
    df = fe_addr_state(df)
    df = fe_purpose(df)
    df = fe_term(df)
    df = fe_emp_length(df)
    df = fe_mths_since_issue_d(df)
    df = fe_int_rate(df)
    df = fe_mths_since_earliest_cr_line(df)
    df = fe_delinq_2yrs(df)
    df = fe_inq_last_6mths(df)
    df = fe_open_acc(df)
    df = fe_pub_rec(df)
    df = fe_total_acc(df)
    df = fe_acc_now_delinq(df)
    df = fe_total_rev_hi_lim(df)
    df = fe_annual_inc(df)
    df = fe_mths_since_last_delinq(df)
    df = fe_dti(df)
    df = fe_mths_since_last_record(df)
    return df


def align_columns(df, expected_cols):
    """
    Add zero-filled columns for any column seen during fit but absent now,
    and drop any extra columns not seen during fit.
    """
    df = df.copy()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]


def validate_raw_columns(df):
    """Raise ValueError if required raw columns are missing."""
    missing = [c for c in _required_raw_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")


def load_and_split(csv_path, reference_date='2017-12-01', test_size=0.2, random_state=42):
    """
    Load raw CSV → general preprocessing → dummies → fill NA → good_bad →
    train/test split.

    Returns X_train, X_test, y_train, y_test (raw, not yet WoE-binned).
    """
    raw = pd.read_csv(csv_path)
    df  = general_preprocessing(raw, reference_date)
    df  = make_dummies(df)
    df  = fill_missing(df)
    df  = create_good_bad(df)

    X = df.drop('good_bad', axis=1)
    y = df['good_bad']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_preprocessor(X_train):
    """
    'Fit' the preprocessor: apply feature engineering to the training set
    and capture the resulting column list.

    Returns
    -------
    expected_cols : list of column names
    """
    validate_raw_columns(X_train)
    dummy = apply_feature_engineering(X_train.copy())
    return dummy.columns.tolist()


def transform(X, expected_cols):
    """
    Apply feature engineering and align columns to the training schema.

    Parameters
    ----------
    X             : raw (but split) input DataFrame
    expected_cols : list returned by fit_preprocessor

    Returns the processed DataFrame.
    """
    validate_raw_columns(X)
    out = apply_feature_engineering(X.copy())
    return align_columns(out, expected_cols)


def fit_transform(X_train, y_train=None):
    """
    Shortcut: fit + transform on the same data.

    Returns (X_processed, expected_cols).
    """
    expected_cols = fit_preprocessor(X_train)
    return transform(X_train, expected_cols), expected_cols



if __name__ == '__main__':
    os.makedirs('out/prepr', exist_ok=True)

    # Fit & transform training data
    print("Processing: in/loan_data.csv")
    X_train, X_test, y_train, y_test = load_and_split('in/loan_data.csv')

    X_train_proc, expected_cols = fit_transform(X_train)
    X_test_proc = transform(X_test, expected_cols)

    print(f"  Train shape: {X_train_proc.shape}")
    print(f"  Test  shape: {X_test_proc.shape}")

    X_train_proc.to_csv('out/prepr/loan_data_inputs_train.csv')
    y_train.to_csv('out/prepr/loan_data_targets_train.csv')
    X_test_proc.to_csv('out/prepr/loan_data_inputs_test.csv')
    y_test.to_csv('out/prepr/loan_data_targets_test.csv')
    pd.Series(expected_cols, name='column').to_csv('out/prepr/expected_cols.csv', index=False)
    print("  Saved train/test CSVs -> out/prepr/")

    # Transform new data (for monitoring) 
    if os.path.exists('in/loan_data_new.csv'):
        print("\nProcessing new data: in/loan_data_new.csv")
        raw_new = pd.read_csv('in/loan_data_new.csv')
        df_new  = general_preprocessing(raw_new, reference_date='2018-12-01')
        df_new  = make_dummies(df_new)
        df_new  = fill_missing(df_new)
        df_new  = create_good_bad(df_new)

        X_new = df_new.drop('good_bad', axis=1)
        y_new = df_new['good_bad']
        X_new = apply_feature_engineering(X_new)
        X_new = align_columns(X_new, expected_cols)

        X_new.to_csv('out/prepr/loan_data_inputs_new.csv')
        y_new.to_csv('out/prepr/loan_data_targets_new.csv')
        print(f"  New data shape: {X_new.shape}")
        print("  Saved new data CSVs -> out/prepr/")
    else:
        print("\nNo new data found at in/loan_data_new.csv — skipping.")

    print("\nDone.")
