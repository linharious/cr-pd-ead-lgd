"""
Credit Risk Modeling - Expected Loss (Pandas Approach)

    EL = PD × LGD × EAD
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def plot_el_distributions(df):
    sns.set()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Expected Loss Component Distributions', fontsize=14)

    for ax, col, color in zip(
        axes.flatten(),
        ['PD', 'LGD', 'CCF', 'EL'],
        ['steelblue', 'darkorange', 'seagreen', 'firebrick'],
    ):
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=50, color=color, alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()



def compute_expected_loss(
    preprocessed_csv,
    pd_inputs_train,
    pd_inputs_test,
    reg,
    pd_model_features,
    lgd_st1,
    lgd_st2,
    ead_model,
    lgd_model_features,
    feature_medians=None,
    out_dir='out/el',
    plot=True,
):
    """
    Compute PD, LGD, EAD and EL for every loan in the portfolio.

    Parameters
    ----------
    preprocessed_csv   : path to loan_data.csv (raw file with dummy variables
                         and continuous features for LGD/EAD)
    pd_inputs_train    : path to loan_data_inputs_train.csv (WoE-binned, for PD)
    pd_inputs_test     : path to loan_data_inputs_test.csv  (WoE-binned, for PD)
    reg                : fitted LogisticRegressionWithPValues (from cr_pd_model_pipeline)
    pd_model_features  : list of feature columns for the PD model
    lgd_st1            : fitted LogisticRegressionWithPValues (LGD Stage 1)
    lgd_st2            : fitted LinearRegressionWithPValues   (LGD Stage 2)
    ead_model          : fitted LinearRegressionWithPValues   (EAD)
    lgd_model_features : list of feature columns for LGD/EAD models
    feature_medians    : Series of train-set medians for LGD/EAD imputation
    out_dir            : directory to write output CSVs
    plot               : whether to show distribution charts

    Returns
    -------
    funded_amnt, PD, LGD, EAD, CCF, EL
    """
    from lgd_ead import select_lgd_ead_features
    from pd_model import select_features

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  EXPECTED LOSS REPORT")
    print("=" * 60)

    print("\n[1/4] Loading full loan data …")
    df = pd.read_csv(preprocessed_csv)
    df['mths_since_last_delinq'].fillna(0, inplace=True)
    df['mths_since_last_record'].fillna(0, inplace=True)
    print(f"  Total loans: {len(df):,}")

    print("[2/4] Computing LGD …")
    X_lgd = select_lgd_ead_features(df, lgd_model_features, feature_medians)

    st1_binary    = lgd_st1.predict(X_lgd)
    st2_pred      = lgd_st2.predict(X_lgd)
    recovery_rate = np.clip(st1_binary * st2_pred, 0, 1)

    df['recovery_rate_st_1'] = st1_binary
    df['recovery_rate_st_2'] = st2_pred
    df['recovery_rate']      = recovery_rate
    df['LGD']                = 1 - recovery_rate

    print(f"  LGD  mean={df['LGD'].mean():.4f}  std={df['LGD'].std():.4f}  "
          f"min={df['LGD'].min():.4f}  max={df['LGD'].max():.4f}")

    print("[3/4] Computing EAD …")
    ccf = np.clip(ead_model.predict(X_lgd), 0, 1)
    df['CCF'] = ccf
    df['EAD'] = ccf * df['funded_amnt']

    print(f"  CCF  mean={df['CCF'].mean():.4f}  std={df['CCF'].std():.4f}  "
          f"min={df['CCF'].min():.4f}  max={df['CCF'].max():.4f}")
    print(f"  EAD  mean={df['EAD'].mean():,.2f}  total={df['EAD'].sum():,.2f}")

    print("[4/4] Computing PD …")
    train_pd = pd.read_csv(pd_inputs_train, index_col=0)
    test_pd  = pd.read_csv(pd_inputs_test,  index_col=0)
    all_pd   = pd.concat([train_pd, test_pd], axis=0)

    pd_inputs = select_features(all_pd, pd_model_features)
    all_pd['PD'] = 1 - reg.predict_proba(pd_inputs)[:, 1]

    print(f"  PD   mean={all_pd['PD'].mean():.4f}  std={all_pd['PD'].std():.4f}  "
          f"min={all_pd['PD'].min():.4f}  max={all_pd['PD'].max():.4f}")

    df_combined = pd.concat([df, all_pd[['PD']]], axis=1)
    df_combined['EL'] = (
        df_combined['PD'] * df_combined['LGD'] * df_combined['EAD']
    )

    total_el     = df_combined['EL'].sum()
    total_funded = df_combined['funded_amnt'].sum()
    el_ratio     = total_el / total_funded

    print(f"\n{'='*60}")
    print(f"  Total Expected Loss       : ${total_el:>15,.2f}")
    print(f"  Total Funded Amount       : ${total_funded:>15,.2f}")
    print(f"  EL as % of Funded Amount  : {el_ratio:>15.4%}")
    print(f"{'='*60}")

    output_cols = ['funded_amnt', 'PD', 'LGD', 'EAD', 'CCF', 'EL']
    out_df = df_combined[[c for c in output_cols if c in df_combined.columns]]
    out_df.to_csv(os.path.join(out_dir, 'expected_loss.csv'), index=False)

    pd.DataFrame({
        'Metric': [
            'Total Expected Loss', 'Total Funded Amount', 'EL / Funded Amount',
            'Mean PD', 'Mean LGD', 'Mean CCF', 'Mean EAD', 'Mean EL',
        ],
        'Value': [
            total_el, total_funded, el_ratio,
            df_combined['PD'].mean(), df_combined['LGD'].mean(),
            df_combined['CCF'].mean(), df_combined['EAD'].mean(),
            df_combined['EL'].mean(),
        ],
    }).to_csv(os.path.join(out_dir, 'el_summary.csv'), index=False)

    print(f"\nOutputs saved to: {out_dir}/")

    if plot:
        plot_el_distributions(df_combined)

    return df_combined


if __name__ == '__main__':
    from preprocessing import load_and_split, fit_transform, transform
    from pd_model import fit_pd_model
    from lgd_ead import load_and_prepare, fit_lgd_ead

    os.makedirs('out/el', exist_ok=True)

    print("Step 1: Preprocessing …")
    X_train, X_test, y_train, y_test = load_and_split('in/loan_data.csv')
    X_train_proc, expected_cols = fit_transform(X_train)
    X_test_proc = transform(X_test, expected_cols)

    print("\nStep 2: Fitting PD model …")
    (reg, summary_table, scorecard,
     min_sum_coef, max_sum_coef,
     pd_model_features) = fit_pd_model(X_train_proc, y_train)

    print("\nStep 3: Fitting LGD/EAD models …")
    (X_lgd_train, X_lgd_test,
     y_lgd1_train, y_lgd1_test,
     y_lgd2_train, y_lgd2_test,
     y_ead_train,  y_ead_test,
     feature_medians) = load_and_prepare('in/loan_data.csv')

    (lgd_st1, lgd_st2, ead_model,
     summary_lgd_st1, summary_lgd_st2, summary_ead,
     lgd_model_features, feature_medians) = fit_lgd_ead(
        X_lgd_train, y_lgd1_train, y_lgd2_train, y_ead_train
    )

    os.makedirs('out/prepr', exist_ok=True)
    X_train_proc.to_csv('out/prepr/loan_data_inputs_train.csv')
    X_test_proc.to_csv('out/prepr/loan_data_inputs_test.csv')

    print("\nStep 4: Computing Expected Loss …")
    results = compute_expected_loss(
        preprocessed_csv   = 'in/loan_data.csv',
        pd_inputs_train    = 'out/prepr/loan_data_inputs_train.csv',
        pd_inputs_test     = 'out/prepr/loan_data_inputs_test.csv',
        reg                = reg,
        pd_model_features  = pd_model_features,
        lgd_st1            = lgd_st1,
        lgd_st2            = lgd_st2,
        ead_model          = ead_model,
        lgd_model_features = lgd_model_features,
        feature_medians    = feature_medians,
        out_dir            = 'out/el',
        plot               = True,
    )

    print("\nSample output (first 5 rows):")
    print(results[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head().to_string())
