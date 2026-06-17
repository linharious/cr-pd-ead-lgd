"""creditrisk — credit risk modelling package.

Models Probability of Default (PD) with a 300-850 scorecard, Loss Given Default
(LGD), Exposure at Default (EAD), Expected Loss (EL = PD x LGD x EAD), and
PSI-based model monitoring.

The submodules keep their original behaviour; this file just exposes the
commonly used entry points so the web app can do `from creditrisk import ...`.
"""
from . import preprocessing, pd_model, lgd_ead, expected_loss, monitoring

from .preprocessing import (
    load_and_split,
    fit_preprocessor,
    fit_transform,
    transform,
)
from .pd_model import (
    fit_pd_model,
    score_applicants,
    predict_proba_pd,
    evaluate_pd_model,
    build_scorecard,
    all_features,
)
from .lgd_ead import (
    load_and_prepare,
    fit_lgd_ead,
    predict_lgd,
    predict_ead,
    evaluate_lgd_ead,
    features_all,
)
from .expected_loss import compute_expected_loss
from .monitoring import run_monitoring, psi_from_csvs, compute_psi

__version__ = "0.1.0"

__all__ = [
    "preprocessing", "pd_model", "lgd_ead", "expected_loss", "monitoring",
    "load_and_split", "fit_preprocessor", "fit_transform", "transform",
    "fit_pd_model", "score_applicants", "predict_proba_pd", "evaluate_pd_model",
    "build_scorecard", "all_features",
    "load_and_prepare", "fit_lgd_ead", "predict_lgd", "predict_ead",
    "evaluate_lgd_ead", "features_all",
    "compute_expected_loss", "run_monitoring", "psi_from_csvs", "compute_psi",
    "__version__",
]
