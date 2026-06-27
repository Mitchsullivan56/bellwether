# ---------------------------------------------------------------
# ff_cache.py  —  shared compute core + content-hash data version.
# Streamlit-free so warm_cache.py (run in CI) and app.py both use it.
# The version is a hash of the parquet CONTENTS (not mtimes), so a
# cache pre-built in the GitHub Action stays valid after git checkout
# on Streamlit Cloud — where the filesystem is read-only and the app
# can't rebuild the cache itself.
# ---------------------------------------------------------------

import hashlib
import pathlib

import pandas as pd

from ff_model import load_eval_table, walk_forward_predictions
from ff_eval import mae_table, mae_by_season, load_fp_rankings, rank_corr

DATA = pathlib.Path(__file__).resolve().parent / "data"
CACHE = DATA / "compute_cache.joblib"
_INPUTS = ("model_table.parquet", "fp_rankings.parquet", "feature_importance.parquet")


def data_version():
    """SHA-256 of the input parquets' contents — stable across git checkout."""
    h = hashlib.sha256()
    for f in _INPUTS:
        p = DATA / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def compute_core():
    """The expensive walk-forward backtest + derived tables (no Streamlit)."""
    df = load_eval_table()
    result, test_seasons = walk_forward_predictions(df)
    result = result[result["season"].isin(test_seasons)]
    fp = load_fp_rankings()
    summary, weekly, merged = rank_corr(result, fp)
    imp_path = DATA / "feature_importance.parquet"
    importance = pd.read_parquet(imp_path) if imp_path.exists() else None
    return {
        "result": result,
        "test_seasons": test_seasons,
        "mae": mae_table(result),
        "mae_season": mae_by_season(result, test_seasons),
        "rank_summary": summary,
        "merged": merged,
        "importance": importance,
    }
