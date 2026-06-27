# ---------------------------------------------------------------
# ff_eval.py  —  shared metric computation used by the CLI scripts
# and the Streamlit app, so every surface reports identical numbers.
# Pure functions: take a predictions frame, return tidy DataFrames.
# ---------------------------------------------------------------

import pathlib
import pandas as pd
from scipy.stats import spearmanr
from ff_model import TARGET

FP_PATH = pathlib.Path(__file__).resolve().parent / "data" / "fp_rankings.parquet"
POSITIONS = ["QB", "RB", "WR", "TE"]


def add_errors(result):
    result = result.copy()
    result["err_naive"] = (result[TARGET] - result["pts_last1"]).abs()
    result["err_avg3"] = (result[TARGET] - result["pts_avg3"]).abs()
    result["err_model"] = (result[TARGET] - result["pred"]).abs()
    return result


def mae_table(result):
    """Mean absolute error by position (plus an ALL row)."""
    result = add_errors(result)
    rows = []
    for pos in POSITIONS + ["ALL"]:
        sub = result if pos == "ALL" else result[result["position"] == pos]
        if sub.empty:
            continue
        rows.append({
            "Position": pos,
            "naive_last": sub["err_naive"].mean(),
            "avg3": sub["err_avg3"].mean(),
            "model": sub["err_model"].mean(),
            "n": len(sub),
        })
    return pd.DataFrame(rows)


def mae_by_season(result, test_seasons):
    """MAE pooled across positions, one row per test season."""
    result = add_errors(result)
    rows = []
    for s in test_seasons:
        sub = result[result["season"] == s]
        rows.append({
            "season": int(s),
            "naive_last": sub["err_naive"].mean(),
            "avg3": sub["err_avg3"].mean(),
            "model": sub["err_model"].mean(),
            "n": len(sub),
        })
    return pd.DataFrame(rows)


def load_fp_rankings():
    return pd.read_parquet(FP_PATH)


def rank_corr(result, fp):
    """Join model/baseline predictions to FantasyPros ECR and score each
    by weekly Spearman rank correlation vs the actual finish order.
    Returns (summary_by_position, weekly_detail, merged_player_weeks)."""
    merged = result.merge(fp, on=["player_id", "season", "week", "position"], how="inner")
    merged["position"] = merged["position"].astype(str)  # avoid empty categorical groups

    group = merged.groupby(["season", "week", "position"], observed=True)
    merged["rank_actual"] = group[TARGET].rank(ascending=False)
    merged["rank_model"] = group["pred"].rank(ascending=False)
    merged["rank_avg3"] = group["pts_avg3"].rank(ascending=False)
    merged["rank_ecr"] = group["ecr"].rank(ascending=True)

    def weekly_spearman(g, col):
        if g[col].nunique() < 2 or len(g) < 3:
            return None
        return spearmanr(g[col], g["rank_actual"]).statistic

    rows = []
    for (season, week, position), g in merged.groupby(["season", "week", "position"], observed=True):
        rows.append({
            "season": int(season), "week": int(week), "position": position, "n": len(g),
            "rho_avg3": weekly_spearman(g, "rank_avg3"),
            "rho_model": weekly_spearman(g, "rank_model"),
            "rho_ecr": weekly_spearman(g, "rank_ecr"),
        })
    weekly = pd.DataFrame(rows).dropna()

    summary = []
    for pos in POSITIONS + ["ALL"]:
        sub = weekly if pos == "ALL" else weekly[weekly["position"] == pos]
        if sub.empty:
            continue
        summary.append({
            "Position": pos,
            "avg3": sub["rho_avg3"].mean(),
            "model": sub["rho_model"].mean(),
            "FP_ecr": sub["rho_ecr"].mean(),
            "weeks": len(sub),
        })
    return pd.DataFrame(summary), weekly, merged
