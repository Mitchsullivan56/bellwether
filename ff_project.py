# ---------------------------------------------------------------
# ff_project.py  —  forward projections for setting a lineup.
# Trains the model on completed games and projects fantasy points
# for a week's players, ranked. Two modes:
#   project_live()        -> the upcoming (unplayed) week, from
#                            projection_frame.parquet (in-season only)
#   project_week(s, w)    -> any completed week, training only on
#                            games before it (demo / accuracy view)
# ---------------------------------------------------------------

import pathlib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ff_model import load_eval_table, TARGET, CATEGORICAL_COLS

PROJ_PATH = pathlib.Path(__file__).resolve().parent / "data" / "projection_frame.parquet"

# A single solid config (no per-fold tuning) keeps projections fast/responsive;
# the tuning gain over this is marginal and doesn't change rank order.
PROD_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=None,
                   l2_regularization=1.0, random_state=0)


def _feature_cols(df):
    drop = ["player_id", "player_display_name", "season", "week", TARGET]
    return [c for c in df.columns if c not in drop]


def _board(score, train, feats, with_actual):
    score = score.copy()
    # align categorical levels to the training data so unseen values -> NaN
    for c in CATEGORICAL_COLS:
        if c in score.columns:
            score[c] = pd.Categorical(score[c], categories=train[c].cat.categories)
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", **PROD_PARAMS)
    model.fit(train[feats], train[TARGET])
    score["Projected"] = model.predict(score[feats])

    rename = {"player_display_name": "Player", "position": "Pos",
              "opponent_team": "Opp"}
    out = score.rename(columns=rename)
    cols = ["Player", "Pos", "Opp", "Projected"]
    if with_actual:
        out = out.rename(columns={TARGET: "Actual"})
        cols.append("Actual")
    return (out[cols].sort_values("Projected", ascending=False)
            .reset_index(drop=True))


def project_week(season, week):
    """Train on every completed game before (season, week); project that week.
    Includes the actual result, so it doubles as an accuracy view."""
    eval_df = load_eval_table()
    feats = _feature_cols(eval_df)
    score = eval_df[(eval_df["season"] == season) & (eval_df["week"] == week)]
    train = eval_df[(eval_df["season"] < season) |
                    ((eval_df["season"] == season) & (eval_df["week"] < week))]
    if score.empty or train.empty:
        return None
    return _board(score, train, feats, with_actual=True)


def project_live():
    """Project the upcoming unplayed week from projection_frame.parquet, trained
    on all completed games. Returns (board, season, week) or None in the offseason."""
    if not PROJ_PATH.exists():
        return None
    score = pd.read_parquet(PROJ_PATH)
    score = score[score["games_so_far"] >= 3]
    if score.empty:
        return None
    eval_df = load_eval_table()
    feats = _feature_cols(eval_df)
    season, week = int(score["season"].iloc[0]), int(score["week"].iloc[0])
    return _board(score, eval_df, feats, with_actual=False), season, week


def completed_weeks():
    """{season: [weeks]} of completed weeks available for the demo view."""
    eval_df = load_eval_table()
    return {int(s): sorted(int(w) for w in g["week"].unique())
            for s, g in eval_df.groupby("season")}
