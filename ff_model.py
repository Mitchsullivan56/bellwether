# ---------------------------------------------------------------
# ff_model.py  —  shared walk-forward training used by train_model.py
# and backtest_vs_experts.py, so both backtests train identically.
#
# Hyperparameters are tuned per outer fold using a NESTED split: the
# test season itself is never used to pick hyperparameters. Inside each
# fold's training data, the most recent season is held out as an inner
# validation set, a small grid is scored on it, and the winning params
# are then refit on the full training data to predict the real test
# season. The one exception is the very first fold, which has only one
# training season and therefore can't be split further — it falls back
# to the grid's first (middle-of-the-road) setting.
# ---------------------------------------------------------------

import pathlib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

IN_PATH = pathlib.Path(__file__).resolve().parent / "data" / "model_table.parquet"
TARGET = "fantasy_points_ppr"
CATEGORICAL_COLS = ["position", "opponent_team", "injury_status", "roof"]

PARAM_GRID = [
    dict(max_iter=100, max_depth=None, learning_rate=0.10, l2_regularization=0.0),
    dict(max_iter=200, max_depth=None, learning_rate=0.05, l2_regularization=0.0),
    dict(max_iter=100, max_depth=4,    learning_rate=0.10, l2_regularization=0.0),
    dict(max_iter=200, max_depth=4,    learning_rate=0.05, l2_regularization=1.0),
    dict(max_iter=150, max_depth=6,    learning_rate=0.05, l2_regularization=1.0),
    dict(max_iter=300, max_depth=None, learning_rate=0.03, l2_regularization=1.0),
]


def load_eval_table():
    df = pd.read_parquet(IN_PATH)
    df = df[df["games_so_far"] >= 3].copy()  # same fair-bar filter as the baselines
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def _fit(train, feature_cols, params):
    model = HistGradientBoostingRegressor(
        categorical_features="from_dtype", random_state=0, **params)
    model.fit(train[feature_cols], train[TARGET])
    return model


def _best_params(train, feature_cols):
    train_seasons = sorted(train["season"].unique())
    if len(train_seasons) < 2:
        return PARAM_GRID[0]
    inner_valid_season = train_seasons[-1]
    inner_train = train[train["season"] < inner_valid_season]
    inner_valid = train[train["season"] == inner_valid_season]

    best_params, best_mae = PARAM_GRID[0], float("inf")
    for params in PARAM_GRID:
        model = _fit(inner_train, feature_cols, params)
        mae = (inner_valid[TARGET] - model.predict(inner_valid[feature_cols])).abs().mean()
        if mae < best_mae:
            best_mae, best_params = mae, params
    return best_params


def walk_forward_predictions(df, per_position=False):
    """Train on seasons strictly before each test season, predict on it.
    Returns a frame with player_id/season/week/position/actual/baselines/pred."""
    drop_cols = ["player_id", "player_display_name", "season", "week", TARGET]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    seasons = sorted(df["season"].unique())
    test_seasons = seasons[1:]  # first season has no prior data to train on
    result_cols = ["player_id", "player_display_name", "season", "week", "position",
                   TARGET, "pts_last1", "pts_avg3"]

    folds = []
    for test_season in test_seasons:
        train_all = df[df["season"] < test_season]
        test_all = df[df["season"] == test_season]

        groups = [None] if not per_position else sorted(test_all["position"].unique())
        for pos in groups:
            train = train_all if pos is None else train_all[train_all["position"] == pos]
            test = test_all if pos is None else test_all[test_all["position"] == pos]
            if train.empty or test.empty:
                continue
            params = _best_params(train, feature_cols)
            model = _fit(train, feature_cols, params)
            fold = test[result_cols].copy()
            fold["pred"] = model.predict(test[feature_cols])
            folds.append(fold)

    return pd.concat(folds, ignore_index=True), test_seasons
