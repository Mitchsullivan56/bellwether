# ---------------------------------------------------------------
# analyze_importance.py  —  what the model actually relies on.
# USAGE: python3 analyze_importance.py
# Trains the tuned model on all-but-last season, then measures
# PERMUTATION importance on the held-out last season: how much MAE
# gets worse (in fantasy points) when each feature is shuffled.
# This is model-agnostic and honest — it reflects what the model
# leans on to be accurate, not just split counts. Saved for the app.
# ---------------------------------------------------------------

import pathlib
import pandas as pd
from sklearn.inspection import permutation_importance
from ff_model import load_eval_table, _fit, _best_params, TARGET

OUT = pathlib.Path(__file__).resolve().parent / "data" / "feature_importance.parquet"

df = load_eval_table()
drop_cols = ["player_id", "player_display_name", "season", "week", TARGET]
feature_cols = [c for c in df.columns if c not in drop_cols]

seasons = sorted(df["season"].unique())
test_season = seasons[-1]
train = df[df["season"] < test_season]
test = df[df["season"] == test_season]

params = _best_params(train, feature_cols)
model = _fit(train, feature_cols, params)

r = permutation_importance(
    model, test[feature_cols], test[TARGET],
    scoring="neg_mean_absolute_error", n_repeats=8, random_state=0, n_jobs=-1)

imp = (pd.DataFrame({"feature": feature_cols,
                     "mae_cost": r.importances_mean,   # MAE pts lost when shuffled
                     "std": r.importances_std})
       .sort_values("mae_cost", ascending=False)
       .reset_index(drop=True))
imp.to_parquet(OUT, index=False)

print(f"Permutation importance on held-out {test_season} "
      f"(MAE pts worse when the feature is shuffled out):")
print("-" * 52)
for _, row in imp.head(20).iterrows():
    print(f"  {row['feature']:<28}{row['mae_cost']:>8.3f}")
print(f"\nSaved: {OUT}")
