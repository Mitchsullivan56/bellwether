# ---------------------------------------------------------------
# train_model.py  —  FF Projections, walk-forward model vs baselines
# USAGE: python3 train_model.py
# Loads data/model_table.parquet (built by build_dataset.py) and
# trains a gradient-boosted model with walk-forward CV: for each
# test season, train only on seasons strictly before it, so the
# model never sees the future. Compares MAE against the
# naive_last/avg3 baselines on the same eval population.
# ---------------------------------------------------------------

from ff_model import load_eval_table, walk_forward_predictions, TARGET

POSITIONS = ["QB", "RB", "WR", "TE"]

df = load_eval_table()
result, test_seasons = walk_forward_predictions(df)

result["err_naive"] = (result[TARGET] - result["pts_last1"]).abs()
result["err_avg3"] = (result[TARGET] - result["pts_avg3"]).abs()
result["err_model"] = (result[TARGET] - result["pred"]).abs()

print(f"\nWalk-forward backtest: each season in {test_seasons[0]}-{test_seasons[-1]} "
      f"is predicted using only prior seasons as training data")
print("MODEL MAE vs BASELINES  (lower = better)")
print("-" * 58)
print(f"  {'Position':<10}{'naive_last':>12}{'avg3':>10}{'model':>10}{'n':>10}")
print("-" * 58)
for pos in POSITIONS:
    sub = result[result["position"] == pos]
    if sub.empty:
        continue
    print(f"  {pos:<10}{sub['err_naive'].mean():>12.2f}"
          f"{sub['err_avg3'].mean():>10.2f}{sub['err_model'].mean():>10.2f}"
          f"{len(sub):>10}")
print("-" * 58)
print(f"  {'ALL':<10}{result['err_naive'].mean():>12.2f}"
      f"{result['err_avg3'].mean():>10.2f}{result['err_model'].mean():>10.2f}"
      f"{len(result):>10}")
print("-" * 58)

print("\nBy test season (ALL positions pooled):")
print("-" * 58)
print(f"  {'Season':<10}{'naive_last':>12}{'avg3':>10}{'model':>10}{'n':>10}")
print("-" * 58)
for s in test_seasons:
    sub = result[result["season"] == s]
    print(f"  {s:<10}{sub['err_naive'].mean():>12.2f}"
          f"{sub['err_avg3'].mean():>10.2f}{sub['err_model'].mean():>10.2f}"
          f"{len(sub):>10}")
print("-" * 58)
