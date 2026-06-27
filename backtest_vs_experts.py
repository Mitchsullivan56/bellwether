# ---------------------------------------------------------------
# backtest_vs_experts.py  —  FF Projections, model vs FantasyPros ECR
# USAGE: python3 backtest_vs_experts.py
# FantasyPros only archives expert-consensus RANKINGS historically,
# not point projections, so the head-to-head metric is rank-order
# accuracy: per (season, week, position), rank players by the
# model's predicted points, by the avg3 baseline, and by FantasyPros'
# ECR, then correlate each against the actual finish order using
# Spearman's rho. Averaged across weeks, higher rho = better at
# sorting players correctly.
# ---------------------------------------------------------------

from ff_model import load_eval_table, walk_forward_predictions
from ff_eval import load_fp_rankings, rank_corr, POSITIONS

df = load_eval_table()
result, test_seasons = walk_forward_predictions(df)
result = result[result["season"].isin(test_seasons)]

fp = load_fp_rankings()
summary, weekly, merged = rank_corr(result, fp)

print(f"\nHead-to-head vs FantasyPros expert consensus rankings "
      f"({merged['season'].min()}-{merged['season'].max()}, {len(merged):,} matched player-weeks)")
print("Mean weekly Spearman rank correlation vs actual finish (higher = better)")
print("-" * 58)
print(f"  {'Position':<10}{'avg3':>10}{'model':>10}{'FP ecr':>10}{'weeks':>10}")
print("-" * 58)
for _, r in summary.iterrows():
    if r["Position"] == "ALL":
        print("-" * 58)
    print(f"  {r['Position']:<10}{r['avg3']:>10.3f}"
          f"{r['model']:>10.3f}{r['FP_ecr']:>10.3f}{int(r['weeks']):>10}")
print("-" * 58)
