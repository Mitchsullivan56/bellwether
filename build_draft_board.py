# ---------------------------------------------------------------
# build_draft_board.py  —  Bellwether pre-draft PPR rankings.
# USAGE: python3 build_draft_board.py
# Pulls the current PPR expert consensus (FantasyPros, refreshed
# weekly via nflverse) and enriches every player with Bellwether's
# 2-year regression-adjusted expected-production score -> per-position
# tiers and a "Value vs ADP" lens. Re-run (via Refresh) to update as
# days pass. Saved to draft_board.parquet.
# ---------------------------------------------------------------

import pathlib
import pandas as pd
import nflreadpy as nfl
from sklearn.cluster import KMeans
import ff_draft

OUT_PATH = pathlib.Path(__file__).resolve().parent / "data" / "draft_board.parquet"
POSITIONS = ["QB", "RB", "WR", "TE"]
DRAFT_SEASON = nfl.get_current_season() + 1     # drafting for next season
PRIOR = DRAFT_SEASON - 1                         # most recent completed season

# --- Current PPR draft consensus (redraft-overall == ppr-cheatsheets) ---
print(f"Loading FantasyPros PPR redraft consensus for {DRAFT_SEASON}...")
r = nfl.load_ff_rankings(type="draft").to_pandas()
r = r[r["page_type"] == "redraft-overall"].copy()
latest = r["scrape_date"].max()
r = r[(r["scrape_date"] == latest) & (r["pos"].isin(POSITIONS))].copy()
r["fantasypros_id"] = r["id"].astype(str)

ids = nfl.load_ff_playerids().to_pandas()[["fantasypros_id", "gsis_id"]].dropna()
ids["fantasypros_id"] = ids["fantasypros_id"].astype(str)
r = r.merge(ids, on="fantasypros_id", how="left")

# --- Bellwether expected-production score (2yr, regression-adjusted) ---
print(f"Scoring expected production from {DRAFT_SEASON - 2}-{PRIOR}...")
scores = ff_draft.value_scores(DRAFT_SEASON)
r = r.merge(scores.drop(columns=["pos", "merge_name"]), left_on="gsis_id",
            right_on="player_id", how="left")
# name fallback: link brand-new rookies the FantasyPros->gsis crosswalk hasn't caught yet
rk = (scores[scores["merge_name"].notna()][["merge_name", "score"]]
      .drop_duplicates("merge_name").rename(columns={"score": "_rk_score"}))
r["merge_name"] = r["player"].map(ff_draft.norm_name)
r = r.merge(rk, on="merge_name", how="left")
r["score"] = r["score"].fillna(r["_rk_score"])

r = r.sort_values("ecr").reset_index(drop=True)
r["pos_rank"] = r.groupby("pos").cumcount() + 1
r["PosRank"] = r["pos"] + r["pos_rank"].astype(str)        # WR1, RB2, ...

board = r[["player", "pos", "PosRank", "team", "bye", "ecr", "sd",
           "g_prior", "ppg_prior", "xppg_prior", "score"]].rename(columns={
    "player": "Player", "pos": "Pos", "team": "Team", "bye": "Bye", "ecr": "ECR",
    "sd": "ECR_sd", "g_prior": f"G_{PRIOR}", "ppg_prior": f"PPG_{PRIOR}",
    "xppg_prior": f"xPPG_{PRIOR}"})
board["as_of"] = pd.to_datetime(latest)
board["prior_season"] = PRIOR


# --- Tiers (per-position k-means on consensus rank) + value vs ADP ---
def with_tiers_and_value(g):
    g = g.copy()
    n = len(g)
    if n >= 3:                                    # cluster ECR into ~6-player tiers
        k = min(8, max(2, n // 6))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(g[["ECR"]])
        order = sorted(range(k), key=lambda j: km.cluster_centers_[j, 0])
        tier_of = {cl: i + 1 for i, cl in enumerate(order)}   # 1 = best tier
        g["Tier"] = [tier_of[l] for l in km.labels_]
    else:
        g["Tier"] = 1
    # value vs ADP: consensus rank minus expected-production rank (scored players).
    # positive => Bellwether's expected-production lens ranks them above consensus.
    has = g["score"].notna()
    cons = g.loc[has, "ECR"].rank(method="first")
    prod = g.loc[has, "score"].rank(ascending=False, method="first")
    g.loc[has, "Value"] = (cons - prod).round()
    return g


board = pd.concat([with_tiers_and_value(g) for _, g in board.groupby("Pos")])
board = board.sort_values("ECR").reset_index(drop=True).drop(columns=["score"])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
board.to_parquet(OUT_PATH, index=False)
hit = board[f"PPG_{PRIOR}"].notna().mean()
print(f"\nDraft board saved: {OUT_PATH}  ({len(board)} players, consensus as of {latest})")
print(f"{PRIOR} production matched for {hit:.0%} of ranked players (rest are rookies / no data).")
print(board.drop(columns=["as_of", "ECR_sd", f"G_{PRIOR}", "prior_season"]).head(12).to_string(index=False))
