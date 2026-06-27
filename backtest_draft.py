# ---------------------------------------------------------------
# backtest_draft.py  —  does Bellwether's draft lean beat ADP?
# USAGE: python3 backtest_draft.py
# A true multi-season holdout (2022-2025): for each season, rebuild
# the board as it would have looked BEFORE that season (its preseason
# PPR consensus + the same 2-year regression-adjusted expected-
# production score, from data available pre-draft), then score it
# against what actually happened. The honest finding: the consensus
# ranking is strong, but the value lean is only a weak tilt — you
# can't beat consensus on the residual from box-score data.
# Saves player-level results (all seasons) to draft_backtest.parquet.
# ---------------------------------------------------------------

import pathlib
import pandas as pd
import nflreadpy as nfl
import ff_draft

OUT = pathlib.Path(__file__).resolve().parent / "data" / "draft_backtest.parquet"
SEASONS = [2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]

ids = nfl.load_ff_playerids().to_pandas()[["fantasypros_id", "gsis_id"]].dropna()
ids["fantasypros_id"] = ids["fantasypros_id"].astype(str)

print("Loading historical preseason consensus (ADP)...")
rk = nfl.load_ff_rankings(type="all").to_pandas()
rk = rk[(rk["page_type"] == "redraft-overall") &
        (rk["fp_page"] == "/nfl/rankings/ppr-cheatsheets.php")].copy()
rk["scrape_date"] = pd.to_datetime(rk["scrape_date"])


def get_adp(year):
    pre = rk[(rk["scrape_date"] >= f"{year}-07-15") & (rk["scrape_date"] < f"{year}-09-05")]
    a = pre[(pre["scrape_date"] == pre["scrape_date"].max()) & (pre["pos"].isin(POSITIONS))].copy()
    a["fantasypros_id"] = a["id"].astype(str)
    a = a.merge(ids, on="fantasypros_id", how="left").sort_values("ecr").reset_index(drop=True)
    a["ADP"] = a.index + 1
    a["PosADP"] = a.groupby("pos").cumcount() + 1
    return a[["gsis_id", "player", "pos", "team", "ADP", "PosADP"]]


print("Loading actual results...")
aps = nfl.load_player_stats(seasons=SEASONS).to_pandas()
aps = aps[aps["season_type"] == "REG"]
actual = (aps.groupby(["player_id", "season"])
          .agg(g=("week", "nunique"), ppr=("fantasy_points_ppr", "sum")).reset_index())
actual["ppg"] = actual["ppr"] / actual["g"].clip(lower=1)
actual["season"] = actual["season"].astype(int)

parts = []
for Y in SEASONS:
    b = get_adp(Y)
    scores = ff_draft.value_scores(Y).drop(columns=["pos"])
    b = b.merge(scores, left_on="gsis_id", right_on="player_id", how="left")
    b["score_rank"] = b.groupby("pos")["score"].rank(ascending=False, method="first")
    b["Value"] = b["PosADP"] - b["score_rank"]
    fin = (actual[actual["season"] == Y][["player_id", "ppr", "ppg", "g"]]
           .rename(columns={"player_id": "gsis_id"}))
    b = b.merge(fin, on="gsis_id", how="left")
    played = b["ppr"].notna()
    b.loc[played, "FinishPos"] = b[played].groupby("pos")["ppr"].rank(ascending=False, method="first")
    b["BeatADP"] = b["PosADP"] - b["FinishPos"]
    b["season"] = Y
    parts.append(b)

board = pd.concat(parts, ignore_index=True).rename(columns={
    "player": "Player", "pos": "Pos", "team": "Team", "ppr": "PPR_actual",
    "ppg": "PPG_actual", "g": "G_actual"})
board = board[["Player", "Pos", "Team", "ADP", "PosADP", "Value", "PPR_actual",
               "PPG_actual", "G_actual", "FinishPos", "BeatADP", "season"]]

OUT.parent.mkdir(parents=True, exist_ok=True)
board.to_parquet(OUT, index=False)

print(f"\nDraft backtest saved: {OUT}  ({len(board)} player-seasons)")
print(f"{'season':>8}{'consensus rho':>16}{'value-lean rho':>16}{'n(draftable)':>14}")
for Y in SEASONS:
    s = board[board["season"] == Y]
    rho_adp = s["PosADP"].corr(s["FinishPos"], method="spearman")
    d = s[(s["ADP"] <= 150) & s["Value"].notna() & s["BeatADP"].notna()]
    rho_val = d["Value"].corr(d["BeatADP"], method="spearman")
    print(f"{Y:>8}{rho_adp:>16.3f}{rho_val:>16.3f}{len(d):>14}")
