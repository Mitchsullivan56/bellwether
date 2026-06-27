# ---------------------------------------------------------------
# ff_draft.py  —  shared draft-value logic.
# Value basis = a 2-year, regression-adjusted EXPECTED-fantasy-points
# per game score (the most stable of the formulas backtested across
# 2022-2025 — see backtest_draft.py). It is a soft "expected
# production vs ADP" lens, NOT a predicted edge: beating expert
# consensus on the draft residual is near-impossible from box-score
# data (the backtest shows ~0 correlation). Used by both the live
# draft board and the multi-season backtest so they never diverge.
# ---------------------------------------------------------------

import re

import numpy as np
import pandas as pd
import nflreadpy as nfl

POSITIONS = ["QB", "RB", "WR", "TE"]
SHRINK_K = 6.0   # games of positional-mean prior mixed in (regression to the mean)


def norm_name(name):
    """Normalize a player name for fallback matching (lowercase, drop suffixes/punct)."""
    s = re.sub(r"[.'’]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def _rookie_scores(draft_season):
    """Expected-PPG prior for incoming rookies from NFL draft capital — historical
    rookie PPG by position x draft round, so rookies get a score instead of a blank."""
    ids = nfl.load_ff_playerids().to_pandas()[["gsis_id", "name", "position", "draft_year", "draft_round"]]
    ids = ids.dropna(subset=["gsis_id"]).copy()
    ids["draft_year"] = pd.to_numeric(ids["draft_year"], errors="coerce")
    ids["rd"] = pd.to_numeric(ids["draft_round"], errors="coerce").fillna(8).clip(upper=8)

    ps = nfl.load_player_stats(seasons=list(range(2020, draft_season))).to_pandas()
    ps = ps[ps["season_type"] == "REG"]
    p = (ps.groupby(["player_id", "season"])
         .agg(g=("week", "nunique"), ppr=("fantasy_points_ppr", "sum")).reset_index())
    p["ppg"] = p["ppr"] / p["g"].clip(lower=1)
    p["season"] = p["season"].astype(int)
    p = p.merge(ids.rename(columns={"gsis_id": "player_id"}), on="player_id", how="inner")
    rook = p[(p["season"] == p["draft_year"]) & p["position"].isin(POSITIONS)]   # rookie seasons
    by = rook.groupby(["position", "rd"])["ppg"].mean()
    posmean = rook.groupby("position")["ppg"].mean()

    new = ids[(ids["draft_year"] == draft_season) & ids["position"].isin(POSITIONS)].copy()
    new["score"] = new.apply(
        lambda r: by.get((r["position"], r["rd"]), posmean.get(r["position"])), axis=1)
    new["merge_name"] = new["name"].map(norm_name)
    return new.rename(columns={"gsis_id": "player_id", "position": "pos"})[
        ["player_id", "pos", "score", "merge_name"]]


def _prod(seasons):
    ps = nfl.load_player_stats(seasons=seasons).to_pandas()
    ps = ps[ps["season_type"] == "REG"]
    p = (ps.groupby(["player_id", "season"])
         .agg(g=("week", "nunique"), ppr=("fantasy_points_ppr", "sum"),
              pos=("position", "first")).reset_index())
    p["ppg"] = p["ppr"] / p["g"].clip(lower=1)
    p["season"] = p["season"].astype(int)
    opp = nfl.load_ff_opportunity(seasons=seasons).to_pandas()
    o = (opp.groupby(["player_id", "season"])
         .agg(xg=("week", "nunique"), xpts=("total_fantasy_points_exp", "sum")).reset_index())
    o["xppg"] = o["xpts"] / o["xg"].clip(lower=1)
    o["season"] = o["season"].astype(int)
    return p.merge(o[["player_id", "season", "xppg"]], on=["player_id", "season"], how="left")


def value_scores(draft_season):
    """Per-player expected-production score for `draft_season`, built from the
    two prior seasons (regression-adjusted, 2-year-weighted expected PPG).
    Returns player_id, pos, score, plus prior-year context (ppg/xppg/g)."""
    prior, prior2 = draft_season - 1, draft_season - 2
    p = _prod([prior2, prior])
    n = (p[p["season"] == prior][["player_id", "pos", "g", "xppg", "ppg"]]
         .rename(columns={"g": "g_prior", "xppg": "xppg_prior", "ppg": "ppg_prior"}))
    n1 = p[p["season"] == prior2][["player_id", "xppg"]].rename(columns={"xppg": "xppg_2yr"})
    s = n.merge(n1, on="player_id", how="left")
    s = s[s["pos"].isin(POSITIONS)].copy()
    s["base2"] = 0.6 * s["xppg_prior"] + 0.4 * s["xppg_2yr"].fillna(s["xppg_prior"])
    pm = s.groupby("pos")["base2"].transform("mean")
    s["score"] = (s["g_prior"] * s["base2"] + SHRINK_K * pm) / (s["g_prior"] + SHRINK_K)
    ret = s[["player_id", "pos", "score", "ppg_prior", "xppg_prior", "g_prior"]].copy()
    ret["merge_name"] = np.nan        # returning players match via gsis_id

    # rookies (no NFL history) get a draft-capital prior on the same PPG scale.
    # merge_name lets the live board link brand-new rookies the ID crosswalk misses.
    rk = _rookie_scores(draft_season)
    rk = rk[~rk["player_id"].isin(ret["player_id"])].copy()
    for c in ["ppg_prior", "xppg_prior", "g_prior"]:
        rk[c] = np.nan
    return pd.concat([ret, rk[ret.columns]], ignore_index=True)
