# ---------------------------------------------------------------
# app.py  —  Bellwether: fantasy projections dashboard
# USAGE: streamlit run app.py   (opens in your browser)
# Loads the feature table, runs the walk-forward backtest once
# (cached), and presents the results across eight pages (st.navigation):
# Overview, 2026 draft board, Draft track record, Weekly rankings, Why
# trust it, Accuracy vs experts, Projections vs actual, and Why it works.
# ---------------------------------------------------------------

import datetime as dt
import pathlib
import subprocess
import sys

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import ff_project
from ff_cache import CACHE, compute_core, data_version
from ff_model import load_eval_table, walk_forward_predictions, TARGET
from ff_eval import mae_table, mae_by_season, load_fp_rankings, rank_corr, POSITIONS

# --- Brand: Bellwether — Pine / Amber / Field / Chalk ---
PINE = "#11352A"
AMBER = "#EBC55C"
FIELD = "#1E5141"
CHALK = "#F0ECDD"
SAGE = "#8FB3A4"        # light sage — 3-game-avg baseline series
SAGE_DARK = "#5A7D6E"   # muted sage — naive baseline series
ASSETS = pathlib.Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Bellwether — Fantasy Projections",
                   page_icon=str(ASSETS / "icon-amber-on-pine.svg"), layout="wide")

# Brand type: Space Grotesk for UI/headings, Space Mono for data (metrics/code).
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: 'Space Grotesk', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.015em; }
    [data-testid="stMetricValue"], code, pre { font-family: 'Space Mono', monospace !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

LONG = {"avg3": "3-game avg", "model": "Bellwether", "FP_ecr": "FantasyPros",
        "naive_last": "Last game"}
COLORS = {"3-game avg": SAGE, "Last game": SAGE_DARK,
          "Bellwether": AMBER, "FantasyPros": CHALK}

PROJ = pathlib.Path(__file__).resolve().parent
DATA = PROJ / "data"

# --- Public-deployment config (set REPO_URL/SEASON_PASS_URL when live;
#     secrets come from .streamlit/secrets.toml locally or Streamlit Cloud) ---
REPO_URL = "https://github.com/Mitchsullivan56/bellwether"   # shown in "Why trust it"
SEASON_PASS_URL = ""     # season-pass link — shown once set
SIGNUPS_FILE = DATA / "signups.csv"


def _secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


BUTTONDOWN_KEY = _secret("BUTTONDOWN_API_KEY")
DEPLOYED = bool(BUTTONDOWN_KEY) or _secret("DEPLOYED") in ("1", "true", "True", True)


def data_age(name):
    p = DATA / name
    if not p.exists():
        return "missing"
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d, %H:%M")


def data_date(name):
    p = DATA / name
    if not p.exists():
        return "—"
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d, %Y")


def save_signup(email):
    """Add a subscriber to Buttondown when deployed; fall back to a local CSV in dev."""
    if BUTTONDOWN_KEY:
        try:
            r = requests.post(
                "https://api.buttondown.email/v1/subscribers",
                headers={"Authorization": f"Token {BUTTONDOWN_KEY}"},
                json={"email_address": email}, timeout=10)
            return r.status_code in (200, 201)
        except Exception:
            return False
    SIGNUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNUPS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{email},{dt.datetime.now().isoformat(timespec='seconds')}\n")
    return True


def refresh_data():
    """Re-pull nflverse + FantasyPros by running the two build scripts, then
    let the caller clear the cache so the model retrains on the new data."""
    steps = [("Pulling stats, lines, injuries, snaps, Next Gen, opportunity…", "build_dataset.py"),
             ("Pulling FantasyPros rankings…", "fetch_fantasypros.py"),
             ("Building the 2026 draft board…", "build_draft_board.py"),
             ("Testing past-draft accuracy…", "backtest_draft.py"),
             ("Recomputing feature importance…", "analyze_importance.py")]
    with st.status("Refreshing from nflverse + FantasyPros…", expanded=True) as status:
        for msg, script in steps:
            st.write(msg)
            r = subprocess.run([sys.executable, script], cwd=str(PROJ),
                               capture_output=True, text=True)
            if r.returncode != 0:
                status.update(label="Refresh failed", state="error")
                st.code((r.stderr or r.stdout or "unknown error")[-3000:])
                return False
        status.update(label="Data refreshed — retraining…", state="complete")
    return True


@st.cache_data(show_spinner="Training the model and scoring its accuracy…")
def compute():
    version = data_version()
    if CACHE.exists():                              # committed / pre-warmed cache -> instant
        try:
            blob = joblib.load(CACHE)
            if blob.get("version") == version:
                return blob["data"]
        except Exception:
            pass
    out = compute_core()
    try:                                            # no-op on read-only hosts (Streamlit Cloud)
        joblib.dump({"version": version, "data": out}, CACHE)
    except Exception:
        pass
    return out


def grouped_bar(df, value_cols, x="Position", title="", y_title="", reverse=False):
    long = df.melt(id_vars=[x], value_vars=value_cols, var_name="method", value_name="val")
    long["method"] = long["method"].map(LONG).fillna(long["method"])
    fig = px.bar(long, x=x, y="val", color="method", barmode="group",
                 color_discrete_map=COLORS, title=title, template="plotly_dark")
    fig.update_layout(yaxis_title=y_title, xaxis_title="", legend_title="",
                      margin=dict(t=40, b=0, l=0, r=0), height=380,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=CHALK, family="Space Grotesk, sans-serif"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(240,236,221,0.12)")
    if reverse:  # for MAE, lower is better — flip so taller still reads as "good"
        fig.update_yaxes(autorange="reversed")
    return fig


FEATURE_LABELS = {
    "pts_season": "Season scoring avg", "pts_avg5": "Last-5 scoring avg",
    "pts_avg3": "Last-3 scoring avg", "pts_last1": "Last game points",
    "games_so_far": "Games played",
    "offense_pct_avg3": "Snap share", "fp_exp_avg3": "Expected pts (opportunity)",
    "fp_over_exp_avg3": "Points over expected", "target_share_avg3": "Target share",
    "air_yards_share_avg3": "Air-yards share", "attempts_avg3": "Pass attempts",
    "carries_avg3": "Carries", "completions_avg3": "Completions",
    "targets_avg3": "Targets", "receptions_avg3": "Receptions",
    "rushing_yards_avg3": "Rushing yards", "receiving_yards_avg3": "Receiving yards",
    "passing_yards_avg3": "Passing yards", "passing_tds_avg3": "Passing TDs",
    "rushing_tds_avg3": "Rushing TDs", "receiving_tds_avg3": "Receiving TDs",
    "team_implied_total": "Vegas team total", "opp_implied_total": "Vegas opp. total",
    "game_total": "Vegas game total", "team_spread": "Vegas spread",
    "def_pts_allowed_avg3": "Opp. defense allowed", "injury_status": "Injury status",
    "is_home": "Home / away", "rest_days": "Rest days", "roof": "Roof",
    "temp": "Temperature", "wind": "Wind", "position": "Position",
    "opponent_team": "Opponent",
    "completion_percentage_above_expectation_avg3": "CPOE (QB)",
    "avg_time_to_throw_avg3": "Time to throw (QB)", "aggressiveness_avg3": "Aggressiveness (QB)",
    "avg_air_yards_differential_avg3": "Air-yards differential (QB)",
    "efficiency_avg3": "Rush efficiency", "rush_yards_over_expected_per_att_avg3": "Rush yds over expected",
    "percent_attempts_gte_eight_defenders_avg3": "Stacked-box rate",
    "avg_separation_avg3": "Separation (WR/TE)", "avg_cushion_avg3": "Cushion (WR/TE)",
    "avg_yac_above_expectation_avg3": "YAC over expected",
    "percent_share_of_intended_air_yards_avg3": "Intended air-yards share",
}
OPPORTUNITY = {"offense_pct_avg3", "fp_exp_avg3", "target_share_avg3",
               "air_yards_share_avg3", "attempts_avg3", "carries_avg3",
               "completions_avg3", "targets_avg3", "receptions_avg3"}
MARKET = {"team_implied_total", "opp_implied_total", "game_total", "team_spread"}
FORM = {"pts_season", "pts_avg5", "pts_avg3", "pts_last1", "games_so_far"}
CONTEXT = {"roof", "temp", "wind", "rest_days", "is_home", "injury_status",
           "position", "opponent_team"}


def feature_label(f):
    return FEATURE_LABELS.get(f, f.replace("_avg3", "").replace("_", " ").strip().capitalize())


def feature_theme(f):
    if f in OPPORTUNITY:
        return "Opportunity / role"
    if f in MARKET:
        return "Market (Vegas)"
    if f in FORM:
        return "Recent form"
    if f == "def_pts_allowed_avg3":
        return "Matchup (defense)"
    if f in CONTEXT:
        return "Context"
    if "yards_avg3" in f or "tds_avg3" in f:
        return "Recent production"
    return "Efficiency (Next Gen)"


D = compute()


@st.cache_data(show_spinner="Projecting the upcoming slate…")
def live_board():
    return ff_project.project_live()


def draft_board():
    # not cached: the file is tiny, and reading fresh avoids stale-cache crashes
    # when the board is rebuilt outside the running app.
    p = DATA / "draft_board.parquet"
    return pd.read_parquet(p) if p.exists() else None


def draft_backtest():
    p = DATA / "draft_backtest.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ================= Pages =================
def page_overview():
    summary = D["rank_summary"].set_index("Position")
    all_model, all_fp = summary.loc["ALL", "model"], summary.loc["ALL", "FP_ecr"]
    wins = [p for p in POSITIONS if summary.loc[p, "model"] >= summary.loc[p, "FP_ecr"]]
    mae_all = D["mae"].set_index("Position").loc["ALL"]
    n_pw = len(D["result"])
    s0, s1 = D["test_seasons"][0], D["test_seasons"][-1]
    pct_avg3 = (mae_all["avg3"] - mae_all["model"]) / mae_all["avg3"] * 100
    pct_naive = (mae_all["naive_last"] - mae_all["model"]) / mae_all["naive_last"] * 100

    # --- Headline accuracy number, front and center ---
    st.markdown(
        f"""
        <div style="background:{FIELD};border-radius:14px;padding:1.3rem 1.6rem;margin-bottom:1.1rem">
          <div style="font-size:0.9rem;letter-spacing:.04em;text-transform:uppercase;
                      color:{CHALK};opacity:.7">Out-of-sample accuracy · {s0}–{s1}</div>
          <div style="font-size:2.7rem;font-weight:700;color:{AMBER};line-height:1.1">
            {pct_avg3:.0f}% lower error than a trailing-average baseline</div>
          <div style="font-size:1rem;color:{CHALK};opacity:.9;margin-top:.3rem">
            Bellwether's mean absolute error is {mae_all['model']:.2f} pts vs
            {mae_all['avg3']:.2f} for a 3-game average ({pct_naive:.0f}% better than a
            last-game baseline) — measured on {n_pw:,} player-weeks it never trained on.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Can a from-scratch model beat the experts?")
    st.markdown(
        "FantasyPros only archives expert-consensus **rankings** historically, so the "
        "honest head-to-head metric is **rank-order accuracy**: each week, rank every "
        "player by projected points and correlate that order with how they actually "
        "finished (Spearman's ρ). Higher = better at sorting players correctly."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Positions where Bellwether beats experts",
              f"{len(wins)} of {len(POSITIONS)}",
              ", ".join(wins) if wins else "none yet")
    c2.metric("Bellwether rank accuracy (ρ, all positions)", f"{all_model:.3f}",
              f"{all_model - all_fp:+.3f} vs experts")
    c3.metric("Bellwether points error (MAE, all positions)", f"{mae_all['model']:.2f}",
              f"{mae_all['model'] - mae_all['avg3']:+.2f} vs 3-game avg",
              delta_color="inverse")

    st.subheader("Rank accuracy vs FantasyPros experts")
    st.plotly_chart(
        grouped_bar(D["rank_summary"][D["rank_summary"]["Position"] != "ALL"],
                    ["avg3", "model", "FP_ecr"], y_title="Spearman ρ vs actual finish"),
        width="stretch")
    if wins:
        st.success(f"**Bellwether beats expert consensus at: {', '.join(wins)}.** "
                   "Elsewhere it's closing the gap — QB stays hardest because its "
                   "fantasy output is rushing/garbage-time driven and experts price in "
                   "late-breaking starter news the model can't see.")
    else:
        st.info("Experts still lead at every position, but the model beats the "
                "3-game-average baseline — see Accuracy vs experts for the gap.")

    st.markdown("---")
    st.subheader("Get the board in your inbox")
    st.caption("One email when the draft board updates and when weekly rankings go live. "
               "No spam.")
    with st.form("signup", clear_on_submit=True):
        fc1, fc2 = st.columns([3, 1])
        email = fc1.text_input("Email", placeholder="you@email.com",
                               label_visibility="collapsed")
        submitted = fc2.form_submit_button("Notify me", width="stretch")
    if submitted and email and "@" in email and "." in email:
        if save_signup(email.strip()):
            st.success("You're on the list.")
        else:
            st.warning("Hmm, that didn't go through — please try again in a moment.")
    elif submitted:
        st.warning("Please enter a valid email.")
    if SEASON_PASS_URL:
        st.caption(f"Going deeper this year? **[Bellwether season pass →]({SEASON_PASS_URL})**")
    else:
        st.caption("A season pass (full weekly projections + lineup tools) is coming.")


def page_draft():
    st.title("2026 draft board — PPR")
    board = draft_board()
    if board is None or "Tier" not in board.columns:
        st.info("Draft board not built (or out of date) — hit **Refresh data** in the "
                "sidebar to (re)build it.")
        return
    prior = int(board["prior_season"].iloc[0]) if "prior_season" in board.columns else 2025
    ppg_c, xppg_c = f"PPG_{prior}", f"xPPG_{prior}"
    ppg_l, xppg_l = f"{prior} PPG", f"{prior} xPPG"
    as_of = pd.to_datetime(board["as_of"].iloc[0])
    st.caption(f"🕒 Updated **{as_of:%b %d, %Y}** · {len(board)} players · PPR scoring")
    st.caption("Ranked by current PPR expert consensus, enriched with Bellwether's "
               "expected-production lens. Pre-draft we can't beat the consensus (testing past "
               "drafts shows that's near-impossible from box-score data), so the board is "
               "**ranked by consensus** and **Value** is a soft 'expected-production vs ADP' "
               "flag — not a predicted edge. Hit **Refresh data** for the latest; once the "
               "season starts, **Weekly rankings** take over.")
    pos = st.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True,
                   key="draft_pos")
    full = board.reset_index(drop=True)
    full.insert(0, "Rank", range(1, len(full) + 1))
    view = (full if pos == "All" else full[full["Pos"] == pos]).copy()
    view["Basis"] = view[ppg_c].notna().map({True: "model", False: "situational"})
    show = view[["Rank", "Tier", "Player", "PosRank", "Team", "Bye",
                 ppg_c, xppg_c, "Value", "Basis"]].rename(
        columns={"PosRank": "Pos", ppg_c: ppg_l, xppg_c: xppg_l, "Value": "Value vs ADP"})

    def _val_color(v):
        if pd.isna(v):
            return ""
        return f"color: {SAGE}; font-weight: 600" if v > 0 else (
            "color: #C98A8A; font-weight: 600" if v < 0 else "")

    def _tier_band(row):     # shade alternating tiers so the draft cliffs are visible
        shade = "background-color: rgba(240,236,221,0.05)" if int(row["Tier"]) % 2 == 0 else ""
        return [shade] * len(row)

    sty = (show.style
           .format({ppg_l: "{:.1f}", xppg_l: "{:.1f}", "Bye": "{:.0f}",
                    "Tier": "{:.0f}", "Value vs ADP": "{:+.0f}"}, na_rep="—")
           .apply(_tier_band, axis=1)
           .bar(subset=[xppg_l], color=AMBER, align="left")
           .map(_val_color, subset=["Value vs ADP"]))
    st.dataframe(sty, width="stretch", hide_index=True, height=620)
    st.caption(f"Rank = consensus ADP · **Tier** = within-position draft tier (shaded bands = "
               f"the cliffs) · {prior} xPPG = expected pts/game from opportunity · "
               "**Value vs ADP** = consensus rank minus Bellwether's expected-production rank "
               "(2-yr, regression-adjusted). A soft tilt (ρ≈0.07 across 2022–25), not a verdict. "
               "· **Basis** = *model* (returning player, data-driven) vs *situational* "
               "(rookie / no recent data — leans on draft capital & role, where preseason "
               "boards are weakest).")


def page_results():
    st.title("Draft track record — did the value lean beat ADP?")
    bt = draft_backtest()
    if bt is None or "season" not in bt.columns:
        st.info("Backtest not built yet — hit **Refresh data** in the sidebar.")
        return
    seasons = sorted(int(s) for s in bt["season"].unique())
    st.markdown(
        f"A fair test across **{seasons[0]}–{seasons[-1]}**: each season the board is "
        "rebuilt from only what was known beforehand (that year's preseason PPR consensus + "
        "the 2-year regression-adjusted expected-production score), then scored against what "
        "actually happened. Two questions: how good was the *ranking*, and did Bellwether's "
        "**Value lean** add anything?"
    )

    rows = []
    for y in seasons:
        s = bt[bt["season"] == y]
        d = s[(s["ADP"] <= 150) & s["Value"].notna() & s["BeatADP"].notna()]
        rows.append({"Season": y,
                     "Consensus ρ": s["PosADP"].corr(s["FinishPos"], method="spearman"),
                     "Value-lean ρ": d["Value"].corr(d["BeatADP"], method="spearman"),
                     "n (draftable)": len(d)})
    summ = pd.DataFrame(rows)
    cons_avg, val_avg = summ["Consensus ρ"].mean(), summ["Value-lean ρ"].mean()
    pos_yrs = int((summ["Value-lean ρ"] > 0).sum())

    c1, c2 = st.columns(2)
    c1.metric("Consensus ranking accuracy", f"ρ {cons_avg:.2f}",
              "strong & stable every season")
    c2.metric("Value-lean edge (ADP ≤ 150)", f"ρ {val_avg:.2f}",
              "weak — a soft tilt at best", delta_color="off")

    st.warning(
        f"**Honest verdict:** the *ranking* (expert consensus) is strong and stable "
        f"(ρ≈{cons_avg:.2f} every year). Bellwether's *Value lean* is only a **weak tilt** "
        f"(ρ≈{val_avg:.2f}, positive in {pos_yrs} of {len(summ)} seasons). That's the real "
        "ceiling: you can't beat a strong consensus on the draft residual from box-score "
        "data — what's left is offseason role/news the market sees and we don't. So trust "
        "the board's order, and treat **Value** as a soft 'expected-production vs ADP' flag, "
        "not a directive. Publishing that honestly is the point."
    )

    st.subheader("Season by season")
    st.dataframe(summ.style.format({"Consensus ρ": "{:.2f}", "Value-lean ρ": "{:+.2f}"}),
                 width="stretch", hide_index=True)

    latest = seasons[-1]
    d = bt[(bt["season"] == latest) & (bt["ADP"] <= 150) &
           bt["Value"].notna() & bt["BeatADP"].notna()]
    st.subheader(f"{latest}: draft slot vs actual finish (ADP ≤ 150)")
    m = int(max(d["PosADP"].max(), d["FinishPos"].max()))
    sc = d.assign(Lean=d["Value"].clip(-30, 30))
    fig = px.scatter(sc, x="PosADP", y="FinishPos", color="Lean", hover_name="Player",
                     color_continuous_scale=["#C98A8A", "#F0ECDD", "#8FB3A4"],
                     template="plotly_dark")
    fig.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(color=CHALK, dash="dot"))
    fig.update_traces(marker=dict(size=8, line=dict(width=0)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=CHALK, family="Space Grotesk, sans-serif"),
                      margin=dict(t=10, b=0, l=0, r=0), height=440,
                      xaxis_title="preseason positional ADP",
                      yaxis_title=f"actual {latest} finish")
    fig.update_xaxes(gridcolor="rgba(240,236,221,0.12)")
    fig.update_yaxes(gridcolor="rgba(240,236,221,0.12)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Below the dotted line = beat the draft slot. Color = Bellwether's preseason "
               "lean (green = liked more). The tilt is faint — green leans slightly below "
               "the line, but consensus does nearly all the work.")

    st.subheader(f"{latest}: Bellwether's boldest draftable calls")
    cols = ["Player", "Pos", "PosADP", "FinishPos", "Value", "BeatADP"]
    fmt = {"PosADP": "{:.0f}", "FinishPos": "{:.0f}", "Value": "{:+.0f}", "BeatADP": "{:+.0f}"}
    h, m2 = st.columns(2)
    h.markdown("**Hits** — liked more, beat ADP")
    h.dataframe((d[d["Value"] > 0].sort_values("BeatADP", ascending=False)
                 .head(8)[cols].style.format(fmt)), width="stretch", hide_index=True)
    m2.markdown("**Misses** — liked more, flopped")
    m2.dataframe((d[d["Value"] > 0].sort_values("BeatADP")
                  .head(8)[cols].style.format(fmt)), width="stretch", hide_index=True)


def page_weekly():
    st.title("Weekly rankings")
    live = live_board()
    if live is None:
        st.info("The 2026 NFL season hasn't started, so weekly rankings are empty for now. "
                "Once games are played, this page shows Bellwether's projected points for "
                "the upcoming week — ranked, by position — to set your lineup (reliable from "
                "about Week 4, once there's enough in-season usage data). Until then, use "
                "the **2026 draft board**. In-season, hit **Refresh data** each week to "
                "update.")
        return
    board, s, w = live
    st.caption(f"🕒 Updated **{data_date('model_table.parquet')}** · {s} Week {w} · PPR")
    st.caption(f"Live Bellwether projections for {s} Week {w} (PPR) — rank your options "
               "and set your lineup.")
    pos = st.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True,
                   key="weekly_pos")
    view = (board if pos == "All" else board[board["Pos"] == pos]).reset_index(drop=True)
    view.insert(0, "Rank", range(1, len(view) + 1))
    sty = view.style.format({"Projected": "{:.1f}"}).bar(
        subset=["Projected"], color=AMBER, align="left")
    st.dataframe(sty, width="stretch", hide_index=True, height=560)
    label = "players" if pos == "All" else f"{pos}s"
    st.caption(f"{len(view)} {label} · ranked by Bellwether projected PPR points.")


def page_trust():
    summary = D["rank_summary"].set_index("Position")
    wins = [p for p in POSITIONS if summary.loc[p, "model"] >= summary.loc[p, "FP_ecr"]]
    all_model, all_fp = summary.loc["ALL", "model"], summary.loc["ALL", "FP_ecr"]
    mae_all = D["mae"].set_index("Position").loc["ALL"]
    n_pw = len(D["result"])
    s0, s1 = D["test_seasons"][0], D["test_seasons"][-1]

    st.title("Why trust Bellwether?")
    st.markdown(
        "Most \"I beat the experts\" projects are overfit, or quietly leak the future "
        "into the past. Bellwether's claim is built to survive scrutiny: every number "
        "in this app is **out-of-sample**, benchmarked against a real standard, and "
        "explainable. Here's the case."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Out-of-sample player-weeks", f"{n_pw:,}")
    c2.metric("Seasons back-tested", f"{s0}–{s1}")
    c3.metric("Beats expert consensus at", ", ".join(wins) if wins else "—")

    st.subheader("1 · Evaluated without hindsight")
    st.markdown(
        "- **Walk-forward.** Each season is predicted using only *earlier* seasons for "
        "training — the model is never scored on data it has already seen.\n"
        "- **Leakage-free features.** Every trailing stat for week *w* uses weeks 1…*w*-1 "
        "only; the pre-game inputs (Vegas lines, injury report) are genuinely known "
        "before kickoff. It never peeks at the box score it is trying to predict.\n"
        "- **Honest tuning.** Hyperparameters are chosen on an inner validation season, "
        "never the test season — so the reported accuracy is not inflated by tuning to "
        "the answer."
    )

    st.subheader("2 · Measured against a real standard, not a strawman")
    st.markdown(
        "- Scored **head-to-head against FantasyPros expert consensus** (an aggregate of "
        "100+ analysts) on the *same* weeks and the *same* actual outcomes.\n"
        "- Also clears two deliberately hard baselines — last game and 3-game average.\n"
        f"- The gap is **quantified and shown**, not asserted: overall rank accuracy "
        f"ρ = {all_model:.3f} vs {all_fp:.3f} for the experts, and {mae_all['model']:.2f} "
        f"MAE vs {mae_all['avg3']:.2f} for the 3-game baseline. See **Accuracy vs experts**."
    )

    st.subheader("3 · What makes it different")
    left, right = st.columns(2)
    left.markdown(
        "**Typical consensus projections**\n\n"
        "- A blend of analyst opinion — strong, but a **black box** you cannot audit\n"
        "- Carry **reputation inertia**: name brands and preseason ADP linger for weeks\n"
        "- Rarely published *with* an out-of-sample accuracy track record"
    )
    right.markdown(
        "**Bellwether**\n\n"
        "- **No reputation bias** — ranks players on *measured role*, not name recognition\n"
        "- Leans on **opportunity** — snap share, target share, and expected points with "
        "TD luck stripped out — which updates every week\n"
        "- **Fully back-tested and explainable** — see what drives it in *Why it works*"
    )
    st.success(
        "**The edge in one sentence:** when a player's usage changes, Bellwether sees it "
        "immediately in opportunity signals, while reputation-driven consensus reprices "
        "slowly — and that lag is largest at **RB**, which is exactly where it wins."
    )

    st.subheader("4 · Honest about the limits")
    st.markdown(
        "- It **loses to experts at QB** and is close at WR/TE — and the app shows you "
        "exactly where, by position and season, instead of hiding it.\n"
        "- It **cannot see late-breaking news** (surprise inactives, beat-writer intel) "
        "the morning of games that consensus prices in.\n"
        "- Fantasy scoring has large **irreducible variance** (TD luck, mid-game "
        "injuries); no model is \"optimal.\" Bellwether targets the predictable part — "
        "opportunity — and is transparent about the rest.\n\n"
        "*A model that tells you where it is worse is easier to trust on where it is better.*"
    )

    st.subheader("5 · The method, in plain English")
    st.markdown(
        "1. **Pull** every player-week, 2020–2025, from nflverse — box-score stats, snap "
        "counts, injury reports, Vegas lines, Next Gen Stats, and expected-points "
        "(opportunity) data.\n"
        "2. **Engineer leakage-free features** — every number used to predict week *w* "
        "comes only from weeks before it (or pre-game info like the betting line).\n"
        "3. **Walk forward** — train on past seasons, predict the next; the model is never "
        "tested on data from its own future.\n"
        "4. **Tune honestly** — pick settings on an inner validation season, so the "
        "headline accuracy isn't fit to the test answer.\n"
        "5. **Score** against the real results *and* the FantasyPros consensus on the same "
        "weeks — then publish the gap, win or lose.\n"
        "6. **Explain** every prediction with permutation importance (see *Why it works*)."
    )
    if REPO_URL:
        st.markdown(f"**[Read the full source on GitHub →]({REPO_URL})**")
    else:
        st.caption("Source code is private for now — available on request, and the "
                   "methodology above is the whole approach, nothing hidden.")


def page_backtest():
    st.title("Accuracy vs experts")
    st.caption("Every season is predicted using only prior seasons as training data "
               "(walk-forward); hyperparameters are tuned on an inner validation "
               "season, never on the test season.")

    st.subheader("Rank accuracy — Spearman ρ vs actual finish (higher = better)")
    win_style = f"background-color: {AMBER}; color: {PINE}; font-weight: 700;"
    rank_disp = D["rank_summary"].rename(columns={
        "avg3": "3-game avg", "model": "Bellwether", "FP_ecr": "FantasyPros"})
    styled = (rank_disp.style
              .format({"3-game avg": "{:.3f}", "Bellwether": "{:.3f}", "FantasyPros": "{:.3f}"})
              .apply(lambda r: [win_style if c == "Bellwether" and
                                r["Bellwether"] >= r["FantasyPros"] else "" for c in r.index], axis=1))
    st.dataframe(styled, width="stretch", hide_index=True)

    st.subheader("Points error — MAE (lower = better)")
    st.plotly_chart(
        grouped_bar(D["mae"][D["mae"]["Position"] != "ALL"],
                    ["naive_last", "avg3", "model"],
                    y_title="mean absolute error (pts)", reverse=True),
        width="stretch")
    mae_disp = D["mae"].rename(columns={
        "naive_last": "Last game", "avg3": "3-game avg", "model": "Bellwether"})
    st.dataframe(
        mae_disp.style.format({"Last game": "{:.2f}", "3-game avg": "{:.2f}",
                               "Bellwether": "{:.2f}", "n": "{:,}"}),
        width="stretch", hide_index=True)

    st.subheader("Bellwether MAE by test season")
    st.plotly_chart(
        grouped_bar(D["mae_season"], ["naive_last", "avg3", "model"], x="season",
                    y_title="mean absolute error (pts)", reverse=True),
        width="stretch")

    st.subheader("Calibration — do the projected points match reality?")
    res = D["result"][["pred", TARGET]].copy()
    res["bin"] = pd.qcut(res["pred"], 10, labels=False, duplicates="drop")
    cal = (res.groupby("bin").agg(projected=("pred", "mean"), actual=(TARGET, "mean"),
                                  n=("pred", "size")).reset_index(drop=True))
    mx = float(max(cal["projected"].max(), cal["actual"].max())) + 1
    figc = px.scatter(cal, x="projected", y="actual", size="n", template="plotly_dark")
    figc.update_traces(marker=dict(color=AMBER, line=dict(width=0)))
    figc.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color=CHALK, dash="dot"))
    figc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color=CHALK, family="Space Grotesk, sans-serif"),
                       margin=dict(t=10, b=0, l=0, r=0), height=420,
                       xaxis_title="projected points (decile mean)",
                       yaxis_title="actual points (decile mean)")
    figc.update_xaxes(gridcolor="rgba(240,236,221,0.12)")
    figc.update_yaxes(gridcolor="rgba(240,236,221,0.12)")
    st.plotly_chart(figc, width="stretch")
    st.caption("Each dot is a decile of projections (size = player-weeks). On the dotted "
               "line = perfectly calibrated — a 15-point projection really does average ~15 "
               "actual points. Bellwether tracks the line at every decile (you can trust the "
               "*magnitude*, not just the rank), with only the top decile running a hair hot "
               "— the usual mild regression at the extreme.")


def page_projections():
    st.title("Projections vs actual, week by week")
    st.caption("For any historical week: what the model projected, how experts ranked "
               "the player, and what actually happened.")
    m = D["merged"]
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season", sorted(m["season"].unique(), reverse=True))
    weeks = sorted(m[m["season"] == season]["week"].unique())
    week = c2.selectbox("Week", weeks, index=len(weeks) - 1)
    pos = c3.selectbox("Position", POSITIONS)

    view = m[(m["season"] == season) & (m["week"] == week) & (m["position"] == pos)].copy()
    view = view.sort_values("pred", ascending=False)
    view["model_rank"] = range(1, len(view) + 1)
    view["fp_rank"] = view["ecr"].rank(method="min").astype(int)
    table = view[["model_rank", "fp_rank", "player_display_name", "pred",
                  "pts_avg3", TARGET]].rename(columns={
        "model_rank": "Bellwether rank", "fp_rank": "FantasyPros rank",
        "player_display_name": "Player", "pred": "Bellwether proj",
        "pts_avg3": "3-game avg", TARGET: "Actual"})
    st.dataframe(
        table.style.format({"Bellwether proj": "{:.1f}", "3-game avg": "{:.1f}",
                            "Actual": "{:.1f}"})
        .bar(subset=["Actual"], color=AMBER, align="left"),
        width="stretch", hide_index=True, height=560)
    st.caption(f"{len(view)} {pos}s · sorted by Bellwether projection. "
               "Compare 'Bellwether rank' vs 'FantasyPros rank' against the 'Actual' column.")


def page_why():
    st.title("Why the model is accurate")
    imp = D["importance"]
    if imp is None:
        st.info("Run `python analyze_importance.py` to generate the importance table.")
        return
    last_season = int(D["test_seasons"][-1])
    st.markdown(
        "**Permutation importance** measures how much the model's error (MAE) gets "
        f"worse when each signal is shuffled out — measured on the held-out {last_season} "
        "season, so it reflects what the model truly leans on (not just how often it "
        "split on a feature). Units are fantasy points of MAE lost."
    )
    top = imp.head(12).copy()
    top["label"] = top["feature"].map(feature_label)
    fig = px.bar(top.iloc[::-1], x="mae_cost", y="label", orientation="h",
                 template="plotly_dark")
    fig.update_traces(marker_color=AMBER)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=CHALK, family="Space Grotesk, sans-serif"),
                      margin=dict(t=10, b=0, l=0, r=0), height=460,
                      xaxis_title="MAE points lost when shuffled (higher = relied on more)",
                      yaxis_title="")
    fig.update_xaxes(gridcolor="rgba(240,236,221,0.12)")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")

    themed = imp.copy()
    themed["theme"] = themed["feature"].map(feature_theme)
    roll = themed.groupby("theme")["mae_cost"].sum().sort_values(ascending=False)
    st.success(
        "**Where the edge comes from.** After the obvious prior — how much a player "
        "normally scores (*season scoring avg*) — the model's accuracy leans on "
        "**opportunity** (snap share, expected points, target share, raw volume) and the "
        "**Vegas game environment** (implied team total). These are *role* signals that "
        "move week to week, so when a player's usage changes the model sees it "
        "immediately — while reputation-driven consensus rankings reprice slowly. "
        "That lag is the gap Bellwether exploits."
    )
    st.caption("Signal families by total importance: " +
               " · ".join(f"{k} {v:.2f}" for k, v in roll.items()))


# ================= Sidebar + navigation =================
with st.sidebar:
    bc1, bc2 = st.columns([1, 4], vertical_alignment="center")
    bc1.image(str(ASSETS / "mark-amber.svg"), width=46)
    bc2.markdown("<div style=\"font-family:'Space Grotesk',sans-serif;font-size:2rem;"
                 "font-weight:700;color:#F0ECDD;letter-spacing:-0.02em;line-height:1.05\">"
                 "Bellwether</div>", unsafe_allow_html=True)
    st.caption("Data-driven fantasy projections, tested against the experts.")

nav = st.navigation({
    "Rankings": [
        st.Page(page_draft, title="2026 draft board",
                icon=":material/format_list_numbered:", default=True, url_path="draft"),
        st.Page(page_results, title="Draft track record",
                icon=":material/history:", url_path="results"),
        st.Page(page_weekly, title="Weekly rankings",
                icon=":material/calendar_month:", url_path="weekly"),
    ],
    "The model": [
        st.Page(page_overview, title="Overview", icon=":material/insights:",
                url_path="overview"),
        st.Page(page_trust, title="Why trust it", icon=":material/verified:",
                url_path="trust"),
        st.Page(page_backtest, title="Accuracy vs experts", icon=":material/bar_chart:",
                url_path="backtest"),
        st.Page(page_projections, title="Projections vs actual",
                icon=":material/table_rows:", url_path="projections"),
        st.Page(page_why, title="Why it works", icon=":material/lightbulb:",
                url_path="why"),
    ],
})

with st.sidebar:
    st.markdown("---")
    seasons = D["test_seasons"]
    st.caption(f"Walk-forward test seasons: {seasons[0]}–{seasons[-1]}  ·  "
               f"{len(D['result']):,} player-weeks scored")
    st.caption(f"Data updated · model: {data_age('model_table.parquet')} · "
               f"experts: {data_age('fp_rankings.parquet')}")
    if DEPLOYED:
        st.caption("↻ Rankings refresh automatically each day.")
    elif st.button("↻ Refresh data",
                   help="Re-pull nflverse + FantasyPros and retrain. Takes a few minutes."):
        if refresh_data():
            st.cache_data.clear()
            st.rerun()

nav.run()
