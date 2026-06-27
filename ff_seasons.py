# ---------------------------------------------------------------
# ff_seasons.py  —  resolve which seasons to pull, robust to the
# offseason and the September season-label rollover.
# Returns 2020..current, but probes player stats and drops any
# trailing season that has no data yet (e.g. a new season label
# that appears before Week 1 has actually been played), so the
# build never crashes on a season nflverse hasn't published.
# ---------------------------------------------------------------

import nflreadpy as nfl

START = 2020


def resolve_seasons(start=START):
    candidate = list(range(start, nfl.get_current_season() + 1))
    while candidate:
        try:
            df = nfl.load_player_stats(seasons=candidate).to_pandas()
            if len(df):
                return candidate  # cached in-process, so the real build re-uses it free
        except Exception:
            pass
        candidate = candidate[:-1]  # newest season not published yet — drop and retry
    raise RuntimeError("no player-stat seasons available from nflreadpy")
