# ---------------------------------------------------------------
# test_app.py  —  smoke tests run by CI on every push (and locally).
# Confirms the brand assets are valid, the committed data artifacts
# load and look sane, and every dashboard page renders without error.
# A broken push or bad committed data fails CI before it reaches the
# live app.
#   CI:    pytest -q test_app.py
#   local: python test_app.py
# ---------------------------------------------------------------

import pathlib
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

SVGS = ["icon-amber-on-pine.svg", "icon-pine-on-amber.svg", "mark-amber.svg",
        "mark-pine.svg", "mark-chalk.svg", "lockup-primary.svg", "lockup-on-chalk.svg"]

# page_draft is the default page (rendered by nav.run() on import), so it's
# exercised separately from these.
OTHER_PAGES = ("page_overview", "page_weekly", "page_results", "page_trust",
               "page_backtest", "page_projections", "page_why")


def test_svgs_wellformed():
    for name in SVGS:
        ET.parse(ASSETS / name)


def test_data_artifacts_load():
    board = pd.read_parquet(DATA / "draft_board.parquet")
    assert len(board) > 200, f"draft board suspiciously small: {len(board)}"
    assert {"Player", "Tier", "Value", "ECR"} <= set(board.columns)
    model = pd.read_parquet(DATA / "model_table.parquet")
    assert len(model) > 10000, f"model table suspiciously small: {len(model)}"
    bt = pd.read_parquet(DATA / "draft_backtest.parquet")
    assert {"BeatADP", "season"} <= set(bt.columns)


def test_entrypoint_and_default_page_render():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, at.exception


def test_all_pages_render():
    from streamlit.testing.v1 import AppTest
    code = "import app\n" + "".join(f"app.{p}()\n" for p in OTHER_PAGES)
    at = AppTest.from_string(code, default_timeout=300).run()
    assert not at.exception, at.exception


if __name__ == "__main__":
    failed = 0
    for t in (test_svgs_wellformed, test_data_artifacts_load,
              test_entrypoint_and_default_page_render, test_all_pages_render):
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if failed else 0)
