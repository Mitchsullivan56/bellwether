# ---------------------------------------------------------------
# warm_cache.py  —  pre-build the model cache so the deployed app
# never trains on load. Run in the GitHub Action after the data is
# rebuilt; the resulting compute_cache.joblib is committed and read
# (read-only) by the app on Streamlit Cloud.
# USAGE: python3 warm_cache.py
# ---------------------------------------------------------------

import joblib

from ff_cache import data_version, compute_core, CACHE

out = compute_core()
joblib.dump({"version": data_version(), "data": out}, CACHE)
print(f"Warm cache written: {CACHE}  (version {data_version()[:12]}…)")
