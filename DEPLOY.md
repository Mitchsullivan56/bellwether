# Taking Bellwether live (offseason stage)

Goal: a public dashboard + landing page, signups flowing to Buttondown, and rankings
that refresh themselves daily. Your code can stay in a **private** repo — Streamlit
Cloud still gives you a **public** app URL. Budget ~20–30 minutes.

---

## 1. Buttondown (the email list)
1. Sign up at <https://buttondown.com>.
2. Settings → **Programming** → copy your **API key**.

## 2. GitHub repo
1. Create a repo (private is fine), e.g. `bellwether`.
2. From the project folder, first pre-build the model cache so the app loads instantly
   on Cloud, then push everything (data parquets + cache are committed on purpose):
   ```bash
   python warm_cache.py
   git init
   git add .
   git commit -m "Bellwether — initial"
   git branch -M main
   git remote add origin https://github.com/<you>/bellwether.git
   git push -u origin main
   ```
   `.gitignore` already keeps secrets and the local signups CSV out.

## 3. Streamlit Community Cloud (the app)
1. Go to <https://share.streamlit.io> → **New app** → pick your repo, branch `main`,
   main file **`app.py`**.
2. **Advanced → Secrets** — paste:
   ```toml
   BUTTONDOWN_API_KEY = "paste-your-key"
   DEPLOYED = "1"
   ```
3. Deploy. You'll get a public URL like `https://bellwether.streamlit.app`.
   - `DEPLOYED` hides the in-app Refresh button (the Action handles updates).
   - Signups now go straight to Buttondown.

## 4. The scheduled updater (already in the repo)
`.github/workflows/update.yml` runs daily (11:00 UTC): it rebuilds the data + draft
board, pre-warms the cache, archives a dated snapshot to `data/archive/`, and commits.
Streamlit Cloud redeploys on each push, so the live app stays fresh and fast.
- Enable it: GitHub → **Actions** tab → enable workflows.
- Test it now: **Actions → Update rankings → Run workflow**.

## 5. Static landing page (instant load, in front of the app)
The page lives in `docs/`. Two free options:
- **GitHub Pages** (needs a *public* repo): Settings → Pages → Source = `main` / `/docs`.
- **Cloudflare Pages** (works with a *private* repo): connect the repo, build output dir `docs`.

Then edit **`docs/index.html`**: replace `href="#"` on the "Launch the dashboard" button
with your Streamlit URL, and drop a `docs/screenshot.png` of the dashboard.

## 6. Final wiring (optional but nice)
In `app.py`, set:
```python
REPO_URL = "https://github.com/<you>/bellwether"   # only if the repo is public
SEASON_PASS_URL = "https://..."                     # when the season pass is live
```

---

### How updates flow once live
`GitHub Action (daily)` → rebuilds + commits data → `Streamlit Cloud redeploys` →
public app shows fresh, pre-warmed rankings. The dated snapshots in `data/archive/`
are your accumulating public track record.

### Local development still works
No secrets locally → signups fall back to `data/signups.csv` and the Refresh button
reappears. `python -m streamlit run app.py` (or `run.cmd`) as before.
