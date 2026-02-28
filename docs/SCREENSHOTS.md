# Screenshots for CrisisLens

Use this checklist before pushing to GitHub. Capture the images and save them in `docs/screenshots/` with the exact filenames below so the README links work.

## Required screenshots

| Filename | What to capture | How |
|----------|------------------|-----|
| `dashboard-analyze.png` | **Analyze tab**: sidebar with a sample selected, main area with an **Input Message** (e.g. Rescue English) and **Analysis Result** showing Crisis Related, CRITICAL, RESCUE_REQUEST, language, relevance %, locations. | Run `streamlit run src/dashboard/app.py`, select "🆘 Rescue (English)", click **Analyze**, then capture the full tab. |
| `dashboard-priority-feed.png` | **Priority Feed tab**: at least one crisis entry (e.g. #1 — CRITICAL \| RESCUE_REQUEST \| en) and the "Export as CSV" button. | After analyzing a sample, open the "🚨 Priority Feed" tab and screenshot. |
| `dashboard-map.png` | **Crisis Map tab**: map with at least one marker (run an analysis that has a geocoded location first, e.g. Rescue English → Hatay). | Open "🗺️ Crisis Map" tab; ensure one analyzed message has a location so a marker appears. |
| `dashboard-analytics.png` | **Analytics tab**: metrics row (Total, Relevant, Critical, Duplicates) and at least one chart (e.g. Urgency distribution). | Open "📊 Analytics" tab after analyzing 1–2 samples. |
| `api-docs.png` | **Swagger UI**: FastAPI `/docs` page with endpoints visible. | Run `uvicorn src.api.main:app --reload`, open http://localhost:8000/docs, capture the page. |

## Optional

- `dashboard-sidebar.png` — Sidebar with Session Stats, Evaluation (HumAID benchmark), Limitations.
- `api-analyze-response.png` — Example JSON response from **POST /api/v1/analyze** in Swagger.

## Tips

- Use a consistent browser window size (e.g. 1200×800) so screenshots look uniform.
- Hide or crop personal bookmarks/tabs; keep the app and URL bar if helpful.
- PNG is preferred; compress with [TinyPNG](https://tinypng.com/) if files are large.
