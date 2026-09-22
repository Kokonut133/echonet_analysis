# docs/

This folder holds the published GitHub Pages site: a single self-contained `index.html` with an interactive AUROC/AUPRC comparison across modeling tiers for the EchoNext structural-heart-disease models.

Regenerate it with `.venv/bin/python scripts/10_site/build_site.py` from the project root (it reads from `reports/` and rewrites `docs/index.html` in place).

To enable Pages: repo **Settings -> Pages -> Deploy from branch -> `master` -> `/docs`**.
