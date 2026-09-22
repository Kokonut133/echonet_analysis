# Results figures

- **hero_information_ladder.png** — Headline figure: test-set AUROC per target across the five information tiers (demographics -> ECG metadata -> waveform features -> combined -> CNN on raw ECG), with 95% CIs.
- **roc_pr_headline_targets.png** — ROC (top) and precision-recall (bottom) curves for the four headline targets, all tiers overlaid; dashed line in the PR panels marks the prevalence baseline.
- **operating_points.png** — For the CNN tier on the four headline targets: specificity, PPV and NPV at fixed sensitivity thresholds of 80/90/95%, i.e. the trade-off from choosing a more sensitive screening cutoff.
- **subgroup_auroc.png** — AUROC with 95% CI by sex, age band, race/ethnicity and care setting for the combined and CNN tiers, on the SHD and LVEF <=45% targets; underlying numbers in reports/subgroup_results.csv.
- **ecg_example_positive_vs_negative.png** — One true-positive and one true-negative 12-lead ECG (LVEF <=45% target) in the standard clinical 3x4 layout plus a lead-II rhythm strip, with the CNN's predicted risk for each.
