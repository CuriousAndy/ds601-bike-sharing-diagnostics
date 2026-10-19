# Diagnosing and Remedying Linear Model Failures in Daily Bike-Sharing Demand

DS601 Statistical Learning, Group 2 (Topic 1).

## Problem

TODO (1-2 sentences, filled in at the end).

## Data source

TODO (Jiang Liuru drafts this in A3). The notebooks download the UCI Bike Sharing dataset
(id 275, `day.csv`) at run time, so no data file is committed.

## How to run

1. Python 3.12, then `pip install -r requirements.txt`.
2. Run the notebooks in order from the `notebooks/` folder: `01` → `02` → `03` → `04`.
3. The first notebook downloads the data into `data/` (ignored by git).

## Key findings

TODO

## Repo structure

```
src/          shared code used by all notebooks
  data.py         loading, train/test split, unit conversion
  diagnostics.py  residual plots, AV plots, VIF, influence table, summary metrics
  remedies.py     exclusion rules and the remedied model specification
  cv.py           forward-chaining and shuffled cross-validation
notebooks/    01 data & EDA, 02 base model, 03 influence & remedies, 04 CV, test audit & fairness
figures/      PNG copies of the figures used in the slides
report/       final report (single file)
```

See `CONTRIBUTIONS.md` for who did what and `AI_USAGE.md` for the AI declaration.
