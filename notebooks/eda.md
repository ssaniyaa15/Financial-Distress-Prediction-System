# Exploration notebook

The executable pipeline lives in `scripts/` and `training/`. This notebook records
the dataset choice and how to inspect processed artifacts after training.

```python
import pandas as pd
from pathlib import Path

root = Path("..") if Path("..").joinpath("data").exists() else Path(".")
raw = pd.read_csv(root / "data/raw/bankruptcy_year3.csv")
print(raw.shape)
print(raw["bankrupt"].value_counts())
print(raw.isna().sum().sum())
```

Processed companies (after `scripts/download_and_prepare.py`):

```python
companies = pd.read_csv(root / "data/processed/companies.csv")
print(companies.split.value_counts())
print(companies.narrative.iloc[0][:500])
```

Evaluation table (after `evaluation/evaluate.py`):

```python
import json
print(Path(root / "evaluation/comparison.md").read_text())
print(json.loads((root / "evaluation/results.json").read_text()))
```
