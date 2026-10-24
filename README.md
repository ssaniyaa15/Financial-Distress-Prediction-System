# Financial Distress Prediction

Tabular bankruptcy-risk model on the UCI Polish companies dataset: 64 financial ratios, leak-safe training, a bake-off of Logistic Regression / Random Forest / XGBoost, imbalance-aware fitting, a validation-tuned decision threshold, SHAP explanations, and a FastAPI + HTML UI.

XGBoost on the ratios is the production predictor. Other modalities were tried and did not beat this baseline; they are documented at the end.

---

## 1. Data

**UCI Polish Companies Bankruptcy**, 3rd-year forecast file  
Zięba, Tomczak & Tomczak (2016), *Expert Systems with Applications*. License: CC BY 4.0.

| Item | Value |
| --- | --- |
| Source | [UCI 365](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) / Hugging Face mirror `weiqi001/bankruptcy-early-warning-data` |
| Rows | 10,503 firm-years |
| Features | 64 financial ratios (profitability, leverage, liquidity, turnover) |
| Target | `bankrupt` = 1 if the firm filed within **3 years** |
| Class balance | 495 distressed / 10,008 healthy (4.71% positive) |
| Missing cells | 9,888 |

Labels are the official EMIS bankruptcy outcomes. **No labels are fabricated.**

---

## 2. Leak-safe preprocessing and split

Stratified **train / validation / test** IDs are written once to `data/processed/splits.json` (`random_state=42`). All later scripts load those IDs; they do not reshuffle.

On **train only**:

* median imputation
* 1st/99th percentile winsorization
* `RobustScaler`

Validation and test are transformed with those frozen statistics. The preprocessor is saved to `models/preprocessor.joblib`.

---

## 3. Model bake-off (Logistic Regression, Random Forest, XGBoost)

`training/train_structured.py` fits three classifiers on the same processed train fold and selects by **validation ROC-AUC** (threshold-free).

Numbers below are from a full retrain of that script (2026-09-08). Copied from the training log / `models/structured_metrics.json`.

| Model | Val ROC-AUC | Val PR-AUC | Test F1 @ 0.5 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.7471 | 0.1219 | 0.1813 |
| Random Forest | 0.8774 | 0.3968 | 0.3609 |
| **XGBoost (selected)** | **0.9232** | **0.6456** | **0.5436** |

Selected model artifact: `models/structured_best.joblib`.

---

## 4. Imbalance handling

The positive class is ~4.7%. Training uses:

* `class_weight="balanced"` on Logistic Regression and Random Forest
* XGBoost `scale_pos_weight = n_negative / n_positive` on the train fold

Accuracy is a weak headline metric here. ROC-AUC and **PR-AUC** (average precision) are both logged because ROC-AUC can look strong while precision stays modest.

---

## 5. Threshold tuning

A probability cutoff of 0.5 is a poor default under this imbalance. After XGBoost is selected, the cutoff is chosen by **maximizing F1 on the validation set only**, then frozen. Test is scored at both 0.5 and the tuned cutoff. FastAPI loads `models/structured_threshold.json` (falls back to 0.5 if that file is missing).

From the same retrain (`threshold=0.620`, val F1=0.6200):

| Threshold | How chosen | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | Test PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | default | 0.9576 | 0.5521 | 0.5354 | 0.5436 | 0.9201 | 0.6253 |
| 0.62 | max F1 on **validation** only | 0.9629 | 0.6479 | 0.4646 | 0.5412 | 0.9201 | 0.6253 |

ROC-AUC and PR-AUC do not depend on the cutoff. On this test fold, the val-tuned threshold raised precision and lowered recall; test F1 was essentially unchanged (0.5436 → 0.5412). That is expected: the cutoff was not chosen on test.

---

## 6. SHAP explainability

After the selected tree model is fit, training saves:

* `models/structured_shap.joblib` — `shap.TreeExplainer` on XGBoost
* `models/structured_shap_test.npz` — SHAP values for the 2,101-row test set

The `/predict` path returns the **top 5 ratios by |SHAP value|** for that row, with sign (increases vs decreases distress probability). Rule-based ratio checks remain available as `rule_based_factors` (labeled fallback), not as the primary explanation.

---

## 7. FastAPI serving

From the repo root:

```bash
export PYTHONPATH=.
uvicorn services.ml_service.app:app --host 0.0.0.0 --port 8002
```

`POST /predict` on the ML service scores the 64-ratio vector with the saved XGBoost model and returns `structured_risk` (probability), `decision_threshold`, `structured_predicted_label`, and `shap_explanations`.

The fusion/UI process (port 8000) wraps the same structured model in a larger workflow. Prefer the ML service if you only need tabular scores and SHAP.

---

## 8. UI

```bash
export PYTHONPATH=.
uvicorn services.fusion_service.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — a single HTML form: pick a firm from the dataset or enter four simple numbers, optionally attach a PDF/image, and get a risk card.

The **Why (SHAP)** list is the TreeExplainer top-5. The UI also prints a **Confidence** percentage; that value is a heuristic (`0.45 + 0.5 * agreement` of modality scores), **not a calibrated probability**.

---

## 9. Setup and training

Python 3.11+ recommended. Create a virtualenv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys only if you want live LLM calls
```

Tabular train (the path that produced the tables above):

```bash
python scripts/download_and_prepare.py
python training/train_structured.py
```

Optional extra modalities and fusion eval:

```bash
python training/train_lstm.py
python training/train_vision.py
python training/train_fusion.py
python evaluation/evaluate.py
```

LLM weights are **never downloaded**. If no API keys are set, a heuristic analyzer implements the same JSON schema.

---

## 10. Additional experiments (did not outperform the tabular baseline)

Text, charts, and PDFs are **reconstructed from the same 64 ratios** (the source table has no company names to join to filings). They share the real bankruptcy label, but they are not independent signals.

Held-out results from a prior `evaluation/evaluate.py` run (F1 at threshold 0.5). Fusion PR-AUC was not recomputed in the 2026-09-08 tabular retrain.

| Approach | Accuracy | Precision | Recall | F1 @ 0.5 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Only (that eval run) | 0.9557 | 0.5294 | 0.5455 | 0.5373 | 0.9176 |
| Text Only | 0.8220 | 0.0968 | 0.3333 | 0.1500 | 0.6427 |
| Vision Only | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.7023 |
| Early Fusion | 0.7273 | 0.1089 | 0.6667 | 0.1872 | 0.7566 |
| Intermediate Fusion | 0.7939 | 0.1354 | 0.6263 | 0.2226 | 0.7671 |
| Late Fusion | 0.9467 | 0.4511 | 0.6061 | 0.5172 | 0.9177 |
| Multimodal + LLM | 0.9472 | 0.4559 | 0.6263 | 0.5277 | 0.9155 |

* Structured XGBoost is the strongest single modality. Late fusion matched its ROC-AUC while changing the precision/recall tradeoff; it did not beat the tabular model on ranking.
* Vision ROC-AUC is 0.70 but the 0.5 threshold never fires on the rare positive class, so F1 is 0 — a real calibration result, not a missing model.
* The Multimodal + LLM row used the heuristic analyzer because no LLM API keys were set.

LangGraph orchestration, OCR, extra FastAPI workers, and Docker/Cloud Run notes live in `langgraph/`, `services/`, `docker-compose.yml`, and `cloud/README.md`. They are supporting experiments around the tabular model, not a better predictor.

---

## 11. Docker

Train first so `models/` is populated, then:

```bash
cp .env.example .env
docker compose up --build
```

UI: http://localhost:8000

---

## 12. Cloud

See [cloud/README.md](cloud/README.md) for Cloud Run + Cloud Storage + Cloud SQL. No Kubernetes.

---

## 13. Citation

```
Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016).
Ensemble boosted trees with synthetic features generation in application
to bankruptcy prediction. Expert Systems with Applications, 58, 93–101.
```
