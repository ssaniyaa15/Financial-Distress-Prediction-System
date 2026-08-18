# Multimodal Financial Risk Prediction

Predict whether a firm is financially distressed using **structured ratios**, **document text (OCR)**, **chart images**, an **LSTM**, **multiple LLM APIs**, and **three fusion architectures**, orchestrated with **LangGraph** and served as **FastAPI microservices**.

Trained models are the primary predictors. LLMs add contextual commentary; they are not the sole decision engine.

---

## 1. Dataset selected

**UCI Polish Companies Bankruptcy**, 3rd-year forecast file  
Ziemba, Tomczak & Tomczak (2016), *Expert Systems with Applications*. License: CC BY 4.0.

| Item | Value |
| --- | --- |
| Source | [UCI 365](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) / Hugging Face mirror `weiqi001/bankruptcy-early-warning-data` |
| Rows | 10,503 firm-years |
| Features | 64 financial ratios (profitability, leverage, liquidity, turnover) |
| Target | `bankrupt` = 1 if the firm filed within **3 years** |
| Class balance | 495 distressed / 10,008 healthy (4.71% positive) |
| Missing cells | 9,888 (median-imputed, then 1%/99% winsorized) |

Labels are the official EMIS bankruptcy outcomes. **No labels are fabricated.**

### Why this dataset, and how the other modalities are obtained

The table is anonymized: there are **no company names, tickers, or filings**. Polish ratios therefore **cannot be joined** to SEC 10-Ks or news without inventing identities.

Matching strategy used here:

| Modality | How it is obtained | Same real label? |
| --- | --- | --- |
| Structured | Original 64 ratios | Yes |
| Text | MD&A-style narrative filled with **that row’s actual ratios** | Yes |
| PDF / OCR | Statement PDF whose table cells are those ratios | Yes |
| Vision | Bar chart of 10 key ratios from the same row | Yes |
| LLM | API call (or heuristic fallback) over the narrative | Used at inference; batch eval uses the documented heuristic when no keys are set |

Sample SEC files are **not** used as training labels. Upload a real 10-K in the UI to exercise OCR on an external document.

---

## 2. What each component is for

| Component | Question it answers |
| --- | --- |
| Structured ML | What do the financial numbers indicate? |
| OCR | What information is inside the financial document? |
| Computer vision | What does a financial chart/statement image show? |
| LSTM | What does the sequence of financial text indicate? |
| LLMs | What contextual risk language is present? |
| Fusion | How do the modalities combine? |
| LangGraph | How are the stages orchestrated? |
| Microservices | How are the components separated? |
| Cloud Run | How is the system deployed? |

---

## 3. Setup

Python 3.11+ recommended (3.12 is the Docker image). Create a virtualenv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys only if you want live LLM calls
```

LLM weights are **never downloaded**. Optional providers are called over HTTP:

* `OPENAI_API_KEY` — GPT
* `GEMINI_API_KEY` — Gemini
* `GROQ_API_KEY` — Llama (`llama-3.1-8b-instant` by default)
* `OLLAMA_BASE_URL` — local OpenAI-compatible Llama endpoint

If no keys are set, a transparent **heuristic financial analyzer** implements the same JSON schema so training and demos still run.

---

## 4. Download data, train, evaluate

This actually trains models and writes real metrics. It does not invent scores.

```bash
python training/train_all.py
```

Equivalent steps:

```bash
python scripts/download_and_prepare.py
python training/train_structured.py
python training/train_lstm.py
python training/train_vision.py
python training/train_fusion.py
python evaluation/evaluate.py
```

Artifacts:

* `data/raw/bankruptcy_year3.csv` — downloaded public data
* `data/processed/companies.csv` — aligned multimodal table
* `data/documents/*.pdf`, `data/images/*.png` — sample statements/charts
* `models/*.joblib`, `models/*.pt` — trained weights
* `evaluation/results.json`, `evaluation/comparison.md` — held-out metrics
* `plots/*.png` — ROC and comparison figures

---

## 5. Run the API and UI

From the repo root, with `PYTHONPATH` set:

```bash
export PYTHONPATH=.
uvicorn services.fusion_service.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

Optional individual services:

| Service | Port | Endpoints |
| --- | --- | --- |
| Data | 8001 | `GET /companies`, `GET /companies/{id}` |
| ML | 8002 | `POST /predict` |
| OCR | 8003 | `POST /ocr` |
| Vision | 8004 | `POST /vision` |
| LLM | 8005 | `POST /llm-analysis`, `POST /analyze-text` |
| Fusion / UI | 8000 | `POST /predict`, `POST /fusion` |

---

## 6. Docker

Train first so `models/` is populated, then:

```bash
cp .env.example .env
docker compose up --build
```

UI: http://localhost:8000

---

## 7. LangGraph workflow

```
START → load_data → ocr_analysis
                      ├ structured_analysis ┐
                      ├ vision_analysis     ├→ llm_analysis → fusion → risk_prediction → generate_report → END
                      └ text_analysis       ┘
```

Implementation: `langgraph/graph.py`, `nodes.py`, `state.py`.  
The folder is loaded via `importlib` as `risk_workflow` so it does not shadow the installed `langgraph` package.

---

## 8. Fusion

* **Early** — concatenate scaled ratios + LSTM embedding + CNN embedding, logistic classifier
* **Intermediate** — modality encoders + fusion MLP
* **Late** — stack per-modality probabilities (structured, text, vision) with logistic regression
* **Multimodal + LLM** — same stacker with a fourth probability from the LLM/heuristic analyzer

Held-out test results from `evaluation/evaluate.py` (real metrics, not placeholders):

| Approach | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Only | 0.9557 | 0.5294 | 0.5455 | 0.5373 | 0.9176 |
| Text Only | 0.8220 | 0.0968 | 0.3333 | 0.1500 | 0.6427 |
| Vision Only | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.7023 |
| Early Fusion | 0.7273 | 0.1089 | 0.6667 | 0.1872 | 0.7566 |
| Intermediate Fusion | 0.7939 | 0.1354 | 0.6263 | 0.2226 | 0.7671 |
| Late Fusion | 0.9467 | 0.4511 | 0.6061 | 0.5172 | 0.9177 |
| Multimodal + LLM | 0.9472 | 0.4559 | 0.6263 | 0.5277 | 0.9155 |

Structured XGBoost is the strongest single modality (ROC-AUC 0.92). Late fusion matches it while raising recall. Early/intermediate fusion trade precision for recall. Vision ROC-AUC is 0.70 but the 0.5 threshold never fires on the rare positive class, so F1 is 0 — that is a real calibration result, not a missing model. The Multimodal + LLM row used the heuristic analyzer because no LLM API keys were set; live GPT/Gemini/Llama calls use the same schema when keys are present.

The fusion service report looks like:

```
Company: Example Corp
Structured Risk: 0.71
Text Risk: 0.64
Vision Risk: 0.58
LLM Risk: 0.76
Final Risk Score: 0.68
Risk Level: HIGH
Confidence: 82%
```

Final score prefers late fusion, then intermediate, then early, then structured-only.

---

## 9. Cloud

See [cloud/README.md](cloud/README.md) for Cloud Run + Cloud Storage + Cloud SQL. No Kubernetes.

---

## 10. Citation

```
Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016).
Ensemble boosted trees with synthetic features generation in application
to bankruptcy prediction. Expert Systems with Applications, 58, 93–101.
```
