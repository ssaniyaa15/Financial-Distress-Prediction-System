"""Project paths and runtime configuration."""

from __future__ import annotations

import os

# Avoid OpenMP conflicts between PyTorch and XGBoost on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
TEXT_DIR = DATA_DIR / "text"
SEC_DIR = DATA_DIR / "sec_samples"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
EVAL_DIR = PROJECT_ROOT / "evaluation"

RAW_CSV = RAW_DIR / "bankruptcy_year3.csv"
COMPANIES_CSV = PROCESSED_DIR / "companies.csv"
SPLIT_JSON = PROCESSED_DIR / "splits.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
STRUCTURED_MODEL_PATH = MODELS_DIR / "structured_best.joblib"
STRUCTURED_METRICS_PATH = MODELS_DIR / "structured_metrics.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_risk.pt"
LSTM_VOCAB_PATH = MODELS_DIR / "lstm_vocab.json"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_baseline.joblib"
VISION_MODEL_PATH = MODELS_DIR / "vision_cnn.pt"
EARLY_FUSION_PATH = MODELS_DIR / "early_fusion.joblib"
INTERMEDIATE_FUSION_PATH = MODELS_DIR / "intermediate_fusion.pt"
LATE_FUSION_PATH = MODELS_DIR / "late_fusion.joblib"
EVAL_RESULTS_PATH = EVAL_DIR / "results.json"
DB_PATH = PROCESSED_DIR / "predictions.db"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:8001")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8002")
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:8003")
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8004")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8005")
FUSION_SERVICE_URL = os.getenv("FUSION_SERVICE_URL", "http://localhost:8000")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")


def ensure_directories() -> None:
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        DOCUMENTS_DIR,
        IMAGES_DIR,
        TEXT_DIR,
        SEC_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        EVAL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
