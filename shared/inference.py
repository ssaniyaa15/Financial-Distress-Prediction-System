"""Load trained models and produce per-modality probabilities."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from shared.charts import chart_to_chw, render_ratio_chart
from shared.config import (
    EARLY_FUSION_PATH,
    DECISION_THRESHOLD_PATH,
    INTERMEDIATE_FUSION_PATH,
    LATE_FUSION_PATH,
    LSTM_MODEL_PATH,
    LSTM_VOCAB_PATH,
    PREPROCESSOR_PATH,
    SHAP_EXPLAINER_PATH,
    STRUCTURED_MODEL_PATH,
    TFIDF_MODEL_PATH,
    VISION_MODEL_PATH,
)
from shared.explain import TREE_MODEL_NAMES, build_tree_explainer, explain_structured_row
from shared.features import FEATURE_COLUMNS
from shared.llm import available_analyzers, ensemble_llm_outputs
from shared.metrics import risk_level_from_score
from shared.nn_models import IntermediateFusionNet, RiskCNN, RiskLSTM
from shared.preprocessing import StructuredPreprocessor
from shared.schemas import LLMRiskOutput
from shared.text_data import load_vocab, texts_to_sequences
from shared.threshold import load_decision_threshold


def _torch_device() -> torch.device:
    import os

    if os.getenv("FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelBundle:
    def __init__(self) -> None:
        self.device = _torch_device()
        self.preprocessor: StructuredPreprocessor = StructuredPreprocessor.load(PREPROCESSOR_PATH)
        packed = joblib.load(STRUCTURED_MODEL_PATH)
        self.structured_model = packed["model"]
        self.structured_name = packed.get("name", "structured")
        self.decision_threshold = load_decision_threshold()
        if not DECISION_THRESHOLD_PATH.exists():
            packed_threshold = packed.get("threshold")
            if packed_threshold is not None:
                try:
                    self.decision_threshold = float(packed_threshold)
                except (TypeError, ValueError):
                    pass
        self.shap_explainer = None
        if SHAP_EXPLAINER_PATH.exists():
            try:
                shap_pack = joblib.load(SHAP_EXPLAINER_PATH)
                self.shap_explainer = shap_pack.get("explainer")
            except Exception:
                self.shap_explainer = None
        if self.shap_explainer is None and self.structured_name in TREE_MODEL_NAMES:
            try:
                self.shap_explainer = build_tree_explainer(self.structured_model)
            except Exception:
                self.shap_explainer = None
        self.tfidf = joblib.load(TFIDF_MODEL_PATH) if TFIDF_MODEL_PATH.exists() else None
        self.vocab = load_vocab(LSTM_VOCAB_PATH)
        lstm_pack = torch.load(LSTM_MODEL_PATH, map_location=self.device, weights_only=False)
        self.lstm_max_len = int(lstm_pack.get("max_len", 160))
        self.lstm = RiskLSTM(vocab_size=int(lstm_pack["vocab_size"])).to(self.device)
        self.lstm.load_state_dict(lstm_pack["state_dict"])
        self.lstm.eval()
        vision_pack = torch.load(VISION_MODEL_PATH, map_location=self.device, weights_only=False)
        self.cnn = RiskCNN().to(self.device)
        self.cnn.load_state_dict(vision_pack["state_dict"])
        self.cnn.eval()
        self.early = joblib.load(EARLY_FUSION_PATH) if EARLY_FUSION_PATH.exists() else None
        self.late = joblib.load(LATE_FUSION_PATH) if LATE_FUSION_PATH.exists() else None
        self.intermediate = None
        if INTERMEDIATE_FUSION_PATH.exists():
            pack = torch.load(INTERMEDIATE_FUSION_PATH, map_location=self.device, weights_only=False)
            net = IntermediateFusionNet(
                struct_dim=int(pack.get("struct_dim", 64)),
                text_dim=int(pack.get("text_dim", 64)),
                vision_dim=int(pack.get("vision_dim", 64)),
            ).to(self.device)
            net.load_state_dict(pack["state_dict"])
            net.eval()
            self.intermediate = net
        self.analyzers = available_analyzers()

    def _feature_frame(self, features: Mapping[str, Any]) -> pd.DataFrame:
        row = {col: features.get(col, np.nan) for col in FEATURE_COLUMNS}
        return pd.DataFrame([row])

    def structured_proba(self, features: Mapping[str, Any]) -> float:
        X = self.preprocessor.transform(self._feature_frame(features))
        return float(self.structured_model.predict_proba(X)[0, 1])

    def structured_vector(self, features: Mapping[str, Any]) -> np.ndarray:
        return self.preprocessor.transform(self._feature_frame(features))[0]

    def shap_explanations(self, features: Mapping[str, Any], k: int = 5) -> list[dict[str, Any]]:
        if self.shap_explainer is None:
            return []
        X = self.preprocessor.transform(self._feature_frame(features))
        return explain_structured_row(self.shap_explainer, X, FEATURE_COLUMNS, k=k)

    @torch.no_grad()
    def text_proba_and_embedding(self, narrative: str) -> tuple[float, np.ndarray]:
        sequences = texts_to_sequences([narrative], self.vocab, self.lstm_max_len)
        tensor = torch.from_numpy(sequences).to(self.device)
        logits = self.lstm(tensor)
        proba = float(torch.softmax(logits, dim=1)[0, 1].cpu())
        embedding = self.lstm.encode(tensor)[0].detach().cpu().numpy()
        return proba, embedding

    def tfidf_proba(self, narrative: str) -> float | None:
        if self.tfidf is None:
            return None
        return float(self.tfidf.predict_proba([narrative])[0, 1])

    @torch.no_grad()
    def vision_proba_and_embedding(
        self,
        features: Mapping[str, Any],
        image: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        if image is None:
            image = render_ratio_chart(features)
        if image.ndim == 3 and image.shape[0] != 3:
            chw = chart_to_chw(image)
        else:
            chw = image.astype(np.float32)
            if chw.max() > 1.5:
                chw = chw / 255.0
        tensor = torch.from_numpy(chw).unsqueeze(0).float().to(self.device)
        logits = self.cnn(tensor)
        proba = float(torch.softmax(logits, dim=1)[0, 1].cpu())
        embedding = self.cnn.encode(tensor)[0].detach().cpu().numpy()
        return proba, embedding

    def llm_outputs(self, narrative: str, summary: dict[str, Any] | None) -> list[LLMRiskOutput]:
        outputs = []
        for analyzer in self.analyzers:
            try:
                outputs.append(analyzer.analyze(narrative, summary))
            except Exception as exc:  # noqa: BLE001
                outputs.append(
                    LLMRiskOutput(
                        risk_score=0.5,
                        risk_level="medium",
                        confidence=0.2,
                        reasoning=f"{analyzer.name} failed: {exc}",
                        model_name=analyzer.name,
                    )
                )
        return outputs

    def early_fusion_proba(self, fused_vector: np.ndarray) -> float | None:
        if self.early is None:
            return None
        return float(self.early.predict_proba(fused_vector.reshape(1, -1))[0, 1])

    @torch.no_grad()
    def intermediate_fusion_proba(
        self,
        structured_vec: np.ndarray,
        text_emb: np.ndarray,
        vision_emb: np.ndarray,
    ) -> float | None:
        if self.intermediate is None:
            return None
        logits = self.intermediate(
            torch.from_numpy(structured_vec.astype(np.float32)).unsqueeze(0).to(self.device),
            torch.from_numpy(text_emb.astype(np.float32)).unsqueeze(0).to(self.device),
            torch.from_numpy(vision_emb.astype(np.float32)).unsqueeze(0).to(self.device),
        )
        return float(torch.softmax(logits, dim=1)[0, 1].cpu())

    def late_fusion_proba(self, parts: np.ndarray) -> float | None:
        if self.late is None:
            return None
        model = self.late["model"]
        return float(model.predict_proba(parts.reshape(1, -1))[0, 1])


def image_from_path(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB").resize((128, 128)))
