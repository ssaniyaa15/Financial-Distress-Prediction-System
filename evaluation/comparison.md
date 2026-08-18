# Held-out test comparison

Metrics come from the official 20% test split of the UCI Polish year-3 bankruptcy file.
The Multimodal + LLM row uses the heuristic financial analyzer in this run because
`OPENAI_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` were unset. No scores were invented.

| Approach | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Only | 0.9557 | 0.5294 | 0.5455 | 0.5373 | 0.9176 |
| Text Only | 0.8220 | 0.0968 | 0.3333 | 0.1500 | 0.6427 |
| Vision Only | 0.9529 | 0.0000 | 0.0000 | 0.0000 | 0.7023 |
| Early Fusion | 0.7273 | 0.1089 | 0.6667 | 0.1872 | 0.7566 |
| Intermediate Fusion | 0.7939 | 0.1354 | 0.6263 | 0.2226 | 0.7671 |
| Late Fusion | 0.9467 | 0.4511 | 0.6061 | 0.5172 | 0.9177 |
| Multimodal + LLM | 0.9472 | 0.4559 | 0.6263 | 0.5277 | 0.9155 |
