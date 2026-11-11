"""Classify a new review and explain the decision.

Usage:
    python src/predict.py "O produto chegou quebrado e ninguém responde"
"""

import sys
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np

from data import ROOT, normalize_text

MODEL_PATH = ROOT / "models" / "sentiment.joblib"


@dataclass
class Prediction:
    text: str
    label: str  # "negativa" | "positiva"
    probability_negative: float
    threshold: float
    contributions: list[tuple[str, float]]  # (term, weight); > 0 pushes to negative

    @property
    def priority(self) -> str:
        if self.probability_negative >= 0.85:
            return "alta"
        if self.probability_negative >= self.threshold:
            return "média"
        return "baixa"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} não existe. Rode antes: python src/train.py")
    return joblib.load(MODEL_PATH)


def predict(text: str, top_k: int = 6) -> Prediction:
    bundle = _load()
    model, threshold = bundle["model"], bundle["threshold"]
    clean = normalize_text(text)
    probability = float(model.predict_proba([clean])[0, 1])

    # The model is linear: logit = intercept + sum(tfidf_j * coef_j). Each term's
    # tfidf_j * coef_j is exactly how much it moved this decision, so the
    # explanation is the model itself, not an approximation of it.
    contributions: list[tuple[str, float]] = []
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    if hasattr(vectorizer, "vocabulary_"):
        row = vectorizer.transform([clean])
        names = vectorizer.get_feature_names_out()
        weights = row.toarray()[0] * clf.coef_[0]
        idx = np.flatnonzero(weights)
        idx = idx[np.argsort(-np.abs(weights[idx]))][:top_k]
        contributions = [(str(names[i]), float(weights[i])) for i in idx]

    return Prediction(
        text=text,
        label="negativa" if probability >= threshold else "positiva",
        probability_negative=probability,
        threshold=threshold,
        contributions=contributions,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = predict(" ".join(sys.argv[1:]))
    print(f"Avaliação: {result.label.upper()}  (probabilidade de ser negativa: {result.probability_negative:.0%})")
    print(f"Prioridade para o atendimento: {result.priority}")
    if result.contributions:
        print("Termos que mais pesaram:")
        for term, weight in result.contributions:
            side = "→ negativa" if weight > 0 else "→ positiva"
            print(f"  {term:<25} {weight:+.2f} {side}")
    else:
        print("Nenhum termo conhecido pelo modelo: a decisão veio só do viés (classe mais comum).")


if __name__ == "__main__":
    main()
