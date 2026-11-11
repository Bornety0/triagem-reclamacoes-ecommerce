"""Candidate models compared during training."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# Hand-written list of complaint words, already accent-free like the
# normalised text. It is the "what a person would do without ML" baseline.
NEGATIVE_TERMS = (
    "nao recebi", "nao chegou", "ainda nao", "atras", "defeito", "quebrad",
    "errad", "pessim", "ruim", "horrivel", "devolv", "reembols", "cancel",
    "nao recomendo", "faltando", "falta", "enganos", "decepcion", "demor",
)


class KeywordBaseline(BaseEstimator, ClassifierMixin):
    """Flags a review as negative when it contains any complaint term."""

    def fit(self, X, y):  # noqa: N803 - sklearn naming
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # noqa: N803
        hits = np.array([any(term in text for term in NEGATIVE_TERMS) for text in X], dtype=float)
        return np.column_stack([1 - hits, hits])

    def predict(self, X):  # noqa: N803
        return self.predict_proba(X)[:, 1].astype(int)


def _words() -> TfidfVectorizer:
    # Stop words are NOT removed: the Portuguese list contains "não", and
    # "não recebi" vs "recebi" is the whole difference between the classes.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _chars() -> TfidfVectorizer:
    # Character n-grams inside word boundaries absorb typos and inflections
    # ("pessimo", "pesimo", "pessima") that a word vocabulary splits apart.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)


def candidates() -> dict[str, BaseEstimator]:
    return {
        "Dummy (classe majoritária)": DummyClassifier(strategy="most_frequent"),
        "Regra de palavras-chave": KeywordBaseline(),
        "TF-IDF palavras + Naive Bayes": Pipeline(
            [("tfidf", _words()), ("clf", ComplementNB(alpha=0.3))]
        ),
        "TF-IDF palavras + Reg. Logística": Pipeline(
            [("tfidf", _words()), ("clf", LogisticRegression(C=4, class_weight="balanced", max_iter=2000))]
        ),
        "TF-IDF palavras+caracteres + Reg. Logística": Pipeline(
            [
                ("tfidf", FeatureUnion([("words", _words()), ("chars", _chars())])),
                ("clf", LogisticRegression(C=4, class_weight="balanced", max_iter=2000)),
            ]
        ),
        "TF-IDF palavras+caracteres + SVM linear": Pipeline(
            [
                ("tfidf", FeatureUnion([("words", _words()), ("chars", _chars())])),
                ("clf", LinearSVC(C=0.3, class_weight="balanced")),
            ]
        ),
    }


def scores(model: BaseEstimator, X) -> np.ndarray:  # noqa: N803
    """A continuous score for the negative class (probability when available)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
