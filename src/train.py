"""Compare models with cross-validation, pick one, evaluate it on the test set.

Steps:
1. 5-fold stratified cross-validation on the TRAIN period for every candidate.
2. The best model by PR-AUC (average precision) of the negative class wins.
3. Its decision threshold is chosen on the out-of-fold predictions of TRAIN
   (never on test), maximising F1 of the negative class.
4. The chosen model is refit on all of TRAIN and evaluated ONCE on TEST.
5. Model, metrics, figures and error examples are written to models/ and reports/.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from data import NEGATIVE, ROOT, labelled, load_reviews, temporal_split
from models import candidates, scores

MODEL_PATH = ROOT / "models" / "sentiment.joblib"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
SEED = 42


def metrics_at(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    return {
        "precision_neg": precision_score(y_true, pred, zero_division=0),
        "recall_neg": recall_score(y_true, pred, zero_division=0),
        "f1_neg": f1_score(y_true, pred, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "accuracy": float((pred == y_true).mean()),
    }


def ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if np.unique(score).size == 1:
        return {"pr_auc": float(y_true.mean()), "roc_auc": 0.5}
    return {
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
    }


def out_of_fold(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: N803
    oof = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, val_idx in folds.split(X, y):
        fitted = clone(model).fit(X[train_idx], y[train_idx])
        oof[val_idx] = scores(fitted, X[val_idx])
    return oof


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def compare(X: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:  # noqa: N803
    rows, oofs = [], {}
    for name, model in candidates().items():
        start = time.perf_counter()
        oof = out_of_fold(model, X, y)
        default_threshold = 0.5 if hasattr(model, "predict_proba") else 0.0
        row = {"modelo": name, **ranking_metrics(y, oof), **metrics_at(y, oof, default_threshold)}
        row["segundos"] = time.perf_counter() - start
        rows.append(row)
        oofs[name] = oof
        print(f"  {name:<45} PR-AUC {row['pr_auc']:.3f}  F1(neg) {row['f1_neg']:.3f}  ({row['segundos']:.0f}s)")
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return table, oofs


def save_figures(y_test: np.ndarray, test_score: np.ndarray, threshold: float, model) -> None:
    pred = (test_score >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=["positiva", "negativa"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matriz de confusão — teste")
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_matriz_confusao.png", dpi=130)
    plt.close(fig)

    precision, recall, thresholds = precision_recall_curve(y_test, test_score)
    idx = np.searchsorted(thresholds, threshold)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#2f6fb0")
    ax.scatter([recall[idx]], [precision[idx]], color="#c0392b", zorder=3, label=f"limiar {threshold:.2f}")
    ax.axhline(y_test.mean(), color="#999", linestyle="--", label="modelo aleatório")
    ax.set_xlabel("recall (negativas encontradas)")
    ax.set_ylabel("precisão (alertas corretos)")
    ax.set_title("Curva precisão × recall — teste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_precisao_recall.png", dpi=130)
    plt.close(fig)

    words = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    if hasattr(words, "transformer_list"):
        words = dict(words.transformer_list)["words"]
        coefs = clf.coef_[0][: len(words.vocabulary_)]
    else:
        coefs = clf.coef_[0]
    vocab = np.array(words.get_feature_names_out())
    order = np.argsort(coefs)
    top = pd.concat(
        [
            pd.Series(coefs[order[:12]], index=vocab[order[:12]]),
            pd.Series(coefs[order[-12:]], index=vocab[order[-12:]]),
        ]
    )
    fig, ax = plt.subplots(figsize=(6.5, 6))
    top.plot.barh(ax=ax, color=["#2f8f5b" if v < 0 else "#c0392b" for v in top])
    ax.set_title("Termos com maior peso (palavras)\nvermelho → negativa, verde → positiva")
    ax.set_xlabel("coeficiente")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_termos.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    reviews = load_reviews()
    train, test = temporal_split(labelled(reviews))
    X_train, y_train = train["clean"].to_numpy(), train["label"].to_numpy()  # noqa: N806
    X_test, y_test = test["clean"].to_numpy(), test["label"].to_numpy()  # noqa: N806

    print(f"Validação cruzada (5 folds) em {len(train)} avaliações de treino:")
    table, oofs = compare(X_train, y_train)
    best_name = table.loc[0, "modelo"]
    threshold = best_threshold(y_train, oofs[best_name])
    print(f"Escolhido: {best_name} | limiar (OOF, máx F1 negativa): {threshold:.3f}")

    model = clone(candidates()[best_name]).fit(X_train, y_train)
    test_score = scores(model, X_test)
    test_metrics = {
        **ranking_metrics(y_test, test_score),
        "at_0.5": metrics_at(y_test, test_score, 0.5),
        "at_threshold": metrics_at(y_test, test_score, threshold),
    }
    print(json.dumps(test_metrics, indent=2))

    # Baselines on the same test set, for the comparison table in the README.
    baseline_rows = []
    for name in ["Dummy (classe majoritária)", "Regra de palavras-chave"]:
        base = clone(candidates()[name]).fit(X_train, y_train)
        s = scores(base, X_test)
        baseline_rows.append({"modelo": name, **ranking_metrics(y_test, s), **metrics_at(y_test, s, 0.5)})
    baseline_rows.append(
        {"modelo": f"{best_name} (limiar {threshold:.2f})", **ranking_metrics(y_test, test_score),
         **metrics_at(y_test, test_score, threshold)}
    )

    save_figures(y_test, test_score, threshold, model)

    test = test.assign(score_neg=test_score, pred=(test_score >= threshold).astype(int))
    false_neg = test[(test["label"] == NEGATIVE) & (test["pred"] == 0)].sort_values("score_neg")
    false_pos = test[(test["label"] == 0) & (test["pred"] == NEGATIVE)].sort_values("score_neg", ascending=False)

    # Errors by comment length: short comments carry little signal.
    test["palavras"] = pd.cut(
        test["clean"].str.split().str.len(), bins=[0, 3, 10, 25, 1000], labels=["1-3", "4-10", "11-25", "26+"]
    )
    by_length = (
        test.groupby("palavras", observed=True)
        .apply(lambda g: pd.Series({"n": len(g), "f1_neg": f1_score(g["label"], g["pred"], zero_division=0),
                                    "acerto": (g["label"] == g["pred"]).mean()}), include_groups=False)
        .reset_index()
    )

    # Score-3 reviews were left out of training; how does the model see them?
    neutral = reviews[(reviews["score"] == 3) & (reviews["date"] >= pd.Timestamp("2018-06-01"))]
    neutral_share = float((scores(model, neutral["clean"].to_numpy()) >= threshold).mean())

    joblib.dump({"model": model, "threshold": threshold, "name": best_name}, MODEL_PATH)

    results = {
        "train_size": len(train),
        "test_size": len(test),
        "train_negative_share": float(y_train.mean()),
        "test_negative_share": float(y_test.mean()),
        "chosen_model": best_name,
        "threshold": threshold,
        "cv": table.drop(columns="segundos").round(4).to_dict(orient="records"),
        "test": test_metrics,
        "test_comparison": pd.DataFrame(baseline_rows).round(4).to_dict(orient="records"),
        "by_length": by_length.round(4).to_dict(orient="records"),
        "neutral_flagged_negative": neutral_share,
        "neutral_size": len(neutral),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))

    cols = ["text", "score", "score_neg"]
    with open(REPORT_DIR / "erros.md", "w") as fh:
        fh.write("# Erros do modelo no conjunto de teste\n\n")
        fh.write(f"Limiar: {threshold:.2f}. `p(neg)` = probabilidade de a avaliação ser negativa.\n\n")
        fh.write(f"## Falsos negativos — reclamações que passariam despercebidas ({len(false_neg)})\n\n")
        fh.write("Os 15 com menor probabilidade de serem negativos:\n\n| nota | p(neg) | comentário |\n|---|---|---|\n")
        for _, r in false_neg[cols].head(15).iterrows():
            fh.write(f"| {r.score} | {r.score_neg:.2f} | {r.text.replace('|', '/')} |\n")
        fh.write(f"\n## Falsos positivos — elogios tratados como reclamação ({len(false_pos)})\n\n")
        fh.write("Os 15 com maior probabilidade de serem negativos:\n\n| nota | p(neg) | comentário |\n|---|---|---|\n")
        for _, r in false_pos[cols].head(15).iterrows():
            fh.write(f"| {r.score} | {r.score_neg:.2f} | {r.text.replace('|', '/')} |\n")

    print("\nComparação no teste:")
    print(pd.DataFrame(baseline_rows).round(3).to_string(index=False))
    print("\nPor tamanho do comentário:")
    print(by_length.round(3).to_string(index=False))
    print(f"\nNotas 3 no período de teste ({len(neutral)}): {neutral_share:.1%} marcadas como negativas")
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
