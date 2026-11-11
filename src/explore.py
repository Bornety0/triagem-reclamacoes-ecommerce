"""Exploratory analysis: prints the key numbers and saves the figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from data import RAW_CSV, ROOT, labelled, load_reviews, temporal_split

FIG_DIR = ROOT / "reports" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_CSV)
    df = load_reviews()
    lab = labelled(df)
    train, test = temporal_split(lab)

    print(f"Avaliações no CSV: {len(raw)}")
    print(f"Sem comentário: {raw['review_comment_message'].isna().mean():.1%}")
    print(f"Com comentário utilizável: {len(df)}")
    print("Notas (com comentário):")
    print(df["score"].value_counts().sort_index().to_string())
    print(f"Rotuladas (sem nota 3): {len(lab)}  negativas: {lab['label'].mean():.1%}")
    print(f"Treino: {len(train)} ({train['date'].min():%Y-%m-%d} a {train['date'].max():%Y-%m-%d})"
          f"  negativas: {train['label'].mean():.1%}")
    print(f"Teste:  {len(test)} ({test['date'].min():%Y-%m-%d} a {test['date'].max():%Y-%m-%d})"
          f"  negativas: {test['label'].mean():.1%}")
    words = df["clean"].str.split().str.len()
    print(f"Palavras por comentário: mediana {words.median():.0f}, p90 {words.quantile(0.9):.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    with_text = raw.dropna(subset=["review_comment_message"])
    counts = pd.DataFrame(
        {
            "todas": raw["review_score"].value_counts().sort_index(),
            "com comentário": with_text["review_score"].value_counts().sort_index(),
        }
    )
    counts.plot.bar(ax=axes[0], color=["#9aa5b1", "#2f6fb0"], rot=0)
    axes[0].set_title("Avaliações por nota")
    axes[0].set_xlabel("nota (estrelas)")
    axes[0].set_ylabel("avaliações")

    share = with_text.groupby("review_score").size() / raw.groupby("review_score").size()
    share.plot.bar(ax=axes[1], color="#2f6fb0", rot=0)
    axes[1].set_title("Fração que deixou comentário")
    axes[1].set_xlabel("nota (estrelas)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_notas.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name, color in [(0, "positiva (4-5)", "#2f8f5b"), (1, "negativa (1-2)", "#c0392b")]:
        lab.loc[lab["label"] == label, "clean"].str.split().str.len().clip(upper=60).plot.hist(
            bins=30, alpha=0.6, ax=ax, label=name, color=color, density=True
        )
    ax.set_title("Tamanho do comentário (palavras, cortado em 60)")
    ax.set_xlabel("palavras")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_tamanho.png", dpi=130)
    plt.close(fig)

    monthly = lab.set_index("date").resample("MS")["label"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly["mean"], marker="o", color="#c0392b")
    ax.axvline(pd.Timestamp("2018-06-01"), color="#555", linestyle="--", label="início do teste")
    ax.set_title("Fração de avaliações negativas por mês")
    ax.set_ylabel("negativas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_negativas_por_mes.png", dpi=130)
    plt.close(fig)
    print(f"Figuras salvas em {FIG_DIR}")


if __name__ == "__main__":
    main()
