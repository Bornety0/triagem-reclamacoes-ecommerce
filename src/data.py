"""Loading, cleaning and labelling of the Olist reviews.

Every step here is shared by training, evaluation and the demo, so the text a
user types in the app goes through exactly the same normalisation the model
was trained on.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "olist_order_reviews_dataset.csv"

NEGATIVE = 1
POSITIVE = 0
LABEL_NAMES = {NEGATIVE: "negativa", POSITIVE: "positiva"}

# Reviews created on or after this date form the test set (temporal split).
TEST_START = pd.Timestamp("2018-06-01")

_WHITESPACE = re.compile(r"\s+")
_LETTER = re.compile(r"[a-z]")


def normalize_text(text: str) -> str:
    """Lower-case, strip accents and collapse whitespace.

    Customers write "ótimo", "Otimo" and "otimo" for the same word; folding them
    together lets the model learn one feature instead of three sparse ones.
    """
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WHITESPACE.sub(" ", text).strip()


def score_to_label(score: int) -> int | None:
    """1-2 stars -> negative, 4-5 -> positive, 3 -> None (ambiguous, left out)."""
    if score <= 2:
        return NEGATIVE
    if score >= 4:
        return POSITIVE
    return None


def load_reviews(path: Path = RAW_CSV) -> pd.DataFrame:
    """All reviews that carry a usable comment, with the normalised text.

    Returns one row per review with: review_id, score, date, text (original)
    and clean (normalised). Score 3 is kept here so it can be analysed; the
    labelled subset comes from `labelled`.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="review_id")
    df = df.dropna(subset=["review_comment_message"])
    df = df.rename(
        columns={
            "review_score": "score",
            "review_comment_message": "text",
            "review_creation_date": "date",
        }
    )
    df["text"] = df["text"].astype(str).str.strip()
    df["clean"] = df["text"].map(normalize_text)
    # Comments with no letter at all ("." , "10", "!!!") carry no sentiment.
    df = df[df["clean"].str.contains(_LETTER)]
    df["date"] = pd.to_datetime(df["date"])
    return df[["review_id", "score", "date", "text", "clean"]].reset_index(drop=True)


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Only the reviews with a binary label (score 3 removed)."""
    out = df.copy()
    out["label"] = out["score"].map(score_to_label)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the past, test on the most recent reviews.

    A random split would let the model see reviews of the same period (same
    products, same delivery problems) in both halves; a temporal split mirrors
    how the model is used: trained once, then applied to reviews that arrive
    later.
    """
    train = df[df["date"] < TEST_START]
    test = df[df["date"] >= TEST_START]
    return train.reset_index(drop=True), test.reset_index(drop=True)
