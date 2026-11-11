"""Download the Olist reviews CSV from the official public Kaggle dataset."""

import shutil
from pathlib import Path

import kagglehub

DATASET = "olistbr/brazilian-ecommerce"
FILE_NAME = "olist_order_reviews_dataset.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / FILE_NAME
    if target.exists():
        print(f"Already present: {target}")
        return
    cache_dir = Path(kagglehub.dataset_download(DATASET))
    shutil.copy(cache_dir / FILE_NAME, target)
    print(f"Saved to {target}")


if __name__ == "__main__":
    main()
