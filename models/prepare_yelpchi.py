"""
Convert YelpChi.mat (CARE-GNN format) to TrustLens CSV.

Download YelpChi.zip from:
  https://github.com/YingtongDou/CARE-GNN/tree/master/data

Usage:
    python models/prepare_yelpchi.py --zip-path data/YelpChi.zip
    python models/prepare_yelpchi.py --mat-path data/YelpChi/YelpChi.mat
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "data" / "yelpchi.csv"


def load_mat(path: Path):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Install scipy: pip install scipy") from exc
    return loadmat(str(path), struct_as_record=False, squeeze_me=True)


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    mat_files = list(dest.rglob("YelpChi.mat"))
    if not mat_files:
        raise FileNotFoundError("YelpChi.mat not found inside zip archive")
    return mat_files[0]


def build_dataframe(mat_path: Path) -> pd.DataFrame:
    data = load_mat(mat_path)
    labels = np.array(data["label"]).astype(int).ravel()

    n = len(labels)
    review_text = _extract_text(data, n)
    ratings = _extract_ratings(data, n)

    df = pd.DataFrame(
        {
            "review_id": np.arange(n),
            "user_id": np.arange(n),
            "product_id": np.zeros(n, dtype=int),
            "review_text": review_text,
            "rating": ratings,
            "timestamp": pd.date_range("2010-01-01", periods=n, freq="h"),
            "label": labels,
        }
    )
    return df


def _extract_text(data: dict, n: int) -> list[str]:
    for key in ("allReviewContent", "review_content", "text", "content"):
        if key in data:
            texts = data[key]
            if isinstance(texts, np.ndarray):
                return [str(t) for t in texts.ravel()[:n]]
    raise ValueError(
        "YelpChi.mat does not contain raw review text. This release cannot train a TF-IDF "
        "review-text classifier; use a dataset with real labeled review_text values instead."
    )


def _extract_ratings(data: dict, n: int) -> list[float]:
    for key in ("rating", "ratings", "stars"):
        if key in data:
            vals = np.array(data[key]).astype(float).ravel()
            return np.clip(vals[:n], 1, 5).tolist()

    if "features" in data:
        feats = np.array(data["features"])
        if feats.ndim == 2 and feats.shape[0] == n:
            # First feature in YelpChi handcrafted set correlates with rating
            col = np.clip(feats[:, 0], 1, 5)
            return col.tolist()

    return [3.0] * n


def main():
    parser = argparse.ArgumentParser(description="Prepare YelpCHI CSV for TrustLens AI")
    parser.add_argument("--zip-path", type=Path, help="Path to YelpChi.zip")
    parser.add_argument("--mat-path", type=Path, help="Path to YelpChi.mat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.mat_path:
        mat_path = args.mat_path
    elif args.zip_path:
        mat_path = extract_zip(args.zip_path, ROOT / "data" / "YelpChi")
    else:
        default_mat = ROOT / "data" / "YelpChi" / "YelpChi.mat"
        if default_mat.exists():
            mat_path = default_mat
        else:
            parser.error("Provide --zip-path or --mat-path, or place YelpChi.mat in data/YelpChi/")

    print(f"Reading {mat_path}...")
    df = build_dataframe(mat_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows -> {args.output}")
    print(f"Fake: {(df['label']==1).sum()} | Genuine: {(df['label']==0).sum()}")


if __name__ == "__main__":
    main()
