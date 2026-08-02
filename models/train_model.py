"""
Train TF-IDF + XGBoost fraud detector on YelpCHI dataset.

Expected CSV columns: review_id, user_id, product_id, review_text, rating, timestamp, label
Phase 1 uses review_text and rating only.

Usage:
    python models/train_model.py
    python models/train_model.py --data-path data/yelpchi.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Allow running from project root or models/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from utils.predictor import ReviewFraudDetector
from utils.preprocessing import build_feature_matrix, clean_review_text


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}\n"
            "Download YelpCHI and place it at data/yelpchi.csv\n"
            "Required columns: review_text, rating, label"
        )

    df = pd.read_csv(path)
    required = {"review_text", "rating", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.dropna(subset=["review_text", "label"]).copy()
    df["review_text"] = df["review_text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def train(data_path: Path, test_size: float = 0.2) -> None:
    print(f"Loading dataset from {data_path}...")
    df = load_dataset(data_path)
    print(f"Samples: {len(df)} | Fake: {(df['label'] == 1).sum()} | Genuine: {(df['label'] == 0).sum()}")

    training_frame = df[["review_text", "rating", "label"]].copy()
    training_frame = training_frame.dropna(subset=["review_text", "rating", "label"])
    training_frame["clean_text"] = training_frame["review_text"].apply(clean_review_text)
    training_frame["rating"] = training_frame["rating"].astype(float).clip(1, 5)

    texts = training_frame["clean_text"].tolist()
    labels = training_frame["label"].astype(int).values
    ratings = training_frame["rating"].astype(float).values

    text_signatures = training_frame["clean_text"].str.replace(r"\d+", "<number>", regex=True)
    if text_signatures.nunique() < max(20, int(len(training_frame) * 0.01)):
        raise ValueError(
            "Training data has insufficient distinct review text after normalisation. "
            "Refusing to overwrite the model with a degenerate classifier."
        )

    X_train_text, X_test_text, y_train, y_test, r_train, r_test = train_test_split(
        texts,
        labels,
        ratings,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    print("Fitting TF-IDF vectorizer with unigrams and bigrams...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    vectorizer.fit(X_train_text)

    X_train = hstack(
        [vectorizer.transform(X_train_text), csr_matrix(r_train.reshape(-1, 1))],
        format="csr",
    )
    X_test = hstack(
        [vectorizer.transform(X_test_text), csr_matrix(r_test.reshape(-1, 1))],
        format="csr",
    )

    print("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=max(1.0, len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Dataset Class Distribution ---")
    print(f"Genuine: {(df['label'] == 0).sum()}")
    print(f"Fake:    {(df['label'] == 1).sum()}")

    print("\n--- Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fake"]))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print(f"\nSaved model      -> {config.MODEL_PATH}")
    print(f"Saved vectorizer -> {config.VECTORIZER_PATH}")

    print("\n--- Example Predictions ---")
    detector = ReviewFraudDetector(model_path=config.MODEL_PATH, vectorizer_path=config.VECTORIZER_PATH)
    example_reviews = [
        "Absolutely fantastic experience, highly recommend this seller",
        "Terrible quality, product broke immediately and I would never buy again",
        "It was okay but nothing special",
        "This product exceeded all expectations and feels like a game changer",
        "I would not recommend this, poor quality and the packaging arrived broken",
        "The service was fine and the delivery arrived on time",
        "Wonderful purchase, smooth experience, I am delighted",
        "Awful quality, refund requested, the item was defective",
        "A decent buy with average quality and average service",
        "The packaging was damaged and I am disappointed",
    ]
    for review in example_reviews:
        result = detector.predict(review, 5.0)
        print(f"- {review} => {result.label_name} ({result.confidence:.4f})")

    confidences = {round(detector.predict(review, 5.0).confidence, 4) for review in example_reviews}
    if len(confidences) > 1:
        print("Prediction pipeline verification: varied model confidence scores were observed across example reviews.")
    else:
        print("Prediction pipeline verification: the example set did not produce varied confidence scores.")


def main():
    parser = argparse.ArgumentParser(description="Train TrustLens AI fraud detector")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "yelpchi.csv",
        help="Path to YelpCHI CSV file",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train(args.data_path, args.test_size)


if __name__ == "__main__":
    main()
