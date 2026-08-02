import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.predictor import ModelIntegrityError, ReviewFraudDetector
import config


def test_prediction_changes_for_distinct_reviews(tmp_path):
    texts = ["excellent quality", "amazing product", "terrible damaged item", "useless waste money"]
    labels = [0, 0, 1, 1]
    vectorizer = TfidfVectorizer().fit(texts)
    matrix = hstack([vectorizer.transform(texts), csr_matrix([[5], [5], [1], [1]])])
    model = LogisticRegression(random_state=42).fit(matrix, labels)
    model_path, vectorizer_path = tmp_path / "model.pkl", tmp_path / "vectorizer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    detector = ReviewFraudDetector(model_path=model_path, vectorizer_path=vectorizer_path)

    review_a = "Absolutely fantastic experience, highly recommend this seller"
    review_b = "Terrible quality, product broke immediately and I would never buy again"

    result_a = detector.predict(review_a, 5.0)
    result_b = detector.predict(review_b, 5.0)

    assert result_a.review_text == review_a
    assert result_b.review_text == review_b
    assert result_a.label_name != result_b.label_name or result_a.confidence != result_b.confidence
    assert result_a.probabilities != result_b.probabilities
    assert result_a.to_dict()["debug"]["processed_review"]


def test_placeholder_artifact_is_rejected(tmp_path):
    vectorizer = TfidfVectorizer().fit(["review sample 1 with generic restaurant feedback text"])
    vectorizer_path = tmp_path / "placeholder_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    detector = ReviewFraudDetector(model_path=config.MODEL_PATH, vectorizer_path=vectorizer_path)
    with __import__("pytest").raises(ModelIntegrityError):
        detector.predict("Any real review text", 3)
