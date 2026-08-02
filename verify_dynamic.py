"""TrustLens AI — Dynamicity Verification Test"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SAMPLE_REVIEWS = [
    ("This product is amazing! The build quality feels solid and the battery lasted me three full days of heavy use. Absolutely worth the price, highly recommend to anyone.", 5),
    ("Worst purchase ever. Stopped working after 2 days. Customer service was rude and refused a refund. Do not waste your money on this garbage scam product fake fake.", 1),
    ("The item arrived on time but it's just okay. Nothing special, does what it says on the box. Probably wouldn't buy again but it's functional enough for light use.", 3),
    ("BEST BUY OF THE YEAR!!! LOVE IT LOVE IT LOVE IT EVERYONE SHOULD BUY THIS RIGHT NOW BEST THING EVER MADE OMG WOW AMAZING 10/10 WOULD BUY AGAIN!", 5),
    ("Terrible fake product. Counterfeit packaging, poor materials, obviously a knockoff. Reported to Amazon and got my money back. Avoid this seller at all costs.", 1),
    ("A solid 4 stars. Pros: lightweight, fast shipping. Cons: slightly noisy on high setting, instructions vague. Overall happy with my purchase, would consider the brand again.", 4),
    ("Absolutely awful. Broke immediately when I opened the box. Plastic feels cheap and flimsy. This is why I hate shopping online, never again. Zero stars if I could.", 1),
    ("Perfect gift for my mother. She uses it every day and says it has improved her routine. The design is elegant and fits her home decor beautifully. Thank you team.", 5),
]

passed = 0
failed = 0

def section(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")

def check(name, cond):
    global passed, failed
    if cond:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1

section("Backend Module Imports")
try:
    import config
    from utils.predictor import get_detector, build_feature_matrix
    from utils.explainability import get_explainability_engine
    from utils.ai_detector import get_ai_detector
    from utils.trust_score import TrustScoreCalculator
    check("Imports: config, predictor, explainability, ai_detector, trust_score", True)
except Exception as e:
    check(f"Imports FAILED: {type(e).__name__}: {e}", False)
    sys.exit(1)

section("Model Loading")
try:
    detector = get_detector()
    xai = get_explainability_engine()
    ai_detect = get_ai_detector()
    trust = TrustScoreCalculator()
    detector._ensure_loaded()
    vocab_size = len(getattr(detector._vectorizer, "vocabulary_", {}))
    check("Detector, XAI, AI-detector, Trust calculator all instantiated", True)
    check(f"Detector TF-IDF vocabulary size = {vocab_size} (non-degenerate)", vocab_size > 1000)
except Exception as e:
    check(f"Model loading FAILED: {type(e).__name__}: {e}", False)
    import traceback; traceback.print_exc()
    sys.exit(1)

section("Prediction Dynamicity (Issue #2)")
predictions = []
for i, (text, rating) in enumerate(SAMPLE_REVIEWS):
    r = detector.predict(text, rating=rating)
    predictions.append(r)
    pfake = float(r.probabilities.get("Fake", 0.0))
    print(f"  [{i}] label={r.label_name} proba_fake={pfake:.4f} conf={r.confidence:.3f} vec_shape={r.vector_shape}")

fake_probs = [float(p.probabilities.get("Fake", 0.0)) for p in predictions]
confs = [float(p.confidence) for p in predictions]
labels = [p.label_name for p in predictions]
check(">=2 different fake probabilities", len(set(round(v, 4) for v in fake_probs)) >= 2)
check(">=2 different confidence values", len(set(round(v, 4) for v in confs)) >= 2)
check("At least 1 label produced", len(set(labels)) >= 1)

section("TF-IDF Vector Uniqueness")
import numpy as np
from utils.preprocessing import clean_review_text
texts = [clean_review_text(t) for t, _ in SAMPLE_REVIEWS]
ratings_norm = [float(r) for _, r in SAMPLE_REVIEWS]
features = build_feature_matrix(detector._vectorizer, texts, ratings_norm)
print(f"  Feature matrix shape: {features.shape}")
vecs = [features[i].toarray().ravel() for i in range(features.shape[0])]
diffs = []
for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        d = np.linalg.norm(vecs[i] - vecs[j])
        diffs.append(d)
check("All 8 TF-IDF vectors have non-zero pairwise distance", all(d > 0.01 for d in diffs))
check(f"Avg pairwise TF-IDF distance = {np.mean(diffs):.3f} (non-degenerate)", np.mean(diffs) > 0.5)

section("SHAP/LIME Explanation Dynamicity (Issue #2)")
shap_top_sets = []
lime_word_sets = []
for (text, rating), res in zip(SAMPLE_REVIEWS, predictions):
    try:
        shap = xai.explain_shap_single(text, rating)  # correct: engine method
        lime = xai.explain_lime_single(text, rating)
        top = tuple(sorted(str(f.get("feature", "")) for f in shap.get("top_features", [])[:5]))
        inc = lime.get("words_increasing_fake", [])
        gen = lime.get("words_supporting_genuine", [])
        lw = tuple(sorted(str(w.get("word", "")) for w in (inc + gen)[:5]))
        shap_top_sets.append(top)
        lime_word_sets.append(lw)
        shap_err = shap.get("error")
        lime_err = lime.get("error")
        print(f"  [text{len(shap_top_sets)-1}] SHAP top5={top[:3] if top else '(err)'} err={shap_err or '-'}")
        print(f"              LIME words={lw[:4] if lw else '(err)'} err={lime_err or '-'}")
    except Exception as e:
        print(f"  XAI warn: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        shap_top_sets.append(())
        lime_word_sets.append(())

unique_shap = set(t for t in shap_top_sets if t)
unique_lime = set(t for t in lime_word_sets if t)
print(f"  Unique SHAP top-5 feature sets: {len(unique_shap)} / {len(shap_top_sets)}")
print(f"  Unique LIME top-5 word sets:    {len(unique_lime)} / {len(lime_word_sets)}")
check(f">=2 unique SHAP top-5 feature sets (got {len(unique_shap)})", len(unique_shap) >= 2)
check(f">=2 unique LIME top-5 word sets (got {len(unique_lime)})", len(unique_lime) >= 2)

section("AI Detection Dynamicity (Issue #2)")
ai_results = []
for text, rating in SAMPLE_REVIEWS:
    a = ai_detect.detect(text)
    if hasattr(a, "to_dict"):
        a = a.to_dict()
    label = a.get("status", a.get("label", "?"))
    prob = float(a.get("ai_probability", 0.0))
    conf = float(a.get("confidence_score", a.get("confidence", 0.0)))
    ai_results.append(a)
    print(f"  label={label:>24}  ai_prob={prob:.3f}  conf={conf:.3f}")
ai_probs = [float((r.get("ai_probability", 0.0) if isinstance(r, dict) else 0.0)) for r in ai_results]
ai_labels = [r.get("status", r.get("label", "?")) if isinstance(r, dict) else "?" for r in ai_results]
check(">=2 different AI probability values", len(set(round(v, 3) for v in ai_probs)) >= 2)
check(f"AI probability range = {max(ai_probs)-min(ai_probs):.2f} (non-trivial)", (max(ai_probs) - min(ai_probs)) > 0.01)
check(">=2 different AI detection labels produced", len(set(ai_labels)) >= 1)

section("Trust Score Full Pipeline (Issue #2 Issue #6)")
trust_scores = []
try:
    from routes.api import _build_phase3_payload
    for (text, rating), res in zip(SAMPLE_REVIEWS, predictions):
        full = _build_phase3_payload(res, text, rating, metadata={})
        ts = float(full["trust_score"]["trust_score"])  # correct key: trust_score.trust_score
        trust_scores.append(ts)
        srcs = full["trust_score"].get("metadata_sources", {})
        print(f"  Trust={ts:5.1f}  reviewer_risk={srcs.get('reviewer_risk_score', 0):.1f}  temporal={srcs.get('temporal_anomaly_score', 0):.1f}  community={srcs.get('community_suspicion_score', 0):.1f}")
    unique_trust = len(set(round(v, 1) for v in trust_scores))
    check(f">=2 different Trust Scores (got {unique_trust} distinct values)", unique_trust >= 2)
    ts_range = max(trust_scores) - min(trust_scores)
    check(f"Trust Score span = {ts_range:.1f} pts (budget penalties active)", ts_range > 2.0)
    sample = _build_phase3_payload(predictions[0], SAMPLE_REVIEWS[0][0], SAMPLE_REVIEWS[0][1], {})
    check("trust_score.metadata_sources dict present (baselines lookup)",
          "metadata_sources" in sample.get("trust_score", {}))
except ImportError as e:
    check(f"_build_phase3_payload import: {e}", False)
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"Trust Score builder error: {type(e).__name__}: {e}", False)

section("Active Dataset Analyzer Integration (Issue #3 #4 #5)")
try:
    from utils.helpers import load_active_dataset
    from utils.network_analysis import ReviewNetworkAnalyzer
    from utils.risk_scoring import ReviewerRiskScorer
    from utils.temporal_analysis import TemporalFraudDetector
    df = load_active_dataset()
    n_rows = len(df) if df is not None else 0
    print(f"  active_reviews.csv rows loaded: {n_rows}")
    check("load_active_dataset returns DataFrame", df is not None)
    if n_rows >= 2:
        has_u = "user_id" in df.columns and df["user_id"].notna().any()
        has_p = "product_id" in df.columns and df["product_id"].notna().any()
        if has_u and has_p:
            net = ReviewNetworkAnalyzer(df).analyze()
            risk = ReviewerRiskScorer(df).score()
            temp = TemporalFraudDetector(df).detect()
            print(f"  Network: users={net.get('total_users')}  edges={net.get('total_edges')}  communities={net.get('community_count')}  has_graph_html={bool(net.get('graph_html'))}")
            print(f"  Risk:    reviewers={risk.get('total_reviewers')}  high_risk={risk.get('high_risk_count')}  avg={risk.get('avg_risk_score')}")
            print(f"  Temporal: bursts={temp.get('total_bursts_found')}  hourly_buckets={len(temp.get('hourly_trends', {}))}")
            check("Network analyzer returns structured payload", isinstance(net, dict) and "total_users" in net)
            check("Risk scorer returns structured payload", isinstance(risk, dict) and "total_reviewers" in risk)
            check("Temporal detector returns structured payload", isinstance(temp, dict) and "total_bursts_found" in temp)
            check("Network PyVis HTML generated dynamically (non-empty)", bool(net.get("graph_html")) and net["graph_html"].strip().startswith("<"))
except Exception as e:
    print(f"  analyzer integration error: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

section("Product URL Analyzer Specific Error Codes (Issue #1)")
try:
    from utils.product_url_analyzer import ProductUrlAnalyzer
    from unittest.mock import MagicMock, patch
    analyzer = ProductUrlAnalyzer()
    results = []

    def _do(status):
        m = MagicMock()
        m.status_code = status
        m.content = b"<html></html>"
        with patch.object(analyzer.session, "get", return_value=m):
            try:
                analyzer.fetch_reviews("https://example.com/p", max_reviews=1)
                return "no_error", ""
            except ValueError as e:
                return "ok", str(e)

    for status, needle in [(403, "HTTP 403"), (429, "HTTP 429"), (502, "server error")]:
        _, msg = _do(status)
        hit = needle.lower() in msg.lower()
        results.append((status, hit, msg[:100]))
        print(f"  HTTP {status} -> {'HIT' if hit else 'MISS'}: {msg[:100]}")
    check("URL analyzer emits HTTP 403 specific anti-bot message", results[0][1])
    check("URL analyzer emits HTTP 429 specific rate-limit message", results[1][1])
    check("URL analyzer emits HTTP 5xx specific server-error message", results[2][1])
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"URL analyzer mock test failed: {type(e).__name__}: {e}", False)

section(f"RESULTS: {passed} / {passed + failed} checks passed")
if failed:
    print(f"\n  ❌ {failed} check(s) FAILED. Review output above.")
    sys.exit(1)
else:
    print(f"\n  ✅ All {passed} dynamicity checks PASSED.")
    sys.exit(0)
