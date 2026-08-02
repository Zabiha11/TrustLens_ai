"""AI-Generated Review Detection Module.

Integrates Groq API to detect synthetic / LLM-generated reviews (ChatGPT/Llama clones),
returning AI probability, confidence score, status, and concise explanation.
Includes offline heuristic fallback for zero-key environments.
"""

from dataclasses import dataclass
from typing import Any, Optional
import json
import re

import config


@dataclass
class AIDetectionResult:
    """Standardized output for AI-generated review detection."""

    review_text: str
    ai_probability: float
    confidence_score: float
    status: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_text": self.review_text,
            "ai_probability": round(self.ai_probability, 4),
            "confidence_score": round(self.confidence_score, 4),
            "status": self.status,
            "explanation": self.explanation,
        }


class GroqAIDetector:
    """Detects synthetic AI-generated review content using Groq LLM API or fallback heuristics."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_MODEL
        self._groq_client = None

    def _get_client(self):
        """Lazy-initialize Groq client if key is present."""
        if self._groq_client is None and self.api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.api_key)
            except Exception:
                self._groq_client = None
        return self._groq_client

    def detect(self, review_text: str) -> AIDetectionResult:
        """Detect synthetic AI origin for a given review text."""
        review_text = (review_text or "").strip()
        if not review_text:
            return AIDetectionResult(
                review_text="",
                ai_probability=0.0,
                confidence_score=1.0,
                status="Likely Human",
                explanation="Empty text submitted.",
            )

        client = self._get_client()
        if client:
            try:
                return self._detect_with_groq(client, review_text)
            except Exception:
                # Fallback to offline heuristic on API error
                return self._detect_with_fallback_heuristics(review_text)
        else:
            return self._detect_with_fallback_heuristics(review_text)

    def _detect_with_groq(self, client, review_text: str) -> AIDetectionResult:
        """Call Groq API with structured JSON output instructions."""
        prompt = f"""You are an expert AI-Generated Content & LLM Fraud Detector analyzing product reviews.
Analyze the following review text for stylistic markers of AI generation (e.g., over-enthusiastic generic phrasing, uniform sentence length, lack of personal experience details, cliché transition words like 'exceeded my expectations', 'game-changer', 'delighted').

Review Text:
"{review_text}"

Return ONLY a valid JSON object with the following schema:
{{
  "ai_probability": <float between 0.0 and 1.0>,
  "confidence_score": <float between 0.0 and 1.0>,
  "status": "<Likely Human | Possibly AI Generated | Highly Likely AI Generated>",
  "explanation": "<1-2 sentence concise explanation of why it was flagged or cleared>"
}}"""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a specialized AI detection API that responds strictly with JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        ai_prob = float(data.get("ai_probability", 0.3))
        confidence = float(data.get("confidence_score", 0.85))
        status = str(data.get("status", self._resolve_status(ai_prob)))
        explanation = str(data.get("explanation", "Analyzed using Groq LLM synthetic content evaluation."))

        return AIDetectionResult(
            review_text=review_text,
            ai_probability=ai_prob,
            confidence_score=confidence,
            status=status,
            explanation=explanation,
        )

    def _detect_with_fallback_heuristics(self, review_text: str) -> AIDetectionResult:
        """Offline statistical heuristic analyzer for synthetic AI review markers."""
        text_lower = review_text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        num_words = len(words)

        if num_words == 0:
            return AIDetectionResult(review_text, 0.0, 1.0, "Likely Human", "Short text.")

        # Common synthetic LLM review clichés
        llm_cliches = [
            "game-changer", "game changer", "exceeded my expectations", "exceeded all expectations",
            "highly recommend", "must-have", "must have", "delighted with", "seamlessly",
            "top-notch", "top notch", "impressed with the quality", "overall, i would say",
            "in conclusion", "furthermore", "moreover", "outstanding performance"
        ]

        cliche_matches = sum(1 for c in llm_cliches if c in text_lower)
        unique_word_ratio = len(set(words)) / num_words if num_words > 0 else 1.0

        # Uniform sentence length heuristic
        sentences = [s.strip() for s in re.split(r"[.!?]+", review_text) if s.strip()]
        avg_sent_len = num_words / max(len(sentences), 1)

        raw_score = (cliche_matches * 0.25) + (0.3 if 15 <= avg_sent_len <= 25 and len(sentences) >= 3 else 0.0)
        if unique_word_ratio < 0.65 and num_words > 20:
            raw_score += 0.2

        ai_prob = min(0.95, max(0.05, round(raw_score, 2)))
        status = self._resolve_status(ai_prob)

        if ai_prob >= 0.7:
            explanation = "High concentration of generic synthetic marketing phrases and uniform sentence structure."
        elif ai_prob >= 0.4:
            explanation = "Contains repetitive vocabulary and formal transition phrases common in AI text."
        else:
            explanation = "Authentic human writing style with natural variance and personal experience details."

        return AIDetectionResult(
            review_text=review_text,
            ai_probability=ai_prob,
            confidence_score=0.82,
            status=status,
            explanation=explanation,
        )

    @staticmethod
    def _resolve_status(ai_prob: float) -> str:
        if ai_prob >= 0.70:
            return "Highly Likely AI Generated"
        elif ai_prob >= 0.40:
            return "Possibly AI Generated"
        else:
            return "Likely Human"


# Singleton instance
_ai_detector: Optional[GroqAIDetector] = None


def get_ai_detector() -> GroqAIDetector:
    global _ai_detector
    if _ai_detector is None:
        _ai_detector = GroqAIDetector()
    return _ai_detector
