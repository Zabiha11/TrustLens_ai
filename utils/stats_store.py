"""Persistent storage for dashboard metrics and recent predictions."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import config


class StatsStore:
    """Thread-safe JSON-backed store for prediction history and aggregates."""

    def __init__(self, filepath: Optional[Path] = None, max_recent: int = 50):
        self.filepath = filepath or config.STATS_FILE
        self.max_recent = max_recent
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self._write(
                {
                    "total_analyzed": 0,
                    "fake_count": 0,
                    "genuine_count": 0,
                    "recent": [],
                    "ai_stats": {
                        "human_count": 0,
                        "possibly_ai_count": 0,
                        "highly_likely_ai_count": 0,
                        "avg_ai_probability": 0.0,
                        "total_ai_evaluations": 0,
                    },
                }
            )

    def _read(self) -> dict[str, Any]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_prediction(
        self,
        review_text: str,
        label_name: str,
        confidence: float,
        rating: Optional[float] = None,
        ai_detection: Optional[dict[str, Any]] = None,
        trust_score: Optional[dict[str, Any]] = None,
    ) -> None:
        preview = review_text[:120] + ("..." if len(review_text) > 120 else "")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "review_preview": preview,
            "prediction": label_name,
            "confidence": round(confidence, 4),
            "rating": rating,
        }
        if ai_detection is not None:
            entry["ai_detection"] = ai_detection
        if trust_score is not None:
            entry["trust_score"] = trust_score

        with self._lock:
            data = self._read()
            data["total_analyzed"] += 1
            if label_name == config.LABEL_NAMES[config.LABEL_FAKE]:
                data["fake_count"] += 1
            else:
                data["genuine_count"] += 1

            ai_stats = data.setdefault("ai_stats", {})
            if ai_detection:
                status = str(ai_detection.get("status", "Likely Human"))
                if status == "Highly Likely AI Generated":
                    ai_stats["highly_likely_ai_count"] = ai_stats.get("highly_likely_ai_count", 0) + 1
                elif status == "Possibly AI Generated":
                    ai_stats["possibly_ai_count"] = ai_stats.get("possibly_ai_count", 0) + 1
                else:
                    ai_stats["human_count"] = ai_stats.get("human_count", 0) + 1

                total_evaluations = ai_stats.get("total_ai_evaluations", 0)
                avg_ai_probability = float(ai_stats.get("avg_ai_probability", 0.0))
                ai_probability = float(ai_detection.get("ai_probability", 0.0))
                ai_stats["avg_ai_probability"] = round(
                    ((avg_ai_probability * total_evaluations) + ai_probability) / (total_evaluations + 1),
                    4,
                )
                ai_stats["total_ai_evaluations"] = total_evaluations + 1

            data["recent"].insert(0, entry)
            data["recent"] = data["recent"][: self.max_recent]
            self._write(data)

    def record_batch(self, results: list[dict[str, Any]]) -> None:
        for item in results:
            self.record_prediction(
                review_text=item.get("review_text", ""),
                label_name=item.get("prediction", "Genuine"),
                confidence=item.get("confidence", 0.0),
                rating=item.get("rating"),
                ai_detection=item.get("ai_detection"),
                trust_score=item.get("trust_score"),
            )

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
        return {
            "total_analyzed": data["total_analyzed"],
            "fake_count": data["fake_count"],
            "genuine_count": data["genuine_count"],
            "recent_predictions": data["recent"],
            "ai_stats": data.get(
                "ai_stats",
                {
                    "human_count": 0,
                    "possibly_ai_count": 0,
                    "highly_likely_ai_count": 0,
                    "avg_ai_probability": 0.0,
                    "total_ai_evaluations": 0,
                },
            ),
        }


_store: Optional[StatsStore] = None


def get_stats_store() -> StatsStore:
    global _store
    if _store is None:
        _store = StatsStore()
    return _store
