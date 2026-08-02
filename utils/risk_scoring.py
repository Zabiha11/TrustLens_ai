"""Reviewer Risk Scoring Module.

Computes a comprehensive Risk Score (0-100) and Risk Level for reviewers based on:
1. Review Volume
2. Rating Behavior & Extremity
3. Product Target Concentration
4. Text Duplication & Similarity (TF-IDF Cosine Similarity)
5. Rapid Reviewing Velocity (Time Intervals)
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ReviewerRiskScorer:
    """Calculates multidimensional risk metrics for all reviewers in a dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.risk_profiles: list[dict[str, Any]] = []
        self._compute_all_scores()

    def _compute_all_scores(self) -> None:
        """Process reviewer profiles and calculate risk components."""
        if self.df.empty:
            self.risk_profiles = []
            return

        # Precompute TF-IDF matrix for duplicate text analysis across dataset
        text_list = self.df["review_text"].astype(str).tolist()
        try:
            tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            tfidf_matrix = tfidf.fit_transform(text_list)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception:
            sim_matrix = np.zeros((len(self.df), len(self.df)))

        # Group reviews by reviewer ID
        user_groups = self.df.groupby("user_id")

        profiles = []
        for user_id, group in user_groups:
            num_reviews = len(group)
            ratings = group["rating"].astype(float).values
            unique_products = group["product_id"].nunique()

            # 1. Volume Factor (0-20 pts)
            volume_score = min(20.0, num_reviews * 3.5)

            # 2. Rating Extremity (0-25 pts)
            avg_rating = float(np.mean(ratings))
            rating_std = float(np.std(ratings)) if num_reviews > 1 else 0.0
            extremity = abs(avg_rating - 3.0) / 2.0  # 0 if avg=3, 1 if avg=1 or 5
            rating_score = (extremity * 20.0) + (5.0 if rating_std == 0 and num_reviews >= 2 else 0.0)

            # 3. Product Coverage / Concentration (0-15 pts)
            # High reviews on very few products or spamming many products rapidly
            concentration_score = min(15.0, (num_reviews / max(unique_products, 1)) * 3.0)

            # 4. Duplicate Review Patterns (0-25 pts)
            user_indices = group.index.tolist()
            duplication_score = 0.0
            if len(user_indices) > 0 and sim_matrix.shape[0] > 0:
                # Compare similarity among user's own reviews and against rest of corpus
                sub_sim = sim_matrix[user_indices, :]
                # Set diagonal (self-match) to 0
                np.fill_diagonal(sim_matrix[user_indices, :][:, user_indices], 0)
                max_sim = float(np.max(sub_sim)) if sub_sim.size > 0 else 0.0
                if max_sim > 0.85:
                    duplication_score = 25.0
                elif max_sim > 0.6:
                    duplication_score = 15.0
                elif max_sim > 0.4:
                    duplication_score = 8.0

            # 5. Rapid Velocity Factor (0-15 pts)
            velocity_score = 0.0
            timestamps = pd.to_datetime(group["timestamp"], errors="coerce").dropna().sort_values()
            if len(timestamps) >= 2:
                time_diffs = timestamps.diff().dropna()
                min_diff_seconds = time_diffs.min().total_seconds()
                if min_diff_seconds < 120:  # < 2 mins between reviews
                    velocity_score = 15.0
                elif min_diff_seconds < 600:  # < 10 mins
                    velocity_score = 10.0
                elif min_diff_seconds < 1800: # < 30 mins
                    velocity_score = 5.0

            # Total aggregate score out of 100
            total_score = min(100.0, round(volume_score + rating_score + concentration_score + duplication_score + velocity_score, 1))

            # Risk level categorization
            if total_score >= 80:
                risk_level = "Critical"
                level_badge = "danger"
            elif total_score >= 60:
                risk_level = "High"
                level_badge = "warning"
            elif total_score >= 35:
                risk_level = "Medium"
                level_badge = "info"
            else:
                risk_level = "Low"
                level_badge = "success"

            profiles.append(
                {
                    "user_id": str(user_id),
                    "risk_score": total_score,
                    "risk_level": risk_level,
                    "level_badge": level_badge,
                    "review_count": num_reviews,
                    "avg_rating": round(avg_rating, 2),
                    "unique_products": unique_products,
                    "breakdown": {
                        "volume_score": round(volume_score, 1),
                        "rating_score": round(rating_score, 1),
                        "concentration_score": round(concentration_score, 1),
                        "duplication_score": round(duplication_score, 1),
                        "velocity_score": round(velocity_score, 1),
                    },
                }
            )

        # Sort profiles by risk_score descending
        profiles.sort(key=lambda p: p["risk_score"], reverse=True)
        self.risk_profiles = profiles

    def get_summary(self) -> dict[str, Any]:
        """Return summary statistics and reviewer risk list."""
        if not self.risk_profiles:
            return {
                "total_reviewers": 0,
                "high_risk_count": 0,
                "avg_risk_score": 0.0,
                "reviewers": [],
            }

        scores = [p["risk_score"] for p in self.risk_profiles]
        high_risk = sum(1 for p in self.risk_profiles if p["risk_score"] >= 60.0)
        avg_score = round(float(np.mean(scores)), 1)

        return {
            "total_reviewers": len(self.risk_profiles),
            "high_risk_count": high_risk,
            "avg_risk_score": avg_score,
            "reviewers": self.risk_profiles,
        }
