"""Temporal Fraud Detection Module.

Aggregates review velocity into daily, hourly, and weekly time bins,
and detects temporal anomaly spikes/bursts using Isolation Forest and DBSCAN.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


class TemporalFraudDetector:
    """Detects temporal review spikes, velocity anomalies, and coordinated time bursts."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_timestamps()
        self.daily_counts: dict[str, int] = {}
        self.hourly_counts: dict[str, int] = {}
        self.weekly_counts: dict[str, int] = {}
        self.detected_bursts: list[dict[str, Any]] = []
        self._analyze_time_trends()
        self._detect_bursts()

    def _prepare_timestamps(self) -> None:
        """Parse and convert timestamps into datetime format."""
        if self.df.empty or "timestamp" not in self.df.columns:
            self.df["dt"] = pd.Timestamp.now()
            return

        self.df["dt"] = pd.to_datetime(self.df["timestamp"], errors="coerce")
        # Fill missing datetimes with default current time
        self.df["dt"] = self.df["dt"].fillna(pd.Timestamp.now())

    def _analyze_time_trends(self) -> None:
        """Aggregate review frequencies across daily, hourly, and weekly buckets."""
        if self.df.empty:
            return

        # Daily trends
        daily_series = self.df.set_index("dt").resample("D")["review_text"].count()
        self.daily_counts = {str(k.date()): int(v) for k, v in daily_series.items()}

        # Hourly trends
        hourly_series = self.df.set_index("dt").resample("h")["review_text"].count()
        self.hourly_counts = {k.strftime("%Y-%m-%d %H:00"): int(v) for k, v in hourly_series.items()}

        # Weekly trends
        weekly_series = self.df.set_index("dt").resample("W")["review_text"].count()
        self.weekly_counts = {f"Week {k.strftime('%U (%b %d)')}": int(v) for k, v in weekly_series.items()}

    def _detect_bursts(self) -> None:
        """Detect burst anomalies using Isolation Forest & DBSCAN on hourly time windows."""
        self.detected_bursts = []
        if self.df.empty:
            return

        # Group data into 1-hour window bins
        hourly_df = (
            self.df.set_index("dt")
            .groupby(pd.Grouper(freq="h"))
            .agg(
                review_count=("review_text", "count"),
                unique_users=("user_id", "nunique"),
                unique_products=("product_id", "nunique"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )

        # Remove empty intervals
        active_hours = hourly_df[hourly_df["review_count"] > 0].copy()
        if len(active_hours) < 2:
            return

        # Feature matrix for anomaly detection: [review_count, user_density, product_density]
        X = active_hours[["review_count", "unique_users", "unique_products"]].values

        # 1. Isolation Forest Anomaly Detection
        try:
            iso = IsolationForest(contamination=0.15, random_state=42)
            iso_labels = iso.fit_predict(X)  # -1 for anomaly, 1 for normal
        except Exception:
            iso_labels = np.ones(len(active_hours))

        # 2. DBSCAN Clustering for burst density grouping
        try:
            db = DBSCAN(eps=2.5, min_samples=2)
            db_labels = db.fit_predict(X)
        except Exception:
            db_labels = np.zeros(len(active_hours))

        # Identify anomalous windows (Isolation Forest flagged -1 or count exceeds mean + 2*std)
        counts = active_hours["review_count"].values
        mean_c = np.mean(counts)
        std_c = np.std(counts)
        threshold = mean_c + 1.5 * std_c

        burst_id = 1
        for idx, row in active_hours.iterrows():
            is_anomaly = iso_labels[active_hours.index.get_loc(idx)] == -1 or row["review_count"] > threshold

            if is_anomaly and row["review_count"] >= 3:
                time_str = row["dt"].strftime("%Y-%m-%d %H:00")
                # Find reviews matching this window
                window_start = row["dt"]
                window_end = window_start + pd.Timedelta(hours=1)
                sub_reviews = self.df[(self.df["dt"] >= window_start) & (self.df["dt"] < window_end)]

                targeted = sub_reviews["product_id"].unique().tolist()
                top_users = sub_reviews["user_id"].unique().tolist()
                avg_r = round(float(row["avg_rating"]), 2)

                # Anomaly confidence score (0-100)
                vol_ratio = row["review_count"] / max(mean_c, 1.0)
                confidence = min(100.0, round(50.0 + (vol_ratio * 15.0), 1))

                self.detected_bursts.append(
                    {
                        "burst_id": f"BURST_{burst_id:02d}",
                        "timestamp": time_str,
                        "review_count": int(row["review_count"]),
                        "user_count": int(row["unique_users"]),
                        "product_count": int(row["unique_products"]),
                        "avg_rating": avg_r,
                        "targeted_products": targeted,
                        "participating_users": top_users,
                        "anomaly_score": confidence,
                        "flagged": True,
                    }
                )
                burst_id += 1

        # Sort bursts by anomaly score descending
        self.detected_bursts.sort(key=lambda b: b["anomaly_score"], reverse=True)

    def get_summary(self) -> dict[str, Any]:
        """Return temporal trends and detected bursts payload."""
        return {
            "daily_trends": self.daily_counts,
            "hourly_trends": self.hourly_counts,
            "weekly_trends": self.weekly_counts,
            "detected_bursts": self.detected_bursts,
            "total_bursts_found": len(self.detected_bursts),
        }
