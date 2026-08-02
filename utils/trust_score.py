"""Unified Trust Score & Natural Language Fraud Summary Module.

Calculates an integrated Trust Score (0-100), assigns Risk Level badges (Green/Yellow/Red),
and generates professional natural language explanations summarizing all fraud signals.
"""

from typing import Any, Optional


class TrustScoreCalculator:
    """Computes multidimensional Trust Score and synthesizes natural language fraud summaries."""

    @staticmethod
    def calculate(
        fake_probability: float,
        reviewer_risk_score: float = 0.0,
        temporal_anomaly_score: float = 0.0,
        community_suspicion_score: float = 0.0,
        ai_generated_probability: float = 0.0,
    ) -> dict[str, Any]:
        """Compute 0-100 Trust Score and synthesize natural language summary."""

        # Convert probabilities (0.0 to 1.0) to 0-100 scale
        p_fake_100 = min(100.0, max(0.0, fake_probability * 100.0))
        p_ai_100 = min(100.0, max(0.0, ai_generated_probability * 100.0))
        s_risk_100 = min(100.0, max(0.0, reviewer_risk_score))
        s_burst_100 = min(100.0, max(0.0, temporal_anomaly_score))
        s_comm_100 = min(100.0, max(0.0, community_suspicion_score))

        # Weighted penalty calculation
        raw_penalty = (
            (0.30 * p_fake_100)
            + (0.25 * s_risk_100)
            + (0.15 * s_burst_100)
            + (0.15 * s_comm_100)
            + (0.15 * p_ai_100)
        )

        trust_score = max(0.0, min(100.0, round(100.0 - raw_penalty, 1)))

        # Risk Level & Badge Color Assignment
        if trust_score >= 75.0:
            risk_level = "Low Risk"
            badge_color = "Green"
            badge_class = "success"
        elif trust_score >= 50.0:
            risk_level = "Medium Risk"
            badge_color = "Yellow"
            badge_class = "warning"
        else:
            risk_level = "High Risk"
            badge_color = "Red"
            badge_class = "danger"

        # Generate Natural Language Fraud Summary
        summary = TrustScoreCalculator._generate_natural_summary(
            risk_level=risk_level,
            fake_prob=p_fake_100,
            reviewer_risk=s_risk_100,
            burst_score=s_burst_100,
            comm_suspicion=s_comm_100,
            ai_prob=p_ai_100,
        )

        return {
            "trust_score": trust_score,
            "risk_level": risk_level,
            "badge_color": badge_color,
            "badge_class": badge_class,
            "natural_summary": summary,
            "breakdown": {
                "text_fake_penalty": round(0.30 * p_fake_100, 1),
                "reviewer_risk_penalty": round(0.25 * s_risk_100, 1),
                "temporal_burst_penalty": round(0.15 * s_burst_100, 1),
                "community_suspicion_penalty": round(0.15 * s_comm_100, 1),
                "ai_generated_penalty": round(0.15 * p_ai_100, 1),
            },
        }

    @staticmethod
    def _generate_natural_summary(
        risk_level: str,
        fake_prob: float,
        reviewer_risk: float,
        burst_score: float,
        comm_suspicion: float,
        ai_prob: float,
    ) -> str:
        """Synthesize natural language explanation combining all active fraud indicators."""
        reasons = []

        if comm_suspicion >= 60.0:
            reasons.append("it belongs to a suspicious reviewer cluster")

        if burst_score >= 50.0:
            reasons.append("was posted during an unusual review velocity burst")

        if reviewer_risk >= 60.0 or fake_prob >= 65.0:
            reasons.append("contains highly repetitive language and abnormal rating patterns")

        if ai_prob >= 50.0:
            reasons.append("exhibits characteristics commonly associated with AI-generated synthetic content")

        if not reasons:
            return (
                "This review shows a High Trust Score with no significant anomalies detected across text patterns, "
                "reviewer history, network topology, or posting timestamps."
            )

        if len(reasons) == 1:
            reason_str = reasons[0]
        elif len(reasons) == 2:
            reason_str = f"{reasons[0]} and {reasons[1]}"
        else:
            reason_str = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

        return f"This review has a {risk_level} profile because {reason_str}."
