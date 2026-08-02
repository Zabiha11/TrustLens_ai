"""Product URL review retrieval and analysis helpers.

This module attempts to retrieve product reviews from a public product URL using
permissible scraping when possible. When the structure is unsupported or blocked,
it raises a clear error so the UI can guide the user to upload a CSV instead.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ProductUrlAnalyzer:
    """Fetches review snippets from a product page and normalizes them for analysis."""

    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def fetch_reviews(self, product_url: str, max_reviews: int = 20) -> list[dict[str, Any]]:
        """Return a normalized list of review dictionaries for downstream analysis."""
        url = (product_url or "").strip()
        if not url:
            raise ValueError("A product URL is required.")
        logger.info("Step 1/5 — Product URL analyzer started: url=%s max_reviews=%s", url, max_reviews)

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Please provide a valid product URL (include http:// or https://).")
        logger.info("Step 2/5 — URL parsed: scheme=%s netloc=%s path=%s", parsed.scheme, parsed.netloc, parsed.path)

        try:
            logger.info("Step 3/5 — Sending HTTP GET request with UA header")
            response = self.session.get(url, timeout=self.timeout)
            logger.info("Step 3/5 — HTTP response received: status=%s len_bytes=%s", response.status_code, len(response.content))
        except requests.Timeout as exc:
            logger.warning("Product URL fetch timed out for %s after %ss: %s", url, self.timeout, exc)
            raise ValueError("Request timed out. The target site may be slow or block automated access. Try again or upload a CSV.") from exc
        except requests.RequestException as exc:
            logger.warning("Product URL fetch failed for %s: %s", url, exc)
            raise ValueError("Unable to load the product page. The site may block automated retrieval or require a browser session.") from exc

        if response.status_code == 403:
            logger.warning("Product URL blocked with 403: %s", url)
            raise ValueError("This site blocks automated review retrieval (HTTP 403 Forbidden — anti-bot protection active). Export reviews manually and upload as a CSV instead.")
        if response.status_code == 429:
            logger.warning("Product URL rate-limited with 429: %s", url)
            raise ValueError("Too many automated requests to this site (HTTP 429 Rate Limited). Wait a few minutes and retry, or upload a CSV with your reviews.")
        if 500 <= response.status_code < 600:
            logger.warning("Product URL returned 5xx: %s status=%s", url, response.status_code)
            raise ValueError(f"The product page returned a server error (HTTP {response.status_code}). Verify the URL is correct or try again later.")
        if response.status_code != 200:
            logger.warning("Product URL returned unexpected status: %s status=%s", url, response.status_code)
            raise ValueError(f"Could not load the product page (HTTP {response.status_code}). Verify the URL or upload reviews as CSV.")

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning("HTTP error on %s: %s", url, exc)
            raise ValueError("Unable to load the product page. The site may require a browser session or block automated access.") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        logger.info("Step 4/5 — BeautifulSoup parse complete, beginning review extraction")
        reviews = self._extract_reviews(soup, parsed.netloc)
        logger.info("Step 5/5 — Review extraction yielded %s candidate reviews before cap", len(reviews))
        if not reviews:
            raise ValueError(
                "Unable to extract reviews from this product page. The site may use client-side dynamic rendering, require authentication, or intentionally block automated retrieval. Export reviews manually and upload as CSV."
            )

        return reviews[:max_reviews]

    def _extract_reviews(self, soup: BeautifulSoup, netloc: str) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []

        logger.info("JSON-LD scan: searching for <script type=application/ld+json> blocks")
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                payload = json.loads(script.string or "")
            except (TypeError, json.JSONDecodeError):
                continue

            discovered = self._collect_from_json_payload(payload)
            if discovered:
                logger.info("JSON-LD block matched: %s reviews appended", len(discovered))
                reviews.extend(discovered)

        if reviews:
            logger.info("Using JSON-LD reviews (%s found after dedupe)", len(self._dedupe(reviews)))
            return self._dedupe(reviews)

        if "amazon." in netloc:
            logger.info("Attempting Amazon-specific review extraction for %s", netloc)
            reviews = self._extract_reviews_from_amazon(soup)
            if reviews:
                logger.info("Amazon extractor produced %s reviews", len(reviews))
                return self._dedupe(reviews)
            logger.info("Amazon extractor returned no matches")

        if "flipkart." in netloc:
            logger.info("Attempting Flipkart-specific review extraction for %s", netloc)
            reviews = self._extract_reviews_from_flipkart(soup)
            if reviews:
                logger.info("Flipkart extractor produced %s reviews", len(reviews))
                return self._dedupe(reviews)
            logger.info("Flipkart extractor returned no matches")

        logger.info("Falling back to generic CSS selector crawl")
        candidates = []
        selectors = "[data-review-id], .review, .review-body, [itemprop='reviewBody'], [itemprop='review']"
        for tag in soup.select(selectors):
            text = self._clean_text(tag.get_text(" ", strip=True))
            if not text or len(text) < 20:
                continue
            candidates.append({"review_text": text, "rating": self._guess_rating(tag), "source": "scraped"})

        logger.info("Generic selectors matched %s candidate nodes with text >= 20 chars", len(candidates))
        if candidates:
            return self._dedupe(candidates)

        return []

    def _collect_from_json_payload(self, payload: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            reviews = payload.get("review") or payload.get("reviews")
            if isinstance(reviews, list):
                for review in reviews:
                    if isinstance(review, dict):
                        text = self._clean_text(review.get("reviewBody") or review.get("description") or review.get("text") or "")
                        if not text:
                            continue
                        rating = review.get("reviewRating") or review.get("rating") or {}
                        if isinstance(rating, dict):
                            rating_value = rating.get("ratingValue")
                        else:
                            rating_value = rating
                        results.append(
                            {
                                "review_text": text,
                                "rating": float(rating_value) if rating_value is not None else 3.0,
                                "source": "scraped",
                            }
                        )
            elif isinstance(payload.get("reviewBody"), str):
                text = self._clean_text(payload.get("reviewBody"))
                if text:
                    results.append({"review_text": text, "rating": 3.0, "source": "scraped"})
        return results

    def _extract_reviews_from_amazon(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        for review_node in soup.select('div[data-hook="review"]'):
            body_node = review_node.select_one('span[data-hook="review-body"]')
            rating_node = review_node.select_one('i[data-hook="review-star-rating"] span.a-icon-alt, i[data-hook="review-star-rating"]')
            text = self._clean_text(body_node.get_text(" ", strip=True)) if body_node else ""
            if not text or len(text) < 20:
                continue
            reviews.append(
                {
                    "review_text": text,
                    "rating": self._guess_rating(rating_node) if rating_node is not None else 3.0,
                    "source": "amazon",
                }
            )
        return reviews

    def _extract_reviews_from_flipkart(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        for review_node in soup.select('div[data-testid="review"] , div._16PBlm'):
            text_node = review_node.select_one('div[class*="_6K-7Co"], div[class*="qwjRop"], div')
            rating_node = review_node.select_one('div._3LWZlK')
            text = self._clean_text(text_node.get_text(" ", strip=True)) if text_node else self._clean_text(review_node.get_text(" ", strip=True))
            if not text or len(text) < 20:
                continue
            reviews.append(
                {
                    "review_text": text,
                    "rating": self._guess_rating(rating_node) if rating_node is not None else 3.0,
                    "source": "flipkart",
                }
            )
        return reviews

    def _guess_rating(self, tag: Any) -> float:
        text = tag.get_text(" ", strip=True)
        match = re.search(r"(\d(?:[.,]\d)?)\s*(?:/|out of)\s*5", text)
        if match:
            return float(match.group(1).replace(",", "."))
        return 3.0

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text

    def _dedupe(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique_items: list[dict[str, Any]] = []
        for item in items:
            review_text = (item.get("review_text") or "").strip()
            if not review_text:
                continue
            key = review_text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(
                {
                    "review_text": review_text,
                    "rating": float(item.get("rating", 3.0) or 3.0),
                    "source": item.get("source", "scraped"),
                }
            )
        return unique_items


def fetch_reviews_from_url(product_url: str, max_reviews: int = 20) -> list[dict[str, Any]]:
    """Convenience wrapper for URL-based review retrieval."""
    return ProductUrlAnalyzer().fetch_reviews(product_url, max_reviews=max_reviews)
