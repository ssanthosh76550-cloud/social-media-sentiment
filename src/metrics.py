"""
Metrics Module.

Calculates high-level Key Performance Indicators (KPIs), summary statistics,
and engagement benchmarks for executive reporting.
"""

from typing import Any
import numpy as np
import pandas as pd


def format_number(val: float) -> str:
    """Format large numbers into human-readable compact strings (e.g. 1.2K, 3.4M)."""
    if pd.isna(val) or val is None:
        return "0"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{val / 1_000:.1f}K"
    if isinstance(val, int) or val.is_integer():
        return f"{int(val):,}"
    return f"{val:.1f}"


def calculate_kpi_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute comprehensive KPI dictionary for dashboard cards.

    Returns:
        dict containing total posts, sentiment percentages, engagement aggregates,
        and primary category benchmarks.
    """
    empty_result = {
        "total_posts": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "positive_pct": 0.0,
        "negative_pct": 0.0,
        "neutral_pct": 0.0,
        "most_common_sentiment": "N/A",
        "avg_sentiment_compound": 0.0,
        "total_likes": 0,
        "total_comments": 0,
        "total_shares": 0,
        "total_engagement": 0,
        "avg_engagement": 0.0,
        "avg_likes": 0.0,
        "avg_comments": 0.0,
        "avg_shares": 0.0,
        "most_active_platform": "N/A",
        "top_topic": "N/A"
    }

    if df is None or df.empty:
        return empty_result

    total_posts = len(df)
    if total_posts == 0:
        return empty_result

    # Sentiment Breakdown
    sentiment_col = df.get("sentiment_label", pd.Series(dtype=object))
    pos_count = int((sentiment_col == "Positive").sum())
    neg_count = int((sentiment_col == "Negative").sum())
    neu_count = int((sentiment_col == "Neutral").sum())

    pos_pct = round((pos_count / total_posts) * 100, 1)
    neg_pct = round((neg_count / total_posts) * 100, 1)
    neu_pct = round((neu_count / total_posts) * 100, 1)

    # Determine most common sentiment
    sentiment_counts = {"Positive": pos_count, "Negative": neg_count, "Neutral": neu_count}
    most_common = max(sentiment_counts, key=sentiment_counts.get) if total_posts > 0 else "N/A"

    # Compound score average
    avg_compound = round(float(df["sentiment_compound"].mean()), 3) if "sentiment_compound" in df.columns else 0.0

    # Engagement totals
    total_likes = int(df["likes"].sum()) if "likes" in df.columns else 0
    total_comments = int(df["comments"].sum()) if "comments" in df.columns else 0
    total_shares = int(df["shares"].sum()) if "shares" in df.columns else 0

    if "total_engagement" in df.columns:
        total_eng = int(df["total_engagement"].sum())
        avg_eng = round(float(df["total_engagement"].mean()), 1)
    else:
        total_eng = total_likes + total_comments + total_shares
        avg_eng = round(total_eng / total_posts, 1)

    avg_likes = round(total_likes / total_posts, 1)
    avg_comments = round(total_comments / total_posts, 1)
    avg_shares = round(total_shares / total_posts, 1)

    # Most active platform & topic
    platform_mode = df["platform"].mode() if "platform" in df.columns and not df["platform"].dropna().empty else []
    most_active_plat = str(platform_mode[0]) if len(platform_mode) > 0 else "N/A"

    topic_mode = df["topic"].mode() if "topic" in df.columns and not df["topic"].dropna().empty else []
    top_topic = str(topic_mode[0]) if len(topic_mode) > 0 else "N/A"

    return {
        "total_posts": total_posts,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": neu_pct,
        "most_common_sentiment": most_common,
        "avg_sentiment_compound": avg_compound,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "total_shares": total_shares,
        "total_engagement": total_eng,
        "avg_engagement": avg_eng,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "avg_shares": avg_shares,
        "most_active_platform": most_active_plat,
        "top_topic": top_topic
    }
