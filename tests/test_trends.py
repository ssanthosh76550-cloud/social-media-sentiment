"""
Unit tests for trend analysis and velocity-based ranking module.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest
from src.trend_analysis import (
    get_average_engagement_metrics,
    get_engagement_by_platform,
    get_engagement_by_sentiment,
    get_post_volume_over_time,
    get_sentiment_trends_over_time,
    get_top_hashtags,
    get_top_keywords,
    get_top_posts,
    get_topic_distribution,
    get_trending_keywords_velocity,
)


@pytest.fixture
def sample_analytics_data():
    """Create realistic sample DataFrame for trend tests."""
    base_time = datetime(2024, 8, 1, 10, 0, 0)
    data = []
    topics = ["Technology", "Sports", "Business"]
    platforms = ["X", "Instagram", "YouTube", "Facebook"]
    sentiments = ["Positive", "Negative", "Neutral"]

    for i in range(30):
        # Evenly spread over 15 days
        ts = base_time + timedelta(days=i % 15, hours=i)
        # In second half (i >= 15), inject surge of word "breakthrough"
        surge_text = "breakthrough innovation future" if i >= 15 else "standard update news"
        data.append({
            "post_id": f"POST_{i:03d}",
            "platform": platforms[i % len(platforms)],
            "username": f"@user_{i}",
            "post_text": f"Post content {i} {surge_text}",
            "cleaned_text": f"post content {surge_text}",
            "hashtags": "#tech #innovation" if i % 2 == 0 else "#sports #match",
            "extracted_hashtags": ["#tech", "#innovation"] if i % 2 == 0 else ["#sports", "#match"],
            "likes": 100 + i * 10,
            "comments": 10 + i,
            "shares": 5 + i * 2,
            "total_engagement": (100 + i * 10) + (10 + i) + (5 + i * 2),
            "post_date": ts,
            "topic": topics[i % len(topics)],
            "sentiment_label": sentiments[i % len(sentiments)],
            "sentiment_compound": 0.5 if sentiments[i % len(sentiments)] == "Positive" else (-0.5 if sentiments[i % len(sentiments)] == "Negative" else 0.0)
        })

    return pd.DataFrame(data)


def test_get_top_hashtags(sample_analytics_data):
    """Verify hashtag frequency calculation."""
    res = get_top_hashtags(sample_analytics_data, top_n=5)
    assert not res.empty
    assert "hashtag" in res.columns
    assert "count" in res.columns
    assert "#tech" in res["hashtag"].values or "#sports" in res["hashtag"].values


def test_get_top_keywords(sample_analytics_data):
    """Verify keyword extraction and ranking."""
    res = get_top_keywords(sample_analytics_data, top_n=5, min_length=4)
    assert not res.empty
    assert "keyword" in res.columns
    assert "frequency" in res.columns


def test_get_post_volume_over_time(sample_analytics_data):
    """Verify daily and weekly aggregation."""
    daily = get_post_volume_over_time(sample_analytics_data, freq="D")
    assert not daily.empty
    assert "date" in daily.columns
    assert "post_count" in daily.columns
    assert "total_engagement" in daily.columns

    weekly = get_post_volume_over_time(sample_analytics_data, freq="W")
    assert not weekly.empty
    assert len(weekly) <= len(daily)


def test_get_sentiment_trends_over_time(sample_analytics_data):
    """Verify temporal sentiment aggregation."""
    trends = get_sentiment_trends_over_time(sample_analytics_data, freq="D")
    assert not trends.empty
    for col in ["Positive", "Negative", "Neutral", "avg_compound"]:
        assert col in trends.columns


def test_get_topic_and_platform_distribution(sample_analytics_data):
    """Verify topic and platform summaries."""
    topics = get_topic_distribution(sample_analytics_data)
    assert not topics.empty
    assert "topic" in topics.columns
    assert "post_count" in topics.columns

    platforms = get_engagement_by_platform(sample_analytics_data)
    assert not platforms.empty
    assert "platform" in platforms.columns
    assert "avg_engagement" in platforms.columns


def test_get_top_posts(sample_analytics_data):
    """Verify retrieval and sorting of top engaging posts."""
    top_5 = get_top_posts(sample_analytics_data, top_n=5)
    assert len(top_5) == 5
    # Verify sorted descending
    assert top_5["total_engagement"].iloc[0] >= top_5["total_engagement"].iloc[-1]


def test_get_average_engagement_metrics(sample_analytics_data):
    """Verify engagement metrics and note formatting."""
    metrics = get_average_engagement_metrics(sample_analytics_data)
    assert metrics["avg_likes"] > 0
    assert metrics["avg_comments"] > 0
    assert metrics["avg_shares"] > 0
    assert "engagement_rate_note" in metrics


def test_get_trending_keywords_velocity(sample_analytics_data):
    """Verify velocity-based trending detection identifies surge in recent window."""
    vel_df = get_trending_keywords_velocity(sample_analytics_data, top_n=5)
    assert not vel_df.empty
    assert "keyword" in vel_df.columns
    assert "growth_pct" in vel_df.columns
    assert "trend_score" in vel_df.columns
    assert "trend_status" in vel_df.columns
    # 'breakthrough' was injected exclusively in the second half
    keywords = vel_df["keyword"].tolist()
    assert "breakthrough" in keywords
