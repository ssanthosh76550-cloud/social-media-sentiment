"""
Trend Analysis Module.

Provides analytical calculations for temporal volume, topic distributions,
engagement patterns across platforms and sentiments, and velocity-based trending keywords.
"""

from collections import Counter
import math
from typing import Optional
import numpy as np
import pandas as pd
from .preprocessing import extract_keywords


def get_top_hashtags(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute most frequent hashtags across posts.

    Returns:
        pd.DataFrame with columns ['hashtag', 'count', 'percentage']
    """
    if df is None or df.empty or "extracted_hashtags" not in df.columns:
        return pd.DataFrame(columns=["hashtag", "count", "percentage"])

    all_tags = []
    for tag_list in df["extracted_hashtags"]:
        if isinstance(tag_list, list):
            all_tags.extend(tag_list)
        elif isinstance(tag_list, str) and tag_list:
            all_tags.extend(tag_list.split())

    if not all_tags:
        return pd.DataFrame(columns=["hashtag", "count", "percentage"])

    counter = Counter(all_tags)
    total_tags = len(all_tags)
    top_items = counter.most_common(top_n)

    result = pd.DataFrame(top_items, columns=["hashtag", "count"])
    result["percentage"] = (result["count"] / total_tags * 100).round(2)
    return result


def get_top_keywords(
    df: pd.DataFrame,
    top_n: int = 15,
    min_length: int = 4
) -> pd.DataFrame:
    """
    Extract highest frequency topical keywords from cleaned text.

    Returns:
        pd.DataFrame with columns ['keyword', 'frequency', 'percentage']
    """
    if df is None or df.empty or "cleaned_text" not in df.columns:
        return pd.DataFrame(columns=["keyword", "frequency", "percentage"])

    all_words = []
    for text in df["cleaned_text"].dropna():
        all_words.extend(extract_keywords(str(text), min_length=min_length))

    if not all_words:
        return pd.DataFrame(columns=["keyword", "frequency", "percentage"])

    counter = Counter(all_words)
    total_words = len(all_words)
    top_items = counter.most_common(top_n)

    res = pd.DataFrame(top_items, columns=["keyword", "frequency"])
    res["percentage"] = (res["frequency"] / total_words * 100).round(2)
    return res


def get_post_volume_over_time(
    df: pd.DataFrame,
    freq: str = "D"
) -> pd.DataFrame:
    """
    Aggregate post volume and total engagement over regular time intervals.

    Args:
        df: Input DataFrame with 'post_date'.
        freq: Frequency alias ('D' for daily, 'W' or 'W-MON' for weekly).

    Returns:
        pd.DataFrame indexed by timestamp with post counts and aggregate engagement.
    """
    if df is None or df.empty or "post_date" not in df.columns:
        return pd.DataFrame(columns=["date", "post_count", "total_engagement", "avg_engagement"])

    temp = df.copy()
    temp["post_date"] = pd.to_datetime(temp["post_date"], errors="coerce")
    temp = temp.dropna(subset=["post_date"])

    if temp.empty:
        return pd.DataFrame(columns=["date", "post_count", "total_engagement", "avg_engagement"])

    if "total_engagement" not in temp.columns:
        temp["total_engagement"] = temp.get("likes", 0) + temp.get("comments", 0) + temp.get("shares", 0)

    # Standardize resample frequency string for pandas 2.x
    resample_freq = "W-MON" if freq.upper().startswith("W") else "D"

    grouped = temp.set_index("post_date").resample(resample_freq).agg(
        post_count=("post_id", "count"),
        total_engagement=("total_engagement", "sum"),
        avg_engagement=("total_engagement", "mean")
    ).reset_index()

    grouped.rename(columns={"post_date": "date"}, inplace=True)
    grouped["avg_engagement"] = grouped["avg_engagement"].fillna(0).round(1)
    return grouped


def get_sentiment_trends_over_time(
    df: pd.DataFrame,
    freq: str = "D"
) -> pd.DataFrame:
    """
    Compute daily or weekly sentiment label breakdowns and average compound scores.

    Returns:
        pd.DataFrame with columns ['date', 'Positive', 'Negative', 'Neutral', 'avg_compound']
    """
    if df is None or df.empty or "post_date" not in df.columns or "sentiment_label" not in df.columns:
        return pd.DataFrame(columns=["date", "Positive", "Negative", "Neutral", "avg_compound"])

    temp = df.copy()
    temp["post_date"] = pd.to_datetime(temp["post_date"], errors="coerce")
    temp = temp.dropna(subset=["post_date"])

    if temp.empty:
        return pd.DataFrame(columns=["date", "Positive", "Negative", "Neutral", "avg_compound"])

    resample_freq = "W-MON" if freq.upper().startswith("W") else "D"

    # Pivot counts of sentiment
    counts = temp.groupby([pd.Grouper(key="post_date", freq=resample_freq), "sentiment_label"]).size().unstack(fill_value=0)

    # Ensure all 3 labels exist
    for lbl in ["Positive", "Negative", "Neutral"]:
        if lbl not in counts.columns:
            counts[lbl] = 0

    # Average compound score
    if "sentiment_compound" in temp.columns:
        avg_compound = temp.groupby(pd.Grouper(key="post_date", freq=resample_freq))["sentiment_compound"].mean()
        counts["avg_compound"] = avg_compound.round(3)
    else:
        counts["avg_compound"] = 0.0

    counts = counts.reset_index().rename(columns={"post_date": "date"})
    return counts


def get_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate post count, engagement, and average sentiment compound per topic.
    """
    if df is None or df.empty or "topic" not in df.columns:
        return pd.DataFrame(columns=["topic", "post_count", "total_engagement", "avg_engagement", "avg_sentiment"])

    temp = df.copy()
    if "total_engagement" not in temp.columns:
        temp["total_engagement"] = temp.get("likes", 0) + temp.get("comments", 0) + temp.get("shares", 0)
    if "sentiment_compound" not in temp.columns:
        temp["sentiment_compound"] = 0.0

    summary = temp.groupby("topic").agg(
        post_count=("post_id", "count"),
        total_engagement=("total_engagement", "sum"),
        avg_engagement=("total_engagement", "mean"),
        avg_sentiment=("sentiment_compound", "mean")
    ).reset_index()

    summary["avg_engagement"] = summary["avg_engagement"].round(1)
    summary["avg_sentiment"] = summary["avg_sentiment"].round(3)
    return summary.sort_values(by="post_count", ascending=False)


def get_engagement_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean and median engagement metrics grouped by sentiment label.
    """
    if df is None or df.empty or "sentiment_label" not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        "likes": ["mean", "median", "sum"],
        "comments": ["mean", "median", "sum"],
        "shares": ["mean", "median", "sum"],
        "total_engagement": ["mean", "median", "sum"]
    }
    # Filter to existing columns
    valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    res = df.groupby("sentiment_label").agg(valid_agg)
    # Flatten multi-index columns
    res.columns = [f"{col}_{stat}" for col, stat in res.columns]
    return res.reset_index()


def get_engagement_by_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate platform-wise volume, average likes, comments, shares, and total engagement.
    """
    if df is None or df.empty or "platform" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    if "total_engagement" not in temp.columns:
        temp["total_engagement"] = temp.get("likes", 0) + temp.get("comments", 0) + temp.get("shares", 0)

    summary = temp.groupby("platform").agg(
        post_count=("post_id", "count"),
        avg_likes=("likes", "mean"),
        avg_comments=("comments", "mean"),
        avg_shares=("shares", "mean"),
        total_engagement=("total_engagement", "sum"),
        avg_engagement=("total_engagement", "mean")
    ).reset_index()

    for col in ["avg_likes", "avg_comments", "avg_shares", "avg_engagement"]:
        summary[col] = summary[col].round(1)

    return summary.sort_values(by="total_engagement", ascending=False)


def get_top_posts(
    df: pd.DataFrame,
    top_n: int = 10,
    sort_by: str = "total_engagement"
) -> pd.DataFrame:
    """
    Retrieve top N most engaging posts.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    metric = sort_by if sort_by in df.columns else "total_engagement"
    if metric not in df.columns:
        return df.head(top_n)

    cols = [
        col for col in [
            "post_id", "platform", "username", "post_text", "topic",
            "sentiment_label", "sentiment_compound", "likes", "comments",
            "shares", "total_engagement", "post_date"
        ] if col in df.columns
    ]

    return df.sort_values(by=metric, ascending=False).head(top_n)[cols]


def get_average_engagement_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive average engagement metrics per post.
    Provides clear notes regarding true Engagement Rate requirements.
    """
    if df is None or df.empty:
        return {
            "avg_likes": 0.0,
            "avg_comments": 0.0,
            "avg_shares": 0.0,
            "avg_total_engagement": 0.0,
            "engagement_rate_note": (
                "Standard Engagement Rate (ERR) requires post Impressions or Account Reach "
                "(ERR = Interactions / Impressions * 100). In public social exports where impressions "
                "are unavailable, we report Average Interactions per Post."
            )
        }

    return {
        "avg_likes": round(float(df["likes"].mean()), 1) if "likes" in df.columns else 0.0,
        "avg_comments": round(float(df["comments"].mean()), 1) if "comments" in df.columns else 0.0,
        "avg_shares": round(float(df["shares"].mean()), 1) if "shares" in df.columns else 0.0,
        "avg_total_engagement": round(float(df["total_engagement"].mean()), 1) if "total_engagement" in df.columns else 0.0,
        "engagement_rate_note": (
            "Standard Engagement Rate (ERR) requires post Impressions or Account Reach "
            "(ERR = Total Interactions / Impressions * 100). Because social media platform APIs "
            "restrict impression telemetry to private page administrators, this dashboard reports "
            "Interactions per Post as the primary engagement benchmark."
        )
    }


def get_trending_keywords_velocity(
    df: pd.DataFrame,
    top_n: int = 12,
    min_keyword_length: int = 4
) -> pd.DataFrame:
    """
    Discover true trending keywords by comparing recent vs. baseline time windows.

    Methodology:
    Unlike static raw frequency which merely reflects evergreen words, Trend Velocity evaluates
    acceleration. The timeline is partitioned into Baseline Period (earlier 50%) and Recent Period (latest 50%).
    
    Formula:
      Growth Rate (%) = ((Recent Count - Baseline Count) / max(Baseline Count, 1)) * 100
      Trend Score = Recent Count * log2(1 + max(Growth Rate, 0))
    
    This ensures that words gaining rapid acceleration in the recent period rank highest,
    while evergreen or declining keywords are deprioritized.
    """
    if df is None or df.empty or "cleaned_text" not in df.columns or "post_date" not in df.columns:
        return pd.DataFrame(columns=[
            "keyword", "recent_count", "baseline_count", "growth_pct", "trend_score", "trend_status"
        ])

    temp = df.copy()
    temp["post_date"] = pd.to_datetime(temp["post_date"], errors="coerce")
    temp = temp.dropna(subset=["post_date"]).sort_values("post_date")

    if len(temp) < 10:
        # Fallback to simple frequency if dataset is very small
        simple_top = get_top_keywords(temp, top_n=top_n, min_length=min_keyword_length)
        if simple_top.empty:
            return pd.DataFrame(columns=["keyword", "recent_count", "baseline_count", "growth_pct", "trend_score", "trend_status"])
        simple_top.rename(columns={"frequency": "recent_count"}, inplace=True)
        simple_top["baseline_count"] = 0
        simple_top["growth_pct"] = 100.0
        simple_top["trend_score"] = simple_top["recent_count"]
        simple_top["trend_status"] = "Emerging"
        return simple_top

    # Split into baseline (first half) and recent (second half)
    median_time = temp["post_date"].median()
    baseline_slice = temp[temp["post_date"] <= median_time]
    recent_slice = temp[temp["post_date"] > median_time]

    baseline_words = []
    for t in baseline_slice["cleaned_text"].dropna():
        baseline_words.extend(extract_keywords(str(t), min_length=min_keyword_length))

    recent_words = []
    for t in recent_slice["cleaned_text"].dropna():
        recent_words.extend(extract_keywords(str(t), min_length=min_keyword_length))

    base_counter = Counter(baseline_words)
    rec_counter = Counter(recent_words)

    # Candidate words that appear at least 2 times in recent window
    candidates = {w for w, c in rec_counter.items() if c >= 2}

    records = []
    for word in candidates:
        r_count = rec_counter[word]
        b_count = base_counter.get(word, 0)
        # Calculate percentage growth
        growth = ((r_count - b_count) / max(b_count, 1)) * 100.0
        # Calculate velocity score
        growth_factor = max(growth, 0.0)
        score = r_count * math.log2(2.0 + (growth_factor / 50.0))

        if growth > 50 and b_count == 0:
            status = "Breakout 🚀"
        elif growth > 25:
            status = "Surging 📈"
        elif growth >= 0:
            status = "Stable ➡️"
        else:
            status = "Cooling 📉"

        records.append({
            "keyword": word,
            "recent_count": r_count,
            "baseline_count": b_count,
            "growth_pct": round(growth, 1),
            "trend_score": round(score, 2),
            "trend_status": status
        })

    if not records:
        return pd.DataFrame(columns=[
            "keyword", "recent_count", "baseline_count", "growth_pct", "trend_score", "trend_status"
        ])

    trend_df = pd.DataFrame(records).sort_values(by="trend_score", ascending=False).head(top_n)
    return trend_df.reset_index(drop=True)
