"""
Data Preprocessing Module.

Provides pure, reusable functions to clean, normalize, validate, and enrich
social media datasets for sentiment analysis and trend discovery.
"""

import re
from typing import Optional
import numpy as np
import pandas as pd

# Comprehensive standard English stop words list for keyword extraction
# Kept self-contained so the module does not crash if external corpuses are missing
STANDARD_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "today", "yesterday", "tomorrow", "day", "just", "get", "got", "also", "one"
}


def clean_text(text: Optional[str]) -> str:
    """
    Safely clean and normalize social media post text while keeping sentiment cues.

    Transformations:
    - Replaces None or empty with empty string
    - Removes URLs (http/https/www)
    - Removes user mentions (@username)
    - Replaces line breaks and multiple whitespaces with single space
    - Converts to lowercase
    - Preserves letters, numbers, punctuation, and emoji representations
    """
    if text is None or not isinstance(text, str):
        return ""

    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_hashtags(text: Optional[str]) -> list[str]:
    """
    Extract all hashtags from text and return them in canonical lowercase format.
    Example: 'Great game #MatchDay #Football' -> ['#matchday', '#football']
    """
    if not text or not isinstance(text, str):
        return []
    # Match hashtag pattern
    tags = re.findall(r"#([a-zA-Z0-9_]+)", text)
    return [f"#{tag.lower()}" for tag in tags]


def extract_keywords(
    text: Optional[str],
    min_length: int = 3,
    custom_stopwords: Optional[set[str]] = None
) -> list[str]:
    """
    Extract meaningful topical keywords from text by filtering out stop words,
    numbers, URLs, and punctuation.
    """
    if not text or not isinstance(text, str):
        return []

    stopwords = STANDARD_STOP_WORDS if custom_stopwords is None else custom_stopwords

    # Remove URLs, mentions, and hashtags
    clean = re.sub(r"https?://\S+|www\.\S+|@\w+|#\w+", " ", text.lower())
    # Match alphabetical words
    words = re.findall(r"\b[a-zA-Z]{" + str(min_length) + r",}\b", clean)
    return [w for w in words if w not in stopwords]


def remove_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows based on subset (default: 'post_id' or 'post_text')."""
    if df.empty:
        return df

    if subset is None:
        if "post_id" in df.columns:
            subset = ["post_id"]
        else:
            subset = ["post_text"]

    valid_subset = [col for col in subset if col in df.columns]
    if not valid_subset:
        return df.drop_duplicates()
    return df.drop_duplicates(subset=valid_subset).copy()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or sanitize missing values across social media columns."""
    if df.empty:
        return df

    df_clean = df.copy()

    # Text columns
    text_cols = ["post_text", "hashtags", "username", "topic", "platform"]
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("").astype(str)

    # Engagement columns
    numeric_cols = ["likes", "comments", "shares"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0).astype(int)

    return df_clean


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline for social media records:
    1. Removes duplicates
    2. Imputes missing values
    3. Validates and parses post_date to pandas datetime
    4. Enforces non-negative engagement values
    5. Computes total engagement (likes + comments + shares)
    6. Generates normalized cleaned_text while preserving original post_text
    7. Extracts structured hashtags and keywords
    """
    if df is None or df.empty:
        # Return empty DataFrame with expected structure
        empty_df = pd.DataFrame(columns=[
            "post_id", "platform", "username", "post_text", "hashtags",
            "likes", "comments", "shares", "post_date", "topic",
            "cleaned_text", "extracted_hashtags", "total_engagement"
        ])
        return empty_df

    df_clean = df.copy()

    # 1. Remove duplicate posts
    df_clean = remove_duplicates(df_clean)

    # 2. Handle missing values
    df_clean = handle_missing_values(df_clean)

    # 3. Validate and parse dates
    if "post_date" in df_clean.columns:
        df_clean["post_date"] = pd.to_datetime(df_clean["post_date"], errors="coerce")
        # If any dates failed parsing, fill with a valid fallback (e.g. current date or median date)
        if df_clean["post_date"].isna().any():
            median_date = df_clean["post_date"].dropna().median()
            fallback_date = median_date if pd.notna(median_date) else pd.Timestamp.now()
            df_clean["post_date"] = df_clean["post_date"].fillna(fallback_date)

    # 4. Enforce non-negative engagement values
    for col in ["likes", "comments", "shares"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # 5. Compute total engagement
    likes = df_clean["likes"] if "likes" in df_clean.columns else 0
    comments = df_clean["comments"] if "comments" in df_clean.columns else 0
    shares = df_clean["shares"] if "shares" in df_clean.columns else 0
    df_clean["total_engagement"] = likes + comments + shares

    # 6. Normalize text safely while preserving original post_text
    if "post_text" in df_clean.columns:
        df_clean["cleaned_text"] = df_clean["post_text"].apply(clean_text)
    else:
        df_clean["cleaned_text"] = ""

    # 7. Extract hashtags (combine hashtags column and text occurrences)
    def parse_all_hashtags(row):
        col_tags = extract_hashtags(str(row.get("hashtags", "")))
        text_tags = extract_hashtags(str(row.get("post_text", "")))
        # Deduplicate while preserving order
        seen = set()
        combined = []
        for tag in col_tags + text_tags:
            if tag not in seen:
                seen.add(tag)
                combined.append(tag)
        return combined

    df_clean["extracted_hashtags"] = df_clean.apply(parse_all_hashtags, axis=1)

    return df_clean
