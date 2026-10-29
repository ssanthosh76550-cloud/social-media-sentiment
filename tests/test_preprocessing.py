"""
Unit tests for data preprocessing and cleaning functions.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    clean_text,
    extract_hashtags,
    extract_keywords,
    handle_missing_values,
    remove_duplicates,
    validate_and_clean_data,
)


def test_clean_text_basic():
    """Verify URL and mention stripping, lowercase conversion, and whitespace normalization."""
    raw = "Check out this AI tool https://example.com/test @elonmusk! It is AMAZING!   "
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "@elonmusk" not in cleaned
    assert "amazing" in cleaned
    assert "  " not in cleaned


def test_clean_text_none_and_empty():
    """Verify clean_text does not raise on None, empty, or non-string inputs."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_extract_hashtags():
    """Verify hashtag extraction and lowercase normalization."""
    text = "Big win tonight! #Champions #Victory #matchday2024"
    tags = extract_hashtags(text)
    assert tags == ["#champions", "#victory", "#matchday2024"]

    assert extract_hashtags("") == []
    assert extract_hashtags(None) == []
    assert extract_hashtags("No hashtags here") == []


def test_extract_keywords():
    """Verify stop words and punctuation removal during keyword extraction."""
    text = "The artificial intelligence revolution will transform healthcare and modern engineering."
    keywords = extract_keywords(text, min_length=4)
    assert "artificial" in keywords
    assert "intelligence" in keywords
    assert "healthcare" in keywords
    assert "engineering" in keywords
    # Stop words should be absent
    assert "the" not in keywords
    assert "and" not in keywords


def test_remove_duplicates():
    """Verify duplicate removal based on post_id."""
    df = pd.DataFrame([
        {"post_id": "P1", "post_text": "First post"},
        {"post_id": "P1", "post_text": "Duplicate post"},
        {"post_id": "P2", "post_text": "Second post"}
    ])
    deduped = remove_duplicates(df, subset=["post_id"])
    assert len(deduped) == 2
    assert deduped["post_id"].tolist() == ["P1", "P2"]


def test_handle_missing_values():
    """Verify NaN values in text and numeric columns are imputed safely."""
    df = pd.DataFrame([
        {"post_text": None, "likes": None, "comments": "5", "shares": np.nan}
    ])
    sanitized = handle_missing_values(df)
    assert sanitized["post_text"].iloc[0] == ""
    assert sanitized["likes"].iloc[0] == 0
    assert sanitized["comments"].iloc[0] == 5
    assert sanitized["shares"].iloc[0] == 0


def test_validate_and_clean_data_pipeline():
    """Verify end-to-end cleaning pipeline handles dates, engagement clipping, and column additions."""
    raw_df = pd.DataFrame([
        {
            "post_id": "P001",
            "platform": "X",
            "username": "@tester",
            "post_text": "Excited about new release! #Innovation",
            "hashtags": "#Tech",
            "likes": -15,  # negative value should be clipped to 0
            "comments": 10,
            "shares": 5,
            "post_date": "2024-08-15 14:30:00",
            "topic": "Technology"
        },
        {
            "post_id": "P002",
            "platform": "Instagram",
            "username": "@tester2",
            "post_text": "Invalid date test",
            "hashtags": "",
            "likes": 50,
            "comments": 2,
            "shares": 1,
            "post_date": "NOT_A_VALID_DATE",
            "topic": "Lifestyle"
        }
    ])

    clean_df = validate_and_clean_data(raw_df)
    assert len(clean_df) == 2
    # Check non-negative engagement enforcement
    assert clean_df.loc[0, "likes"] == 0
    assert clean_df.loc[0, "total_engagement"] == 15  # 0 + 10 + 5
    # Check original text preservation and cleaned text addition
    assert clean_df.loc[0, "post_text"] == "Excited about new release! #Innovation"
    assert "excited about new release" in clean_df.loc[0, "cleaned_text"]
    # Check hashtags merged
    assert "#tech" in clean_df.loc[0, "extracted_hashtags"]
    assert "#innovation" in clean_df.loc[0, "extracted_hashtags"]
    # Check invalid date was coerced into valid timestamp
    assert pd.notna(clean_df.loc[1, "post_date"])


def test_validate_and_clean_empty():
    """Verify empty DataFrame does not crash pipeline and yields expected columns."""
    empty_df = pd.DataFrame()
    out = validate_and_clean_data(empty_df)
    assert out.empty
    assert "total_engagement" in out.columns
    assert "cleaned_text" in out.columns
