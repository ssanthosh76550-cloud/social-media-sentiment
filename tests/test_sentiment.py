"""
Unit tests for NLTK VADER sentiment analysis module.
"""

import pandas as pd
import pytest
from src.sentiment_analysis import (
    analyze_dataframe_sentiment,
    analyze_text_sentiment,
    initialize_vader,
)


def test_vader_initialization():
    """Verify that VADER initialization succeeds or safely activates fallback."""
    success, message = initialize_vader()
    assert isinstance(success, bool)
    assert isinstance(message, str)
    assert len(message) > 0


def test_positive_sentiment():
    """Verify classification of distinctly positive text."""
    text = "Absolutely phenomenal progress! Loving this breakthrough innovation. Brilliant work!"
    res = analyze_text_sentiment(text)
    assert res["label"] == "Positive"
    assert res["compound"] >= 0.05
    assert res["pos"] > 0


def test_negative_sentiment():
    """Verify classification of distinctly negative text."""
    text = "Worst customer service ever! The device is broken, defective, and a complete waste of money. Horrible!"
    res = analyze_text_sentiment(text)
    assert res["label"] == "Negative"
    assert res["compound"] <= -0.05
    assert res["neg"] > 0


def test_neutral_sentiment():
    """Verify classification of objective, factual text."""
    text = "The monthly meeting has been scheduled for Thursday at 10 AM in conference room B."
    res = analyze_text_sentiment(text)
    assert res["label"] == "Neutral"
    assert -0.05 <= res["compound"] <= 0.05


def test_empty_and_none_text():
    """Verify empty text, None, and whitespace return Neutral with zero compound score."""
    for empty_input in ["", None, "    ", "\n\t"]:
        res = analyze_text_sentiment(empty_input)
        assert res["label"] == "Neutral"
        assert res["compound"] == 0.0
        assert res["neu"] == 1.0


def test_custom_thresholds():
    """Verify custom positive and negative thresholds operate correctly."""
    # Score slightly above 0.05, but below 0.30
    mild_pos = "The product is good and decent."
    default_res = analyze_text_sentiment(mild_pos)
    strict_res = analyze_text_sentiment(mild_pos, pos_threshold=0.70)
    assert default_res["label"] == "Positive"
    assert strict_res["label"] == "Neutral"


def test_analyze_dataframe_sentiment():
    """Verify batch DataFrame sentiment enrichment adds expected columns."""
    df = pd.DataFrame([
        {"post_id": "1", "cleaned_text": "Incredible victory and fantastic performance!"},
        {"post_id": "2", "cleaned_text": "Terrible crash, catastrophic failure and loss."},
        {"post_id": "3", "cleaned_text": "Meeting protocol and scheduled timings."}
    ])

    scored = analyze_dataframe_sentiment(df, text_column="cleaned_text")
    for col in ["sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_label"]:
        assert col in scored.columns

    assert scored.loc[0, "sentiment_label"] == "Positive"
    assert scored.loc[1, "sentiment_label"] == "Negative"
    assert scored.loc[2, "sentiment_label"] == "Neutral"


def test_analyze_dataframe_sentiment_empty():
    """Verify empty DataFrame input returns empty DataFrame with schema preserved."""
    empty_df = pd.DataFrame()
    out = analyze_dataframe_sentiment(empty_df)
    assert out.empty
    assert "sentiment_label" in out.columns
    assert "sentiment_compound" in out.columns
