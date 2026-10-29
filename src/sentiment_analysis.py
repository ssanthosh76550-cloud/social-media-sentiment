"""
Sentiment Analysis Module.

Utilizes NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to perform
rule-based sentiment scoring on social media posts.

Assigns discrete categories: Positive, Negative, Neutral, along with continuous
compound scores ranging from -1.0 (most negative) to +1.0 (most positive).
"""

import logging
from typing import Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)

# Global singleton holder for VADER analyzer
_VADER_ANALYZER = None
_FALLBACK_LEXICON_MODE = False

# Built-in lightweight fallback lexicon if NLTK data directory is unavailable offline
FALLBACK_POSITIVE_WORDS = {
    "love", "loved", "loving", "great", "awesome", "fantastic", "amazing", "breathtaking",
    "excellent", "good", "best", "triumph", "joy", "milestone", "excited", "happy", "honor",
    "honored", "wonderful", "masterpiece", "success", "successful", "thrilled", "beautiful",
    "profound", "legendary", "super", "breakthrough", "cheers", "stunning", "delight", "clean"
}

FALLBACK_NEGATIVE_WORDS = {
    "bad", "terrible", "worst", "fail", "broken", "horrible", "awful", "scam", "ruined",
    "hate", "hated", "disaster", "poor", "pain", "disappointed", "crisis", "outraged", "gouge",
    "greed", "unacceptable", "delay", "crash", "buggy", "frustrated", "sick", "shameful",
    "leak", "vulnerability", "loss", "plunge", "injury", "emergency", "flood", "stolen"
}


def initialize_vader() -> tuple[bool, str]:
    """
    Safely download and initialize the NLTK VADER sentiment analyzer.

    Returns:
        (success, message): Initialization status and informational notice.
    """
    global _VADER_ANALYZER, _FALLBACK_LEXICON_MODE

    if _VADER_ANALYZER is not None:
        return True, "VADER analyzer already initialized."

    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
            _FALLBACK_LEXICON_MODE = False
            return True, "NLTK VADER initialized successfully from local resources."
        except (LookupError, AttributeError):
            # Attempt automated download
            try:
                nltk.download("vader_lexicon", quiet=True)
                _VADER_ANALYZER = SentimentIntensityAnalyzer()
                _FALLBACK_LEXICON_MODE = False
                return True, "NLTK VADER lexicon downloaded and initialized successfully."
            except Exception as dl_err:
                logger.warning(
                    f"Automated NLTK download failed ({dl_err}). Switching to resilient fallback scoring."
                )
                _FALLBACK_LEXICON_MODE = True
                return False, (
                    "NLTK vader_lexicon was not found locally and download was blocked by network/sandbox. "
                    "Operating in high-compatibility fallback heuristic mode. "
                    "Run 'python -m nltk.downloader vader_lexicon' to activate full NLTK model."
                )
    except ImportError:
        _FALLBACK_LEXICON_MODE = True
        return False, (
            "NLTK is not installed. Using fallback heuristic dictionary. "
            "Install nltk with 'pip install nltk'."
        )


def _heuristic_sentiment(text: str) -> dict[str, Union[float, str]]:
    """Emergency heuristic sentiment fallback if VADER lexicon cannot be loaded."""
    words = set(text.lower().split())
    pos_count = len(words.intersection(FALLBACK_POSITIVE_WORDS))
    neg_count = len(words.intersection(FALLBACK_NEGATIVE_WORDS))

    if pos_count > neg_count:
        compound = min(0.15 * (pos_count - neg_count) + 0.2, 0.95)
        label = "Positive"
    elif neg_count > pos_count:
        compound = max(-0.15 * (neg_count - pos_count) - 0.2, -0.95)
        label = "Negative"
    else:
        compound = 0.0
        label = "Neutral"

    return {
        "compound": float(round(compound, 4)),
        "pos": float(round(min(pos_count * 0.1, 1.0), 3)),
        "neg": float(round(min(neg_count * 0.1, 1.0), 3)),
        "neu": float(round(1.0 - min((pos_count + neg_count) * 0.1, 0.8), 3)),
        "label": label
    }


def analyze_text_sentiment(
    text: Optional[str],
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05
) -> dict[str, Union[float, str]]:
    """
    Score sentiment of an individual text passage using VADER.

    Args:
        text: String content of the post.
        pos_threshold: Minimum compound score to qualify as Positive (standard: 0.05).
        neg_threshold: Maximum compound score to qualify as Negative (standard: -0.05).

    Returns:
        Dict with keys: 'compound', 'pos', 'neg', 'neu', 'label'
    """
    if text is None or not str(text).strip():
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
            "label": "Neutral"
        }

    global _VADER_ANALYZER
    if _VADER_ANALYZER is None and not _FALLBACK_LEXICON_MODE:
        initialize_vader()

    if _FALLBACK_LEXICON_MODE or _VADER_ANALYZER is None:
        return _heuristic_sentiment(str(text))

    try:
        scores = _VADER_ANALYZER.polarity_scores(str(text))
        compound = float(scores["compound"])

        if compound >= pos_threshold:
            label = "Positive"
        elif compound <= neg_threshold:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound": round(compound, 4),
            "pos": round(float(scores["pos"]), 4),
            "neg": round(float(scores["neg"]), 4),
            "neu": round(float(scores["neu"]), 4),
            "label": label
        }
    except Exception as e:
        logger.error(f"Error scoring sentiment for text: {e}")
        return _heuristic_sentiment(str(text))


def analyze_dataframe_sentiment(
    df: pd.DataFrame,
    text_column: str = "cleaned_text",
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05
) -> pd.DataFrame:
    """
    Process entire DataFrame and append sentiment columns.

    Appended Columns:
    - sentiment_compound: Normalized compound score (-1.0 to +1.0)
    - sentiment_pos: Positive proportion
    - sentiment_neg: Negative proportion
    - sentiment_neu: Neutral proportion
    - sentiment_label: 'Positive', 'Negative', or 'Neutral'
    """
    if df is None or df.empty:
        empty = df.copy() if df is not None else pd.DataFrame()
        for col in ["sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_label"]:
            if col not in empty.columns:
                empty[col] = pd.Series(dtype=object)
        return empty

    df_out = df.copy()

    # Determine source text column
    col_to_use = text_column if text_column in df_out.columns else "post_text"
    if col_to_use not in df_out.columns:
        # Fallback to empty strings
        df_out["sentiment_compound"] = 0.0
        df_out["sentiment_pos"] = 0.0
        df_out["sentiment_neg"] = 0.0
        df_out["sentiment_neu"] = 1.0
        df_out["sentiment_label"] = "Neutral"
        return df_out

    # Vectorized / apply scoring
    results = df_out[col_to_use].apply(
        lambda t: analyze_text_sentiment(t, pos_threshold, neg_threshold)
    )

    df_out["sentiment_compound"] = results.apply(lambda r: r["compound"])
    df_out["sentiment_pos"] = results.apply(lambda r: r["pos"])
    df_out["sentiment_neg"] = results.apply(lambda r: r["neg"])
    df_out["sentiment_neu"] = results.apply(lambda r: r["neu"])
    df_out["sentiment_label"] = results.apply(lambda r: r["label"])

    return df_out
