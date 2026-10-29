"""
Data Loader Module.

Responsible for safely loading CSV datasets, validating schema compliance,
and providing access to the built-in social media dataset.
"""

from io import BytesIO, StringIO
from pathlib import Path
from typing import Union
import pandas as pd

# Standard required columns for the social media analytics pipeline
REQUIRED_COLUMNS = [
    "post_id",
    "platform",
    "username",
    "post_text",
    "hashtags",
    "likes",
    "comments",
    "shares",
    "post_date",
    "topic",
]

# Supported file encodings to try sequentially
ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]


def get_default_dataset_path() -> Path:
    """Return the absolute path to the default sample dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "social_media_posts.csv"


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that a loaded DataFrame contains all necessary schema columns.

    Returns:
        (is_valid, missing_columns): Tuple containing boolean status and list of missing column names.
    """
    if df is None or df.empty:
        return False, ["Dataset is completely empty."]

    normalized_cols = {str(c).strip().lower(): str(c) for c in df.columns}
    missing = [col for col in REQUIRED_COLUMNS if col.lower() not in normalized_cols]
    return len(missing) == 0, missing


def load_dataset(source: Union[str, Path, BytesIO, StringIO]) -> pd.DataFrame:
    """
    Load a CSV dataset safely from a file path, BytesIO buffer, or StringIO.

    Args:
        source: File path (str/Path) or uploaded file buffer.

    Returns:
        pd.DataFrame: Loaded and preliminary-validated DataFrame.

    Raises:
        FileNotFoundError: If the local path does not exist.
        ValueError: If file cannot be parsed or lacks required schema.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at: {path}")

    # Attempt reading with various encodings
    df = None
    last_error = None

    for enc in ENCODINGS:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc)
            break
        except Exception as e:
            last_error = e
            continue

    if df is None:
        raise ValueError(f"Failed to parse CSV file with supported encodings. Error: {last_error}")

    # Standardize column headers: strip whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize lowercase mapping if needed
    col_map = {}
    for col in df.columns:
        for req in REQUIRED_COLUMNS:
            if col.strip().lower() == req.lower():
                col_map[col] = req
    df.rename(columns=col_map, inplace=True)

    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise ValueError(
            f"Uploaded dataset is missing required columns: {', '.join(missing)}. "
            f"Expected schema: {', '.join(REQUIRED_COLUMNS)}"
        )

    return df


def get_sample_csv_template() -> str:
    """Return a minimal CSV template string for users to prepare custom datasets."""
    sample_rows = [
        "post_id,platform,username,post_text,hashtags,likes,comments,shares,post_date,topic",
        'POST_9001,X,@sample_user,"Loved the keynote presentation on AI! Fantastic insights.","#AI #Tech",150,12,35,2024-09-01 10:00:00,Technology',
        'POST_9002,Instagram,@creator_daily,"Terrible customer support wait time today. Frustrated!","#SupportFail #BadService",45,8,2,2024-09-01 11:30:00,Lifestyle',
        'POST_9003,Facebook,@news_update,"Central bank meets to discuss fiscal interest policies.","#Finance #News",85,4,10,2024-09-01 14:00:00,Business'
    ]
    return "\n".join(sample_rows)
