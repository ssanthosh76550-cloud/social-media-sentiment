"""
Unit tests for data loader and schema validation module.
"""

from io import StringIO
from pathlib import Path
import pytest
from src.data_loader import (
    get_default_dataset_path,
    get_sample_csv_template,
    load_dataset,
    validate_schema,
)


def test_default_dataset_exists_and_loads():
    """Verify built-in dataset file exists and contains at least 1000 records."""
    path = get_default_dataset_path()
    assert path.exists(), f"Default dataset missing at: {path}"
    df = load_dataset(path)
    assert len(df) >= 1000, f"Expected at least 1,000 posts, got {len(df)}"


def test_schema_validation_success():
    """Verify that a DataFrame with all required columns passes validation."""
    csv_data = get_sample_csv_template()
    df = load_dataset(StringIO(csv_data))
    is_valid, missing = validate_schema(df)
    assert is_valid is True
    assert missing == []


def test_schema_validation_missing_columns():
    """Verify that missing required columns are properly reported."""
    invalid_csv = "post_id,post_text\n1,Hello world"
    with pytest.raises(ValueError) as excinfo:
        load_dataset(StringIO(invalid_csv))
    assert "missing required columns" in str(excinfo.value)


def test_load_dataset_nonexistent_file():
    """Verify FileNotFoundError raised for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("non_existent_directory/fake_file.csv"))


def test_sample_csv_template_format():
    """Verify sample CSV template is non-empty and well-formed."""
    template = get_sample_csv_template()
    assert "post_id" in template
    assert "likes" in template
    assert "topic" in template
