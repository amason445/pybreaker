from __future__ import annotations

from uuid import UUID

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)


def resolve_columns(
    df: pd.DataFrame,
    data_type: str,
    columns: list[str] | None = None,
) -> list[str]:
    """Return explicit columns after validation or auto-detect matching columns."""
    normalized_type = data_type.strip().lower()

    if columns is not None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Unknown columns requested: {missing_list}")

        invalid = [column for column in columns if not column_matches_type(df[column], normalized_type)]
        if invalid:
            invalid_list = ", ".join(invalid)
            raise ValueError(f"Columns do not match data type '{normalized_type}': {invalid_list}")
        return columns

    return [column for column in df.columns if column_matches_type(df[column], normalized_type)]


def column_matches_type(series: pd.Series, data_type: str) -> bool:
    """Check whether a single pandas Series matches a requested breaker data type."""
    normalized_type = data_type.strip().lower()

    if normalized_type == "uuid":
        return is_uuid_series(series)
    if normalized_type == "int":
        return is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if normalized_type == "float":
        return is_float_dtype(series)
    if normalized_type == "numeric":
        return column_matches_type(series, "int") or column_matches_type(series, "float")
    if normalized_type == "string":
        return (
            (is_string_dtype(series) or is_object_dtype(series))
            and not is_uuid_series(series)
            and not is_date_series(series)
        )
    if normalized_type == "datetime":
        return is_datetime64_any_dtype(series)
    if normalized_type == "date":
        return is_date_series(series, require_datetime=False)

    raise ValueError(f"Unsupported data type: {data_type}")


def is_uuid_series(series: pd.Series) -> bool:
    """Heuristically detect UUID-like columns by sampling a few non-null values."""
    non_null_values = series.dropna()
    if non_null_values.empty:
        return False

    sample = non_null_values.head(5)
    for value in sample:
        try:
            UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return False
    return True


def is_date_series(series: pd.Series, require_datetime: bool = False) -> bool:
    """Heuristically detect Python date columns while excluding datetime64 columns."""
    if require_datetime:
        return is_datetime64_any_dtype(series)

    if is_datetime64_any_dtype(series):
        return False

    non_null_values = series.dropna()
    if non_null_values.empty:
        return False

    sample = non_null_values.head(5)
    return all(hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day") for value in sample)
