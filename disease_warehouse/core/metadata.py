"""Auto-profiling: features, descriptions, categorical/numerical attributes.

For every loaded dataset this module produces a uniform "column report" that
combines the **declared** profile metadata (description, role, type) with
**inferred** statistics (dtype, null %, distinct values, summary stats). The
result feeds two consumers:

1. ``metadata_data_dictionary`` rows in the warehouse.
2. A standalone ``data_dictionary.json`` per warehouse build.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from disease_warehouse.core.profile import ColumnSpec, DatasetProfile


CATEGORICAL_DISTINCT_CAP = 20
SAMPLE_VALUE_COUNT = 5


@dataclass
class ColumnMetadata:
    dataset: str
    source_column: str
    fact_column: str
    declared_type: str
    inferred_type: str            # "numeric" or "categorical" — data-driven
    attribute_type: str           # one of the warehouse types
    role: str | None
    description: str
    dtype: str
    null_count: int
    null_pct: float
    distinct_count: int
    sample_values: list[Any]
    numeric_stats: dict[str, float] = field(default_factory=dict)
    categorical_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _safe_native(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python so JSON / SQLite are happy."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            pass
    return value


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    counts = series.value_counts(dropna=True)
    top = counts.head(CATEGORICAL_DISTINCT_CAP)
    return {
        "top_values": {str(_safe_native(k)): int(v) for k, v in top.items()},
        "mode": str(_safe_native(counts.index[0])) if not counts.empty else None,
    }


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    cleaned = pd.to_numeric(series, errors="coerce")
    if cleaned.dropna().empty:
        return {}
    return {
        "min": _safe_native(cleaned.min()),
        "max": _safe_native(cleaned.max()),
        "mean": _safe_native(cleaned.mean()),
        "median": _safe_native(cleaned.median()),
        "std": _safe_native(cleaned.std()),
        "q25": _safe_native(cleaned.quantile(0.25)),
        "q75": _safe_native(cleaned.quantile(0.75)),
    }


def profile_column(
    df: pd.DataFrame,
    spec: ColumnSpec,
    dataset_name: str,
) -> ColumnMetadata:
    if spec.source not in df.columns:
        # Column was dropped or renamed already — still emit a placeholder so
        # the data dictionary records the declared intent.
        return ColumnMetadata(
            dataset=dataset_name,
            source_column=spec.source,
            fact_column=spec.fact_column,
            declared_type=spec.type,
            inferred_type="absent",
            attribute_type=spec.type,
            role=spec.role,
            description=spec.description,
            dtype="absent",
            null_count=0,
            null_pct=0.0,
            distinct_count=0,
            sample_values=[],
        )

    series = df[spec.source]
    inferred = _infer_kind(series)
    nulls = int(series.isna().sum())
    pct = round(nulls / max(len(series), 1) * 100, 4)
    distinct = int(series.nunique(dropna=True))
    samples = [_safe_native(v) for v in series.dropna().head(SAMPLE_VALUE_COUNT).tolist()]

    return ColumnMetadata(
        dataset=dataset_name,
        source_column=spec.source,
        fact_column=spec.fact_column,
        declared_type=spec.type,
        inferred_type=inferred,
        attribute_type=spec.type,
        role=spec.role,
        description=spec.description,
        dtype=str(series.dtype),
        null_count=nulls,
        null_pct=pct,
        distinct_count=distinct,
        sample_values=samples,
        numeric_stats=_numeric_stats(series) if inferred == "numeric" else {},
        categorical_stats=_categorical_stats(series) if inferred == "categorical" else {},
    )


def profile_dataset(df: pd.DataFrame, profile: DatasetProfile) -> list[ColumnMetadata]:
    return [profile_column(df, spec, profile.name) for spec in profile.columns]
