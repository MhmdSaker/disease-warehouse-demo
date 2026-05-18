"""Profile-driven cleaning pipeline.

Every transformation comes from the YAML profile — there are no per-disease
hardcoded branches in this module. Order of operations:

1. Load raw file with declared NA tokens.
2. Strip whitespace on string columns.
3. Apply ``drop_rows_where`` rules (e.g. remove sparse gender='Other').
4. Drop columns marked ``drop: true`` (e.g. source ``id``).
5. Build hierarchies (concept hierarchies / OLAP bands).
6. Run per-column imputation (e.g. BMI by age-bracket median).
7. Apply per-column value mappings (e.g. Yes/No -> 1/0).
8. Domain validation on nominal columns.
9. Target reconciliation against declared positive/negative labels.
10. Return cleaned dataframe + structured lineage dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from disease_warehouse.core.profile import (
    ColumnSpec,
    DatasetProfile,
    DropRule,
    HierarchySpec,
)


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def _apply_drop_rule(df: pd.DataFrame, rule: DropRule) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rule.column not in df.columns:
        return df, {"matched": 0, "note": rule.note, "skipped": True}

    series = df[rule.column]
    if rule.case_insensitive and pd.api.types.is_object_dtype(series):
        series = series.str.lower()
        target = str(rule.value).lower()
    else:
        target = rule.value

    if rule.operator == "equals":
        mask = series == target
    elif rule.operator == "in":
        if isinstance(target, (list, tuple, set)):
            mask = series.isin(list(target))
        else:
            mask = series.isin([target])
    elif rule.operator == "lt":
        mask = series < target
    elif rule.operator == "gt":
        mask = series > target
    else:
        raise ValueError(f"Unsupported drop-rule operator: {rule.operator!r}")

    matched = int(mask.sum())
    df = df.loc[~mask].copy().reset_index(drop=True)
    return df, {"matched": matched, "note": rule.note}


def _bracket_label(value: float, bands: list) -> str:
    for band in bands:
        if value < band.upper_exclusive:
            return band.label
    return bands[-1].label


def _build_hierarchy(df: pd.DataFrame, spec: HierarchySpec) -> pd.DataFrame:
    if spec.source_column not in df.columns:
        return df
    df = df.copy()
    # Coerce (not astype): silently turn unparseable strings like 'NaNN', '?',
    # or 'unknown' into NaN so hierarchy building doesn't crash on dirty data.
    numeric = pd.to_numeric(df[spec.source_column], errors="coerce")
    df[spec.new_column] = numeric.map(
        lambda v: _bracket_label(float(v), spec.bands) if pd.notna(v) else None
    )
    return df


def _apply_grouped_median(
    df: pd.DataFrame,
    col_spec: ColumnSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rule = col_spec.imputation or {}
    source_col = col_spec.source
    bands = rule["group_bands"]
    group_source = rule["group_source_column"]
    bracket_col = f"__{col_spec.source}_impute_bracket"

    def _label(value: float) -> str | None:
        if pd.isna(value):
            return None
        for band in bands:
            upper = band["upper_exclusive"]
            if isinstance(upper, str) and upper.lower() in {".inf", "inf"}:
                upper_val = float("inf")
            else:
                upper_val = float(upper)
            if float(value) < upper_val:
                return band["label"]
        return bands[-1]["label"]

    df = df.copy()
    df[bracket_col] = df[group_source].astype(float).map(_label)

    missing_before = int(df[source_col].isna().sum())
    medians = df.groupby(bracket_col, observed=True)[source_col].median()
    missing_by_bracket = (
        df.loc[df[source_col].isna()]
        .groupby(bracket_col, observed=True)
        .size()
    )

    df[source_col] = df.groupby(bracket_col, observed=True)[source_col].transform(
        lambda g: g.fillna(g.median())
    )

    if df[source_col].isna().any() and rule.get("fallback") == "global_median":
        df[source_col] = df[source_col].fillna(float(df[source_col].median()))

    df = df.drop(columns=[bracket_col])
    missing_after = int(df[source_col].isna().sum())

    table = []
    for band in bands:
        label = band["label"]
        median = medians.get(label, None)
        table.append(
            {
                "bracket": label,
                "median": None if median is None or pd.isna(median) else float(round(float(median), 3)),
                "missing": int(missing_by_bracket.get(label, 0)),
            }
        )

    return df, {
        "strategy": "grouped_median",
        "column": source_col,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "buckets": table,
    }


def _apply_value_mapping(df: pd.DataFrame, col_spec: ColumnSpec) -> pd.DataFrame:
    if not col_spec.mapping or col_spec.source not in df.columns:
        return df
    df = df.copy()
    df[col_spec.source] = df[col_spec.source].replace(col_spec.mapping)
    return df


def _validate_domain(df: pd.DataFrame, col_spec: ColumnSpec) -> dict[str, Any]:
    if not col_spec.domain or col_spec.source not in df.columns:
        return {}
    actual = set(df[col_spec.source].dropna().unique().tolist())
    expected = set(col_spec.domain)
    unexpected = actual - expected
    if unexpected:
        raise ValueError(
            f"Column {col_spec.source!r}: unexpected values {sorted(unexpected)} "
            f"not in declared domain {sorted(expected)}"
        )
    return {"checked_domain": sorted(expected), "unexpected": []}


def clean_dataset(profile: DatasetProfile, root_dir) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = root_dir / profile.source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source file for profile {profile.name!r} missing: {source_path}")

    raw = pd.read_csv(
        source_path,
        na_values=profile.na_values,
        keep_default_na=True,
        sep=profile.delimiter,
    )
    lineage: dict[str, Any] = {
        "profile": profile.name,
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "transformations": [],
        "drop_rules": [],
        "imputations": [],
        "domain_validations": [],
    }

    raw_target_counts: dict[str, int] = {}
    if profile.target.column in raw.columns:
        raw_target_counts = {
            str(k): int(v) for k, v in raw[profile.target.column].value_counts(dropna=False).items()
        }
    lineage["raw_target"] = raw_target_counts

    df = raw
    if profile.cleaning.strip_whitespace:
        df = _strip_strings(df)
        lineage["transformations"].append("stripped string whitespace")

    removed_total = 0
    removed_positive = 0
    target_col = profile.target.column
    positive_value = profile.target.positive_value

    for rule in profile.cleaning.drop_rows_where:
        # capture target leakage from rows we're about to remove
        if rule.column in df.columns and target_col in df.columns:
            if rule.case_insensitive and pd.api.types.is_object_dtype(df[rule.column]):
                pre_mask = df[rule.column].str.lower() == str(rule.value).lower()
            else:
                pre_mask = df[rule.column] == rule.value
            removed_positive += int(((pre_mask) & (df[target_col] == positive_value)).sum())
        df, drop_info = _apply_drop_rule(df, rule)
        removed_total += drop_info.get("matched", 0)
        lineage["drop_rules"].append({**drop_info, "rule": rule.__dict__})

    for col in profile.columns:
        if col.drop and col.source in df.columns:
            df = df.drop(columns=[col.source])
            lineage["transformations"].append(f"dropped column {col.source}")

    for col in profile.columns:
        if col.scale_factor is not None and col.source in df.columns:
            df = df.copy()
            df[col.source] = pd.to_numeric(df[col.source], errors="coerce") * float(col.scale_factor)
            lineage["transformations"].append(
                f"scaled {col.source} by {col.scale_factor}"
            )

    for col in profile.columns:
        if col.imputation:
            strategy = col.imputation.get("strategy")
            if strategy == "grouped_median":
                df, imp_info = _apply_grouped_median(df, col)
                lineage["imputations"].append(imp_info)
                lineage["transformations"].append(
                    f"imputed {col.source} via grouped_median"
                )
            elif strategy == "median":
                missing_before = int(df[col.source].isna().sum())
                df[col.source] = df[col.source].fillna(df[col.source].median())
                lineage["imputations"].append(
                    {"strategy": "median", "column": col.source,
                     "missing_before": missing_before, "missing_after": int(df[col.source].isna().sum())}
                )
            elif strategy == "constant":
                value = col.imputation.get("value")
                missing_before = int(df[col.source].isna().sum())
                df[col.source] = df[col.source].fillna(value)
                lineage["imputations"].append(
                    {"strategy": "constant", "column": col.source, "value": value,
                     "missing_before": missing_before, "missing_after": int(df[col.source].isna().sum())}
                )
            else:
                raise ValueError(f"Unsupported imputation strategy: {strategy!r}")

    for hierarchy in profile.hierarchies:
        df = _build_hierarchy(df, hierarchy)
        lineage["transformations"].append(
            f"built hierarchy {hierarchy.new_column} from {hierarchy.source_column}"
        )

    for col in profile.columns:
        df = _apply_value_mapping(df, col)

    for col in profile.columns:
        info = _validate_domain(df, col)
        if info:
            lineage["domain_validations"].append({"column": col.source, **info})

    if target_col in df.columns and profile.target.rename_to != target_col:
        df = df.rename(columns={target_col: profile.target.rename_to})

    label_col = profile.target.rename_to
    if label_col in df.columns:
        df[label_col] = df[label_col].replace(
            {profile.target.positive_value: profile.target.positive_int,
             profile.target.negative_value: profile.target.negative_int}
        ).astype(np.int8)

    if df.isna().sum().sum() > 0:
        bad = df.isna().sum()
        bad = bad[bad > 0]
        raise ValueError(
            f"Profile {profile.name}: missing values remain after cleaning: {bad.to_dict()}"
        )

    if label_col in df.columns:
        cleaned_counts = {
            "positive_1": int((df[label_col] == profile.target.positive_int).sum()),
            "negative_0": int((df[label_col] == profile.target.negative_int).sum()),
        }
    else:
        cleaned_counts = {}
    lineage["cleaned_rows"] = int(df.shape[0])
    lineage["cleaned_columns"] = int(df.shape[1])
    lineage["cleaned_missing_cells"] = 0
    lineage["cleaned_target"] = cleaned_counts
    lineage["removed_rows_total"] = removed_total
    lineage["removed_rows_positive"] = removed_positive

    return df, lineage
