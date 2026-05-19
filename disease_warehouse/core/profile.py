"""Dataset profile schema and loader.

A profile is a YAML file under ``disease_warehouse/profiles/`` that declares
everything the engine needs to ingest, clean, model, and warehouse a single
disease dataset. The schema is intentionally explicit so that a clinician
or analyst can author one without writing Python.

Top-level keys
--------------
name             unique slug; becomes ``fact_<name>`` in the warehouse
display_name     human-friendly title
description      free-text dataset description (multi-line allowed)
domain           optional clinical grouping (e.g. metabolic, cardiovascular)
source           where to read the raw file from
target           target column metadata + class encoding
columns          per-column declarations (type, mapping, role, description)
cleaning         dataset-level cleaning steps
hierarchies      derived ordinal columns (concept hierarchies / OLAP bands)

Per-column entry
----------------
source           required; raw column name
fact_column      optional; warehouse column name (default: snake_case of source)
type             one of: numeric, binary, nominal, ordinal, identifier
role             optional conformed-dim role: age, gender, target, symptom, lab,
                 comorbidity, lifestyle, geography, descriptor
mapping          optional value mapping (e.g. {Yes: 1, No: 0})
domain           optional allowed value set for nominal columns
description      free-text column description
drop             if true, column is dropped during cleaning
imputation       optional inline imputation rule for this column
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class _ClinicalSafeLoader(yaml.SafeLoader):
    """SafeLoader that uses YAML 1.2 boolean semantics.

    PyYAML defaults to YAML 1.1, where ``yes``, ``no``, ``on``, ``off`` (and
    their cased variants) parse as bool. That silently turns clinical mappings
    like ``{Yes: 1, No: 0}`` into ``{True: 1, False: 0}`` — they then never
    match the actual ``'Yes'/'No'`` strings in the data, and every mapped cell
    falls through to NULL. We strip those four tokens from the bool resolver
    so only ``true``/``false`` keep their YAML 1.2 boolean meaning.
    """


_BOOL_TAG = "tag:yaml.org,2002:bool"
for _ch in ("y", "Y", "n", "N", "o", "O"):
    if _ch in _ClinicalSafeLoader.yaml_implicit_resolvers:
        _ClinicalSafeLoader.yaml_implicit_resolvers[_ch] = [
            (tag, regexp)
            for tag, regexp in _ClinicalSafeLoader.yaml_implicit_resolvers[_ch]
            if tag != _BOOL_TAG
        ]


VALID_ATTRIBUTE_TYPES = {"numeric", "binary", "nominal", "ordinal", "identifier"}
VALID_ROLES = {
    "age",
    "gender",
    "target",
    "symptom",
    "lab",
    "comorbidity",
    "lifestyle",
    "geography",
    "descriptor",
}


def _snake(name: str) -> str:
    """Normalize an arbitrary source column name into a safe SQL identifier.

    Replaces any character that's not [a-z0-9_] with an underscore, collapses
    runs of underscores, and strips leading/trailing underscores. Handles
    real-world column names like 'A/G Ratio', 'tumor-size', 'Body Mass Index'.
    """
    import re as _re
    slug = _re.sub(r"[^a-z0-9_]", "_", name.strip().lower())
    slug = _re.sub(r"_+", "_", slug).strip("_")
    return slug or "col"


@dataclass
class ColumnSpec:
    source: str
    type: str
    fact_column: str = ""
    role: str | None = None
    mapping: dict[Any, Any] | None = None
    domain: list[Any] | None = None
    description: str = ""
    drop: bool = False
    imputation: dict[str, Any] | None = None
    scale_factor: float | None = None  # multiply the column by this scalar before binning/mapping

    def __post_init__(self) -> None:
        if not self.fact_column:
            self.fact_column = _snake(self.source)
        if self.type not in VALID_ATTRIBUTE_TYPES:
            raise ValueError(
                f"Column {self.source!r}: invalid type {self.type!r}; "
                f"expected one of {sorted(VALID_ATTRIBUTE_TYPES)}"
            )
        if self.role is not None and self.role not in VALID_ROLES:
            raise ValueError(
                f"Column {self.source!r}: invalid role {self.role!r}; "
                f"expected one of {sorted(VALID_ROLES)}"
            )
        # Defensive: even with the clinical-safe loader, if a profile is loaded
        # by some other path with a bool-coerced mapping ({True: 1, False: 0}),
        # restore the Yes/No string keys so .replace() matches the source data.
        if self.mapping:
            restored: dict[Any, Any] = {}
            for k, v in self.mapping.items():
                if isinstance(k, bool):
                    k = "Yes" if k else "No"
                restored[k] = v
            self.mapping = restored


@dataclass
class TargetSpec:
    column: str
    positive_value: Any = 1
    negative_value: Any = 0
    positive_int: int = 1
    negative_int: int = 0
    rename_to: str = "label"


@dataclass
class HierarchyBand:
    label: str
    upper_exclusive: float


@dataclass
class HierarchySpec:
    source_column: str
    new_column: str
    bands: list[HierarchyBand]


@dataclass
class DropRule:
    column: str
    operator: str  # equals, in, lt, gt
    value: Any
    case_insensitive: bool = False
    note: str = ""


@dataclass
class CleaningSpec:
    strip_whitespace: bool = True
    drop_rows_where: list[DropRule] = field(default_factory=list)


@dataclass
class DatasetProfile:
    name: str
    display_name: str
    description: str
    domain: str
    source_path: Path
    source_format: str
    na_values: list[str]
    delimiter: str
    target: TargetSpec
    columns: list[ColumnSpec]
    cleaning: CleaningSpec
    hierarchies: list[HierarchySpec]
    profile_path: Path

    @property
    def fact_table(self) -> str:
        return f"fact_{self.name}"

    def column_by_source(self, source: str) -> ColumnSpec | None:
        for col in self.columns:
            if col.source == source:
                return col
        return None

    def column_by_role(self, role: str) -> ColumnSpec | None:
        for col in self.columns:
            if col.role == role:
                return col
        return None

    def fact_columns(self) -> list[ColumnSpec]:
        """Columns that belong in the per-disease fact table.

        Excludes dropped columns, the target (handled separately as ``label``),
        and conformed-dim roles age/gender which live on dim_patient_record.
        """
        out = []
        for col in self.columns:
            if col.drop:
                continue
            if col.source == self.target.column:
                continue
            if col.role in {"age", "gender"}:
                continue
            out.append(col)
        return out


def _parse_band(raw: dict[str, Any]) -> HierarchyBand:
    upper = raw["upper_exclusive"]
    if isinstance(upper, str) and upper.lower() in {".inf", "inf", "infinity"}:
        upper = float("inf")
    return HierarchyBand(label=str(raw["label"]), upper_exclusive=float(upper))


def _parse_drop_rule(raw: dict[str, Any]) -> DropRule:
    return DropRule(
        column=raw["column"],
        operator=raw.get("operator", "equals"),
        value=raw["value"],
        case_insensitive=bool(raw.get("ci", raw.get("case_insensitive", False))),
        note=raw.get("note", ""),
    )


def load_profile(path: str | Path) -> DatasetProfile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_ClinicalSafeLoader)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: profile must be a YAML mapping at the top level")

    name = data["name"]
    target_raw = data["target"]
    target = TargetSpec(
        column=target_raw["column"],
        positive_value=target_raw.get("positive_value", 1),
        negative_value=target_raw.get("negative_value", 0),
        positive_int=int(target_raw.get("positive_int", 1)),
        negative_int=int(target_raw.get("negative_int", 0)),
        rename_to=str(target_raw.get("rename_to", "label")),
    )

    columns = [ColumnSpec(**col) for col in data.get("columns", [])]

    cleaning_raw = data.get("cleaning", {}) or {}
    cleaning = CleaningSpec(
        strip_whitespace=bool(cleaning_raw.get("strip_whitespace", True)),
        drop_rows_where=[_parse_drop_rule(r) for r in cleaning_raw.get("drop_rows_where", [])],
    )

    hierarchies = [
        HierarchySpec(
            source_column=h["source_column"],
            new_column=h["new_column"],
            bands=[_parse_band(b) for b in h["bands"]],
        )
        for h in data.get("hierarchies", [])
    ]

    source_raw = data["source"]
    return DatasetProfile(
        name=name,
        display_name=str(data.get("display_name", name)),
        description=str(data.get("description", "")).strip(),
        domain=str(data.get("domain", "general")),
        source_path=Path(source_raw["path"]),
        source_format=str(source_raw.get("format", "csv")),
        na_values=list(source_raw.get("na_values", ["?", "", " ", "N/A", "n/a", "NA", "na", "null", "NULL", "None"])),
        delimiter=str(source_raw.get("delimiter", ",")),
        target=target,
        columns=columns,
        cleaning=cleaning,
        hierarchies=hierarchies,
        profile_path=path,
    )
