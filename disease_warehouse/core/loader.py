"""Gold-layer loader: SQLite warehouse + per-table Parquet exports.

Given a list of profiles, runs the full build sequence:

    profile  ->  clean  ->  metadata  ->  schema (DDL)  ->  load (SQLite + Parquet)

Writes:
    outputs/gold.db                       SQLite warehouse (subject-oriented)
    outputs/schema.sql                    Generated DDL for review
    outputs/parquet/<table>.parquet       Columnar exports for analytics
    outputs/data_dictionary.json          Combined declared+inferred metadata
    outputs/lineage_report.json           Per-profile cleaning lineage
    outputs/cleaned/<profile>.csv         Cleaned silver-layer CSVs
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from disease_warehouse.core.cleaner import clean_dataset
from disease_warehouse.core.diagram import build_markdown_with_diagram, build_mermaid_erdiagram
from disease_warehouse.core.metadata import profile_dataset, ColumnMetadata
from disease_warehouse.core.profile import DatasetProfile
from disease_warehouse.core.schema_builder import build_schema_sql


AGE_BRACKET_ROWS = [
    ("0-17", 1, 0.0, 17.999),
    ("18-39", 2, 18.0, 39.999),
    ("40-59", 3, 40.0, 59.999),
    ("60-79", 4, 60.0, 79.999),
    ("80+", 5, 80.0, None),
]


def _age_bracket(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value < 18:
        return "0-17"
    if value < 40:
        return "18-39"
    if value < 60:
        return "40-59"
    if value < 80:
        return "60-79"
    return "80+"


def _json_or_none(value: Any) -> str | None:
    if not value:
        return None
    return json.dumps(value, default=str)


class WarehouseBuilder:
    def __init__(self, root_dir: Path, output_dir: Path, pipeline_version: str = "disease-warehouse-v1"):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.parquet_dir = output_dir / "parquet"
        self.cleaned_dir = output_dir / "cleaned"
        self.db_path = output_dir / "gold.db"
        self.schema_path = output_dir / "schema.sql"
        self.dictionary_path = output_dir / "data_dictionary.json"
        self.lineage_path = output_dir / "lineage_report.json"
        self.diagram_mmd_path = output_dir / "star_schema.mmd"
        self.diagram_md_path = output_dir / "star_schema.md"
        self.pipeline_version = pipeline_version

    def _ensure_dirs(self) -> None:
        for p in [self.output_dir, self.parquet_dir, self.cleaned_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def build(self, profiles: list[DatasetProfile]) -> dict[str, Any]:
        self._ensure_dirs()
        if not profiles:
            raise ValueError("No profiles registered; drop a YAML in disease_warehouse/profiles/.")

        schema_sql = build_schema_sql(profiles)
        self.schema_path.write_text(schema_sql, encoding="utf-8")

        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except PermissionError:
                # File held by another reader (DB Browser etc). Schema DDL contains
                # DROP IF EXISTS for every table, so we can just rebuild in place.
                pass

        cleaned: dict[str, pd.DataFrame] = {}
        lineages: dict[str, Any] = {}
        metadata_records: list[ColumnMetadata] = []
        raw_frames: dict[str, pd.DataFrame] = {}

        for prof in profiles:
            raw = pd.read_csv(
                self.root_dir / prof.source_path,
                na_values=prof.na_values,
                keep_default_na=True,
                sep=prof.delimiter,
            )
            raw_frames[prof.name] = raw
            cleaned_df, lineage = clean_dataset(prof, self.root_dir)
            cleaned[prof.name] = cleaned_df
            lineages[prof.name] = lineage
            metadata_records.extend(profile_dataset(raw, prof))

            cleaned_path = self.cleaned_dir / f"{prof.name}_cleaned.csv"
            cleaned_df.to_csv(cleaned_path, index=False)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(schema_sql)
            cursor = conn.cursor()

            snapshot_id = self._insert_snapshot(cursor)
            self._populate_lookups(cursor)
            dataset_ids = self._insert_datasets(cursor, profiles, snapshot_id)

            warehouse_counts: dict[str, int] = {}
            for prof in profiles:
                count = self._load_disease(
                    cursor,
                    profile=prof,
                    cleaned_df=cleaned[prof.name],
                    snapshot_id=snapshot_id,
                    dataset_id=dataset_ids[prof.name],
                )
                warehouse_counts[prof.fact_table] = count

            self._insert_data_dictionary(cursor, metadata_records, snapshot_id)
            self._insert_lineage(cursor, profiles, lineages, warehouse_counts, snapshot_id)
            self._insert_audits(cursor, profiles, lineages, snapshot_id)

            conn.commit()
        finally:
            conn.close()

        self._export_parquet(cleaned, profiles)
        self._write_data_dictionary(metadata_records, profiles)
        self.lineage_path.write_text(json.dumps(lineages, indent=2, default=str), encoding="utf-8")
        self._write_diagram(profiles, warehouse_counts)

        return {
            "snapshot_id": 1,
            "db_path": str(self.db_path),
            "profiles": [p.name for p in profiles],
            "warehouse_counts": warehouse_counts,
        }

    def _write_diagram(
        self,
        profiles: list[DatasetProfile],
        warehouse_counts: dict[str, int],
    ) -> None:
        mermaid = build_mermaid_erdiagram(profiles)
        self.diagram_mmd_path.write_text(mermaid, encoding="utf-8")

        # Read the snapshot timestamp out of the freshly-written DB so the
        # markdown page reflects exactly when this build ran.
        snapshot_ts: str | None = None
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT snapshot_timestamp FROM dim_etl_snapshot ORDER BY snapshot_id DESC LIMIT 1"
                ).fetchone()
                if row:
                    snapshot_ts = row[0]
            finally:
                conn.close()
        except sqlite3.Error:
            pass

        markdown = build_markdown_with_diagram(profiles, warehouse_counts, snapshot_ts)
        self.diagram_md_path.write_text(markdown, encoding="utf-8")

    def _insert_snapshot(self, cursor: sqlite3.Cursor) -> int:
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        cursor.execute(
            """
            INSERT INTO dim_etl_snapshot
                (snapshot_timestamp, pipeline_version, load_type, source_system, notes)
            VALUES (?, ?, 'full_refresh', 'disease-warehouse-engine', ?)
            """,
            (ts, self.pipeline_version, "Generic profile-driven build."),
        )
        return int(cursor.lastrowid)

    def _populate_lookups(self, cursor: sqlite3.Cursor) -> None:
        cursor.executemany(
            "INSERT INTO dim_age_bracket (age_bracket, sort_order, lower_age, upper_age) VALUES (?, ?, ?, ?)",
            AGE_BRACKET_ROWS,
        )
        cursor.executemany(
            "INSERT INTO dim_gender (gender_id, gender_code) VALUES (?, ?)",
            [(0, "Female"), (1, "Male"), (-1, "Unknown")],
        )

    def _insert_datasets(
        self,
        cursor: sqlite3.Cursor,
        profiles: list[DatasetProfile],
        snapshot_id: int,
    ) -> dict[str, int]:
        ids: dict[str, int] = {}
        for prof in profiles:
            cursor.execute(
                """
                INSERT INTO dim_dataset
                    (dataset_name, display_name, domain, description, profile_path,
                     source_path, fact_table, snapshot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prof.name,
                    prof.display_name,
                    prof.domain,
                    prof.description,
                    str(prof.profile_path),
                    str(prof.source_path),
                    prof.fact_table,
                    snapshot_id,
                ),
            )
            ids[prof.name] = int(cursor.lastrowid)
        return ids

    def _load_disease(
        self,
        cursor: sqlite3.Cursor,
        profile: DatasetProfile,
        cleaned_df: pd.DataFrame,
        snapshot_id: int,
        dataset_id: int,
    ) -> int:
        age_spec = profile.column_by_role("age")
        gender_spec = profile.column_by_role("gender")
        label_col = profile.target.rename_to

        first_id = cursor.execute(
            "SELECT COALESCE(MAX(patient_record_id), 0) + 1 FROM dim_patient_record"
        ).fetchone()[0]

        dim_rows = []
        for i, row in enumerate(cleaned_df.itertuples(index=False), start=1):
            row_dict = row._asdict()
            age = float(row_dict[age_spec.source]) if age_spec else None
            if gender_spec:
                gv = row_dict[gender_spec.source]
                try:
                    gender_id = int(gv)
                except (TypeError, ValueError):
                    gender_id = -1
                # Clamp to the dim_gender lookup set; everything else becomes Unknown.
                if gender_id not in (0, 1):
                    gender_id = -1
            else:
                gender_id = -1
            dim_rows.append((snapshot_id, dataset_id, i, age, gender_id, _age_bracket(age)))

        cursor.executemany(
            """
            INSERT INTO dim_patient_record
                (snapshot_id, dataset_id, source_record_num, age, gender_id, age_bracket)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            dim_rows,
        )
        record_ids = list(range(first_id, first_id + len(dim_rows)))

        fact_columns = profile.fact_columns()
        hierarchy_extra = [h.new_column for h in profile.hierarchies if h.new_column != "age_bracket"]

        column_names = (
            ["patient_record_id", "snapshot_id", "dataset_id"]
            + [c.fact_column for c in fact_columns]
            + hierarchy_extra
            + ["label"]
        )
        placeholders = ",".join("?" * len(column_names))
        insert_sql = (
            f"INSERT INTO {profile.fact_table} ({','.join(column_names)}) VALUES ({placeholders})"
        )

        fact_rows = []
        for record_id, (_, row) in zip(record_ids, cleaned_df.iterrows()):
            values: list[Any] = [record_id, snapshot_id, dataset_id]
            for spec in fact_columns:
                v = row[spec.source] if spec.source in row.index else row.get(spec.fact_column)
                values.append(_normalize_value(v, spec.type))
            for col in hierarchy_extra:
                values.append(row[col] if col in row.index else None)
            label = int(row[label_col]) if label_col in row.index else None
            values.append(label)
            fact_rows.append(tuple(values))

        cursor.executemany(insert_sql, fact_rows)
        return len(fact_rows)

    def _insert_data_dictionary(
        self,
        cursor: sqlite3.Cursor,
        metadata_records: list[ColumnMetadata],
        snapshot_id: int,
    ) -> None:
        rows = []
        for md in metadata_records:
            rows.append(
                (
                    snapshot_id,
                    md.dataset,
                    f"fact_{md.dataset}",
                    md.fact_column,
                    md.source_column,
                    md.declared_type,
                    md.inferred_type,
                    md.attribute_type,
                    md.role,
                    md.description,
                    md.dtype,
                    md.null_count,
                    md.null_pct,
                    md.distinct_count,
                    _json_or_none(md.sample_values),
                    _json_or_none(md.numeric_stats),
                    _json_or_none(md.categorical_stats),
                )
            )
        cursor.executemany(
            """
            INSERT INTO metadata_data_dictionary
                (snapshot_id, dataset_name, table_name, column_name, source_column,
                 declared_type, inferred_type, attribute_type, role, description,
                 dtype, null_count, null_pct, distinct_count, sample_values,
                 numeric_stats_json, categorical_stats_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_lineage(
        self,
        cursor: sqlite3.Cursor,
        profiles: list[DatasetProfile],
        lineages: dict[str, Any],
        warehouse_counts: dict[str, int],
        snapshot_id: int,
    ) -> None:
        rows = []
        for prof in profiles:
            li = lineages[prof.name]
            raw_target = li.get("raw_target", {})
            cleaned_target = li.get("cleaned_target", {})
            rows.append(
                (
                    snapshot_id, prof.name, "raw",
                    str(prof.source_path), str(prof.source_path),
                    int(li["raw_rows"]),
                    _raw_positive(prof, raw_target), _raw_negative(prof, raw_target),
                    int(li["raw_missing_cells"]), int(li["duplicate_rows"]),
                    "source CSV inspected without mutation",
                )
            )
            rows.append(
                (
                    snapshot_id, prof.name, "cleaned",
                    str(prof.source_path),
                    f"outputs/cleaned/{prof.name}_cleaned.csv",
                    int(li["cleaned_rows"]),
                    int(cleaned_target.get("positive_1", 0)),
                    int(cleaned_target.get("negative_0", 0)),
                    int(li["cleaned_missing_cells"]), int(li["duplicate_rows"]),
                    "; ".join(li["transformations"]) or "profile-driven cleaning",
                )
            )
            rows.append(
                (
                    snapshot_id, prof.name, "warehouse",
                    f"outputs/cleaned/{prof.name}_cleaned.csv",
                    prof.fact_table,
                    int(warehouse_counts[prof.fact_table]),
                    int(cleaned_target.get("positive_1", 0)),
                    int(cleaned_target.get("negative_0", 0)),
                    0, int(li["duplicate_rows"]),
                    f"loaded into {prof.fact_table} with FKs to dim_patient_record/dim_dataset/dim_etl_snapshot",
                )
            )
        cursor.executemany(
            """
            INSERT INTO metadata_lineage
                (snapshot_id, dataset_name, pipeline_stage, source_object, target_object,
                 row_count, positive_count, negative_count, missing_cells,
                 duplicate_rows, transformation_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_audits(
        self,
        cursor: sqlite3.Cursor,
        profiles: list[DatasetProfile],
        lineages: dict[str, Any],
        snapshot_id: int,
    ) -> None:
        rows = []
        for prof in profiles:
            li = lineages[prof.name]
            cleaned_target = li.get("cleaned_target", {})
            rows.append((snapshot_id, prof.name, "row_reconciliation", "raw_rows",
                         float(li["raw_rows"]), "PASS", "Raw row count captured"))
            rows.append((snapshot_id, prof.name, "row_reconciliation", "cleaned_rows",
                         float(li["cleaned_rows"]), "PASS", "Cleaned row count captured"))
            rows.append((snapshot_id, prof.name, "missing_value_audit", "cleaned_missing_cells",
                         float(li["cleaned_missing_cells"]), "PASS", "No nulls remain in cleaned dataframe"))
            rows.append((snapshot_id, prof.name, "duplicate_audit", "duplicate_rows",
                         float(li["duplicate_rows"]), "INFO", "Duplicate count from raw (reported only)"))
            if cleaned_target:
                rows.append((snapshot_id, prof.name, "target_reconciliation", "positive_count",
                             float(cleaned_target.get("positive_1", 0)), "PASS",
                             "Cleaned positives captured"))
                rows.append((snapshot_id, prof.name, "target_reconciliation", "negative_count",
                             float(cleaned_target.get("negative_0", 0)), "PASS",
                             "Cleaned negatives captured"))
            for imp in li.get("imputations", []):
                rows.append((snapshot_id, prof.name, "imputation",
                             f"{imp.get('column','?')}_missing_before",
                             float(imp.get("missing_before", 0)), "INFO",
                             f"{imp.get('strategy','?')} on {imp.get('column','?')} (before)"))
                rows.append((snapshot_id, prof.name, "imputation",
                             f"{imp.get('column','?')}_missing_after",
                             float(imp.get("missing_after", 0)), "PASS",
                             f"{imp.get('strategy','?')} on {imp.get('column','?')} (after)"))
        cursor.executemany(
            """
            INSERT INTO etl_audit_log
                (snapshot_id, dataset_name, audit_name, metric_name, metric_value, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _export_parquet(
        self,
        cleaned: dict[str, pd.DataFrame],
        profiles: list[DatasetProfile],
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            for table in [
                "dim_etl_snapshot",
                "dim_dataset",
                "dim_age_bracket",
                "dim_gender",
                "dim_patient_record",
                "metadata_data_dictionary",
                "metadata_lineage",
                "etl_audit_log",
            ]:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df.to_parquet(self.parquet_dir / f"{table}.parquet", index=False)
            for prof in profiles:
                df = pd.read_sql_query(f"SELECT * FROM {prof.fact_table}", conn)
                df.to_parquet(self.parquet_dir / f"{prof.fact_table}.parquet", index=False)
        finally:
            conn.close()

    def _write_data_dictionary(
        self,
        metadata_records: list[ColumnMetadata],
        profiles: list[DatasetProfile],
    ) -> None:
        by_dataset: dict[str, list[dict[str, Any]]] = {p.name: [] for p in profiles}
        for md in metadata_records:
            by_dataset.setdefault(md.dataset, []).append(md.to_dict())
        payload = {
            "datasets": [
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "domain": p.domain,
                    "description": p.description,
                    "source_path": str(p.source_path),
                    "fact_table": p.fact_table,
                    "columns": by_dataset.get(p.name, []),
                }
                for p in profiles
            ]
        }
        self.dictionary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _raw_positive(profile: DatasetProfile, raw_target: dict[str, int]) -> int:
    for key in (str(profile.target.positive_value), str(profile.target.positive_int), "1", "Positive"):
        if key in raw_target:
            return int(raw_target[key])
    return 0


def _raw_negative(profile: DatasetProfile, raw_target: dict[str, int]) -> int:
    for key in (str(profile.target.negative_value), str(profile.target.negative_int), "0", "Negative"):
        if key in raw_target:
            return int(raw_target[key])
    return 0


def _normalize_value(value: Any, attribute_type: str) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if attribute_type in {"binary", "identifier"}:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if attribute_type == "numeric":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value)
