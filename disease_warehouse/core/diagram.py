"""Mermaid star-schema diagram emitter.

Emits a Mermaid ``erDiagram`` showing the current shape of the gold warehouse:
conformed dims at the centre, one ``fact_<name>`` per registered profile, FK
edges to ``dim_patient_record`` plus the metadata governance tables.

The diagram is regenerated on every build and written to:

    outputs/star_schema.mmd     raw Mermaid source
    outputs/star_schema.md      embedded in a markdown file (renders on GitHub)

Mermaid ``erDiagram`` syntax cheat-sheet:

    A ||--o{ B : "label"      # A has many B (1-to-many)
    A {
        type1 col1 PK
        type2 col2 FK
        type3 col3
    }
"""

from __future__ import annotations

from typing import Iterable

from disease_warehouse.core.profile import ColumnSpec, DatasetProfile
from disease_warehouse.core.schema_builder import SQL_TYPE_BY_ATTR


def _mermaid_type(attr_type: str) -> str:
    return {
        "REAL": "real",
        "INTEGER": "int",
        "TEXT": "text",
    }.get(SQL_TYPE_BY_ATTR.get(attr_type, "TEXT"), "text")


def _fact_columns_for_diagram(profile: DatasetProfile) -> list[tuple[str, str]]:
    cols: list[tuple[str, str]] = [
        ("int", "fact_id PK"),
        ("int", "patient_record_id FK"),
        ("int", "snapshot_id FK"),
        ("int", "dataset_id FK"),
    ]
    for spec in profile.fact_columns():
        cols.append((_mermaid_type(spec.type), spec.fact_column))
    for h in profile.hierarchies:
        if h.new_column == "age_bracket":
            continue  # lives on dim_patient_record
        cols.append(("text", h.new_column))
    cols.append(("int", "label"))
    return cols


# Mermaid identifiers can't contain hyphens or quotes in entity names. Most
# fact_<name> slugs are already safe, but we normalize defensively.
def _entity_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def build_mermaid_erdiagram(profiles: list[DatasetProfile]) -> str:
    lines: list[str] = []
    lines.append("erDiagram")

    # ── Relationships (conformed dims → patient_record → facts) ────────────
    lines.append('    dim_etl_snapshot   ||--o{ dim_patient_record : "snapshot"')
    lines.append('    dim_dataset        ||--o{ dim_patient_record : "cohort"')
    lines.append('    dim_age_bracket    ||--o{ dim_patient_record : "age band"')
    lines.append('    dim_gender         ||--o{ dim_patient_record : "gender"')
    lines.append('    dim_etl_snapshot   ||--o{ dim_dataset        : "registered in"')
    for prof in profiles:
        ent = _entity_name(prof.fact_table)
        lines.append(f'    dim_patient_record ||--o{{ {ent} : "patient"')
        lines.append(f'    dim_dataset        ||--o{{ {ent} : "cohort"')
        lines.append(f'    dim_etl_snapshot   ||--o{{ {ent} : "snapshot"')
    # Metadata governance edges (these tables reference dim_etl_snapshot)
    lines.append('    dim_etl_snapshot   ||--o{ metadata_data_dictionary : "documents"')
    lines.append('    dim_etl_snapshot   ||--o{ metadata_lineage         : "tracks"')
    lines.append('    dim_etl_snapshot   ||--o{ etl_audit_log            : "audits"')
    lines.append("")

    # ── Entity definitions ─────────────────────────────────────────────────
    lines.append("    dim_etl_snapshot {")
    lines.append("        int  snapshot_id PK")
    lines.append("        text snapshot_timestamp")
    lines.append("        text pipeline_version")
    lines.append("        text load_type")
    lines.append("        text source_system")
    lines.append("        text notes")
    lines.append("    }")
    lines.append("")

    lines.append("    dim_dataset {")
    lines.append("        int  dataset_id PK")
    lines.append("        text dataset_name")
    lines.append("        text display_name")
    lines.append("        text domain")
    lines.append("        text fact_table")
    lines.append("        int  snapshot_id FK")
    lines.append("    }")
    lines.append("")

    lines.append("    dim_age_bracket {")
    lines.append("        text age_bracket PK")
    lines.append("        int  sort_order")
    lines.append("        real lower_age")
    lines.append("        real upper_age")
    lines.append("    }")
    lines.append("")

    lines.append("    dim_gender {")
    lines.append("        int  gender_id PK")
    lines.append("        text gender_code")
    lines.append("    }")
    lines.append("")

    lines.append("    dim_patient_record {")
    lines.append("        int  patient_record_id PK")
    lines.append("        int  snapshot_id FK")
    lines.append("        int  dataset_id FK")
    lines.append("        int  source_record_num")
    lines.append("        real age")
    lines.append("        int  gender_id FK")
    lines.append("        text age_bracket FK")
    lines.append("    }")
    lines.append("")

    lines.append("    metadata_data_dictionary {")
    lines.append("        int  dictionary_id PK")
    lines.append("        int  snapshot_id FK")
    lines.append("        text dataset_name")
    lines.append("        text column_name")
    lines.append("        text declared_type")
    lines.append("        text inferred_type")
    lines.append("        text role")
    lines.append("    }")
    lines.append("")

    lines.append("    metadata_lineage {")
    lines.append("        int  lineage_id PK")
    lines.append("        int  snapshot_id FK")
    lines.append("        text dataset_name")
    lines.append("        text pipeline_stage")
    lines.append("        int  row_count")
    lines.append("        int  positive_count")
    lines.append("        int  negative_count")
    lines.append("    }")
    lines.append("")

    lines.append("    etl_audit_log {")
    lines.append("        int  audit_id PK")
    lines.append("        int  snapshot_id FK")
    lines.append("        text dataset_name")
    lines.append("        text audit_name")
    lines.append("        text metric_name")
    lines.append("        real metric_value")
    lines.append("        text status")
    lines.append("    }")
    lines.append("")

    # ── One block per disease fact ─────────────────────────────────────────
    for prof in profiles:
        ent = _entity_name(prof.fact_table)
        lines.append(f"    {ent} {{")
        for sql_type, col_name in _fact_columns_for_diagram(prof):
            lines.append(f"        {sql_type}  {col_name}")
        lines.append("    }")
        lines.append("")

    return "\n".join(lines)


def build_markdown_with_diagram(
    profiles: list[DatasetProfile],
    warehouse_counts: dict[str, int] | None = None,
    snapshot_timestamp: str | None = None,
) -> str:
    """Wrap the Mermaid diagram in a markdown page that renders on GitHub /
    in any Mermaid-aware viewer.
    """
    md: list[str] = []
    md.append("# Gold-layer star schema")
    md.append("")
    md.append(
        "Auto-regenerated by `disease_warehouse.core.diagram` on every build. "
        "Conformed dimensions sit at the centre; each registered profile gets "
        "its own `fact_<name>` table joined through `dim_patient_record`."
    )
    md.append("")
    if snapshot_timestamp:
        md.append(f"**Snapshot:** `{snapshot_timestamp}`")
        md.append("")

    md.append("## Registered profiles")
    md.append("")
    md.append("| dataset | domain | fact table | rows |")
    md.append("| --- | --- | --- | ---: |")
    for prof in profiles:
        n = (warehouse_counts or {}).get(prof.fact_table, "—")
        n_str = f"{n:,}" if isinstance(n, int) else str(n)
        md.append(f"| `{prof.name}` | {prof.domain} | `{prof.fact_table}` | {n_str} |")
    md.append("")

    md.append("## Diagram")
    md.append("")
    md.append("```mermaid")
    md.append(build_mermaid_erdiagram(profiles))
    md.append("```")
    md.append("")
    md.append(
        "_If your viewer doesn't render Mermaid, copy the block above into "
        "[mermaid.live](https://mermaid.live)._"
    )
    md.append("")
    return "\n".join(md)
