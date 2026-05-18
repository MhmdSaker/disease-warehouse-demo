"""Emit profile dicts as human-readable YAML with optional REVIEW comments.

The standard yaml.dump output is functional but ugly and loses ordering. This
emitter writes a fixed top-level key order (name, display_name, ..., columns,
hierarchies) and lets us inject inline ``# REVIEW: <reason>`` comments on
column entries whose decisions came back LOW-confidence.
"""

from __future__ import annotations

from typing import Any

from disease_warehouse.core.profile_inference import Decision, InferenceResult


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == float("inf"):
            return ".inf"
        return repr(value)
    s = str(value)
    if s == ".inf":
        return ".inf"
    if any(ch in s for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "'", '"', "%", "@", "`"]):
        return f'"{s}"'
    if s.strip() != s or "\n" in s or s == "":
        return f'"{s}"'
    return s


def _emit_mapping_inline(mapping: dict[Any, Any]) -> str:
    parts = []
    for k, v in mapping.items():
        parts.append(f"{_yaml_scalar(k)}: {_yaml_scalar(v)}")
    return "{" + ", ".join(parts) + "}"


def _emit_list_inline(value: list[Any]) -> str:
    return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"


def _column_review_comments(col_entry: dict, decisions: list[Decision]) -> list[str]:
    """Emit REVIEW comments for LOW-confidence decisions, EXCEPT when a
    later HIGH/MEDIUM decision for the same field has overridden them
    (e.g., LLM enrichment filled in what the deterministic pass left LOW).
    Dropped columns (``drop: true``) get no REVIEWs — their role/description
    are not used downstream.
    """
    if col_entry.get("drop"):
        return []
    name = col_entry["source"]
    # For each (column, field) pair, find the best (highest-confidence)
    # decision. Only the worst-case decisions remaining LOW deserve a REVIEW.
    conf_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    best_by_field: dict[str, Decision] = {}
    for d in decisions:
        if d.column != name:
            continue
        cur = best_by_field.get(d.field)
        if cur is None or conf_rank[d.confidence] < conf_rank[cur.confidence]:
            best_by_field[d.field] = d
    return [
        f"# REVIEW ({d.field}): {d.reason}"
        for d in best_by_field.values()
        if d.is_review()
    ]


def _emit_column_block(col_entry: dict, decisions: list[Decision]) -> list[str]:
    lines: list[str] = []
    reviews = _column_review_comments(col_entry, decisions)
    # Place REVIEW comments at the list-item indent level so they read as a
    # block-level note above the next column entry, not as fields of the
    # previous one.
    for review in reviews:
        lines.append(f"  {review}")
    lines.append(f"  - source: {_yaml_scalar(col_entry['source'])}")
    ordered_keys = ["fact_column", "type", "role", "drop", "domain", "mapping", "scale_factor", "imputation", "description"]
    for key in ordered_keys:
        if key not in col_entry:
            continue
        val = col_entry[key]
        if key == "mapping":
            lines.append(f"    mapping: {_emit_mapping_inline(val)}")
        elif key == "domain":
            lines.append(f"    domain: {_emit_list_inline(val)}")
        elif key == "imputation":
            inner = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in val.items())
            lines.append(f"    imputation: {{{inner}}}")
        elif key == "description":
            lines.append(f"    description: {_yaml_scalar(val)}")
        else:
            lines.append(f"    {key}: {_yaml_scalar(val)}")
    return lines


def _emit_hierarchy_block(h: dict) -> list[str]:
    lines = []
    lines.append(f"  - source_column: {_yaml_scalar(h['source_column'])}")
    lines.append(f"    new_column: {_yaml_scalar(h['new_column'])}")
    lines.append("    bands:")
    for band in h["bands"]:
        upper = band["upper_exclusive"]
        if isinstance(upper, str) and upper.lower() == ".inf":
            upper_str = ".inf"
        else:
            upper_str = _yaml_scalar(upper)
        lines.append(f'      - {{label: "{band["label"]}", upper_exclusive: {upper_str}}}')
    return lines


def emit_yaml(result: InferenceResult, include_decision_header: bool = True) -> str:
    profile = result.profile
    lines: list[str] = []

    if include_decision_header:
        # Effective review count: per (column, field), keep best confidence —
        # mirrors what the YAML body actually emits after LLM overrides. Skip
        # dropped columns since their REVIEWs are suppressed downstream too.
        conf_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        dropped = {c["source"] for c in profile["columns"] if c.get("drop")}
        best_by_pair: dict[tuple[str, str], Decision] = {}
        for d in result.decisions:
            if d.column in dropped:
                continue
            key = (d.column, d.field)
            cur = best_by_pair.get(key)
            if cur is None or conf_rank[d.confidence] < conf_rank[cur.confidence]:
                best_by_pair[key] = d
        review_count = sum(1 for d in best_by_pair.values() if d.is_review())
        lines.append("# ---------------------------------------------------------------------------")
        lines.append("# Auto-generated by disease_warehouse.core.profile_inference")
        lines.append(f"# Delimiter detected: {result.delimiter!r}")
        lines.append(f"# Decisions: {len(result.decisions)} total, {review_count} need review (search for '# REVIEW:')")
        lines.append("# ---------------------------------------------------------------------------")
        lines.append("")

    lines.append(f"name: {_yaml_scalar(profile['name'])}")
    lines.append(f"display_name: {_yaml_scalar(profile['display_name'])}")
    lines.append(f"domain: {_yaml_scalar(profile['domain'])}")
    desc = profile["description"]
    if "\n" in desc:
        lines.append("description: |")
        for ln in desc.splitlines():
            lines.append(f"  {ln}")
    else:
        lines.append(f"description: {_yaml_scalar(desc)}")
    lines.append("")

    lines.append("source:")
    lines.append(f"  path: {_yaml_scalar(profile['source']['path'])}")
    lines.append(f"  format: {_yaml_scalar(profile['source'].get('format', 'csv'))}")
    if "delimiter" in profile["source"]:
        lines.append(f"  delimiter: {_yaml_scalar(profile['source']['delimiter'])}")
    lines.append("")

    lines.append("target:")
    t = profile["target"]
    lines.append(f"  column: {_yaml_scalar(t['column'])}")
    lines.append(f"  rename_to: {_yaml_scalar(t.get('rename_to', 'label'))}")
    for key in ["positive_value", "negative_value", "positive_int", "negative_int"]:
        if key in t:
            lines.append(f"  {key}: {_yaml_scalar(t[key])}")
    lines.append("")

    lines.append("columns:")
    for col_entry in profile["columns"]:
        lines.extend(_emit_column_block(col_entry, result.decisions))
    lines.append("")

    lines.append("cleaning:")
    cleaning = profile["cleaning"]
    lines.append(f"  strip_whitespace: {_yaml_scalar(cleaning.get('strip_whitespace', True))}")
    if cleaning.get("drop_rows_where"):
        lines.append("  drop_rows_where:")
        for rule in cleaning["drop_rows_where"]:
            lines.append(f"    - column: {_yaml_scalar(rule['column'])}")
            for k in ["operator", "value", "ci", "note"]:
                if k in rule:
                    lines.append(f"      {k}: {_yaml_scalar(rule[k])}")
    else:
        lines.append("  drop_rows_where: []")
    lines.append("")

    lines.append("hierarchies:")
    if profile.get("hierarchies"):
        for h in profile["hierarchies"]:
            lines.extend(_emit_hierarchy_block(h))
    else:
        lines[-1] = "hierarchies: []"
    lines.append("")

    return "\n".join(lines)
