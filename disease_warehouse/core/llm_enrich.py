"""LLM fallback for low-confidence column inference.

After the deterministic pass in :mod:`profile_inference` finishes, any column
still tagged LOW-confidence (typically: no keyword match and no value-range
fingerprint) is sent to Google Gemini in a single batched call to recover a
``role``, a one-sentence ``description``, and an optional ``mapping`` block.

Design rules
------------
* **Optional.** If ``google-generativeai`` isn't installed, ``GEMINI_API_KEY``
  isn't set, or the network call fails, the function is a no-op — the
  scaffold output remains identical to the pre-LLM behaviour.
* **Cached.** Suggestions are keyed by ``(csv_basename, column, sha1(samples))``
  and persisted to ``disease_warehouse/outputs/.llm_cache.json`` so repeated
  scaffold runs against the same CSV don't re-bill.
* **Batched.** A single request covers every LOW-confidence column from a CSV
  to keep latency and cost low.
* **Structured.** Gemini is asked for JSON; only fields that match our
  allowed-value lists are accepted, everything else is dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# Must match disease_warehouse.core.profile.ColumnSpec.__post_init__ allowed
# set. Identifiers are NOT a role — they get `drop: true` from the
# deterministic pass — so the LLM should never propose 'identifier'.
ALLOWED_ROLES = {
    "age", "gender", "symptom", "lab", "comorbidity",
    "lifestyle", "geography", "descriptor",
}

# Hard caps to keep prompts small and predictable.
MAX_SAMPLES_PER_COLUMN = 12
MAX_COLUMNS_PER_REQUEST = 40
DEFAULT_MODEL = "gemini-2.5-flash"

CACHE_FILENAME = ".llm_cache.json"


@dataclass
class LLMSuggestion:
    column: str
    role: str | None
    description: str | None
    mapping: dict | None
    confidence: str  # HIGH / MEDIUM / LOW from the model's self-assessment
    reason: str


def _gemini_available() -> tuple[bool, str]:
    if not os.environ.get("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY env var not set"
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        return False, "google-generativeai not installed (pip install google-generativeai)"
    return True, ""


def _sample_payload(series: pd.Series) -> dict[str, Any]:
    """Build a compact JSON-safe summary for one column."""
    n = int(len(series))
    nulls = int(series.isna().sum())
    s = series.dropna()
    distinct = int(s.nunique())

    sample_vals: list[Any]
    if distinct <= MAX_SAMPLES_PER_COLUMN:
        sample_vals = sorted(s.unique().tolist(), key=lambda v: str(v))
    else:
        sample_vals = s.sample(min(MAX_SAMPLES_PER_COLUMN, len(s)),
                               random_state=0).tolist()
    # JSON-safe
    sample_vals = [
        (None if pd.isna(v) else
         (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)))
        for v in sample_vals
    ]

    stats: dict[str, Any] = {
        "rows": n,
        "nulls": nulls,
        "distinct": distinct,
    }
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce").dropna()
        if not num.empty:
            stats["min"] = float(num.min())
            stats["max"] = float(num.max())
            stats["median"] = float(num.median())
    return {"samples": sample_vals, "stats": stats}


def _sample_hash(payload: dict) -> str:
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass  # caching is best-effort


def _build_prompt(csv_name: str, target_col: str, items: list[dict]) -> str:
    return (
        "You are a clinical-data column classifier. The columns below come from a "
        f"healthcare dataset named {csv_name!r} whose prediction target is {target_col!r}. "
        "For EACH input column, produce one object with these exact fields:\n"
        '  - "name": echo the input column name\n'
        f'  - "role": one of {sorted(ALLOWED_ROLES)}, or the literal string "null" if unknown\n'
        '  - "description": one short clinical sentence (under 25 words)\n'
        '  - "mapping": a JSON-encoded string like "{\\"Yes\\": 1, \\"No\\": 0}" '
        'ONLY if the column needs value normalisation; otherwise the empty string ""\n'
        '  - "confidence": "HIGH" | "MEDIUM" | "LOW" (your own assessment)\n'
        '  - "reason": brief justification (under 20 words)\n\n'
        'If you cannot judge a column, set role="null" and confidence="LOW".\n\n'
        "Input columns:\n"
        + json.dumps(items, indent=2, default=str)
    )


def _coerce_suggestion(raw: dict) -> LLMSuggestion | None:
    name = raw.get("name")
    if not isinstance(name, str):
        return None
    role = raw.get("role")
    if isinstance(role, str) and role.lower() in {"null", "none", ""}:
        role = None
    if role is not None and role not in ALLOWED_ROLES:
        role = None
    desc = raw.get("description")
    if not isinstance(desc, str) or not desc.strip():
        desc = None
    # mapping comes back either as a JSON-encoded string (new schema) or a dict
    # (legacy / cached). Normalise both shapes to {str: int} or None.
    raw_map = raw.get("mapping")
    mapping: dict | None = None
    if isinstance(raw_map, str) and raw_map.strip():
        try:
            raw_map = json.loads(raw_map)
        except json.JSONDecodeError:
            raw_map = None
    if isinstance(raw_map, dict) and raw_map:
        clean = {}
        for k, v in raw_map.items():
            try:
                clean[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        mapping = clean or None
    conf = str(raw.get("confidence", "LOW")).upper()
    if conf not in {"HIGH", "MEDIUM", "LOW"}:
        conf = "LOW"
    return LLMSuggestion(
        column=name,
        role=role,
        description=desc,
        mapping=mapping,
        confidence=conf,
        reason=str(raw.get("reason", ""))[:200],
    )


def enrich_low_confidence(
    df: pd.DataFrame,
    low_confidence_columns: list[str],
    *,
    target_col: str,
    csv_name: str,
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
) -> tuple[dict[str, LLMSuggestion], str]:
    """Return ``{column: LLMSuggestion}`` for the columns we could enrich.

    The second tuple element is a status string suitable for the CLI banner
    (e.g. ``"used gemini-2.0-flash for 4 columns (2 cached)"`` or
    ``"skipped: GEMINI_API_KEY not set"``).
    """
    if not low_confidence_columns:
        return {}, "no low-confidence columns to enrich"

    ok, reason = _gemini_available()
    if not ok:
        return {}, f"skipped: {reason}"

    cache_path = cache_dir / CACHE_FILENAME
    cache = _load_cache(cache_path)
    cache_ns = cache.setdefault(csv_name, {})

    items: list[dict] = []
    item_hashes: dict[str, str] = {}
    results: dict[str, LLMSuggestion] = {}
    cached_hits = 0

    for col in low_confidence_columns:
        if col not in df.columns:
            continue
        payload = _sample_payload(df[col])
        h = _sample_hash({"col": col, **payload})
        item_hashes[col] = h
        cached = cache_ns.get(h)
        if cached:
            suggestion = _coerce_suggestion(cached)
            if suggestion:
                results[col] = suggestion
                cached_hits += 1
                continue
        items.append({"name": col, **payload})

    if not items:
        return results, f"all {cached_hits} suggestions served from cache"

    # Chunk to keep prompts within reasonable size.
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)

    # Structured-output schema: forces Gemini to return an array of objects
    # with exactly these fields. Stricter than plain "application/json".
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name":        {"type": "string"},
                "role":        {"type": "string",
                                "enum": sorted(ALLOWED_ROLES) + ["null"]},
                "description": {"type": "string"},
                "mapping":     {"type": "string",
                                "description": "JSON-encoded {value:int} map, or empty string"},
                "confidence":  {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "reason":      {"type": "string"},
            },
            "required": ["name", "role", "description", "confidence", "reason"],
        },
    }

    api_calls = 0
    for start in range(0, len(items), MAX_COLUMNS_PER_REQUEST):
        chunk = items[start:start + MAX_COLUMNS_PER_REQUEST]
        prompt = _build_prompt(csv_name, target_col, chunk)
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": 0.0,
                },
            )
            api_calls += 1
            text = resp.text or "[]"
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                continue
            for raw in parsed:
                if not isinstance(raw, dict):
                    continue
                suggestion = _coerce_suggestion(raw)
                if suggestion is None:
                    continue
                results[suggestion.column] = suggestion
                h = item_hashes.get(suggestion.column)
                if h:
                    cache_ns[h] = {
                        "name": suggestion.column,
                        "role": suggestion.role,
                        "description": suggestion.description,
                        "mapping": suggestion.mapping,
                        "confidence": suggestion.confidence,
                        "reason": suggestion.reason,
                    }
        except Exception as exc:  # noqa: BLE001 — never let LLM kill the scaffold
            return results, f"LLM error after {api_calls} call(s): {exc}"

    _save_cache(cache_path, cache)
    status = (
        f"used {model_name} for {len(items)} column(s) across {api_calls} call(s)"
        + (f" ({cached_hits} cached)" if cached_hits else "")
    )
    return results, status
