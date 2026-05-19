"""Provider-chain enrichment for LOW-confidence column inference.

After the deterministic pass in :mod:`profile_inference`, columns still tagged
LOW-confidence are passed through an ordered chain of providers. Each provider
fills only the fields the upstream chain left blank — never overrides a more
confident decision — and returns a ``LLMSuggestion`` per column it could
classify.

Providers (in catalog order):

    EmbeddingProvider   sentence-transformers role classification only
                        (BGE / PubMedBERT). Cheap and offline-friendly.
    OllamaProvider      local SLM via the ollama REST API
                        (phi3.5, qwen2.5, biomistral, meditron, ...).
    GeminiProvider      Google Gemini Flash, cloud fallback when allowed.

Privacy modes (selected from ``slm_catalog.PRIVACY_CHOICES``):

    strict      local providers only — never call Gemini.
    balanced    local first, Gemini for residual LOW if a key is set.
    cloud-only  Gemini only — legacy ``--use-llm`` behavior.

Caching: each provider keeps its own cache namespace under
``outputs/.llm_cache.json``, keyed by ``(csv, column, sha1(samples))`` so a
preset change doesn't poison prior results.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from disease_warehouse.core.slm_catalog import (
    OLLAMA_HOST,
    OLLAMA_PORT,
    PRESETS,
    chain_for,
    get_preset,
    ollama_reachable,
    resolve_preset,
)


# Must match disease_warehouse.core.profile.ColumnSpec.__post_init__ allowed
# set. Identifiers are NOT a role — they get `drop: true` from the
# deterministic pass — so no provider should ever propose 'identifier'.
ALLOWED_ROLES = {
    "age", "gender", "symptom", "lab", "comorbidity",
    "lifestyle", "geography", "descriptor",
}

# Hard caps to keep prompts small and predictable.
MAX_SAMPLES_PER_COLUMN = 12
MAX_COLUMNS_PER_REQUEST = 40
CACHE_FILENAME = ".llm_cache.json"

# Default ollama context window. The model's own default (often 128K) makes
# ollama reserve a massive KV cache up-front — ~50 GB on Phi-3.5-mini, which
# crashes any consumer laptop. Our prompts are 2-3 k tokens of structured JSON,
# and outputs are under 6 k tokens for a 40-column batch, so 8192 is generous.
# Override per call by setting DW_OLLAMA_NUM_CTX in the environment.
DEFAULT_OLLAMA_NUM_CTX = int(os.environ.get("DW_OLLAMA_NUM_CTX", "8192"))
DEFAULT_OLLAMA_NUM_PREDICT = int(os.environ.get("DW_OLLAMA_NUM_PREDICT", "6144"))

# Embedding-classifier acceptance rules. BGE-small produces dense embeddings
# where every healthcare-flavored prototype tends to score around 0.6 against
# every column, so a raw threshold isn't enough — short column names (``BS``,
# ``HR``, ``BP``) drift into the closest prototype regardless of meaning. We
# additionally require the top role to beat the runner-up by a clear margin,
# which makes the provider conservative: it only fills the role when it has
# real signal, and drops everything else through to the SLM / cloud chain.
EMBEDDING_ACCEPT_THRESHOLD = 0.55
EMBEDDING_MIN_MARGIN = 0.07


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class LLMSuggestion:
    column: str
    role: str | None
    description: str | None
    mapping: dict | None
    confidence: str           # HIGH / MEDIUM / LOW
    reason: str
    provider: str = ""        # which provider produced this — for telemetry


# ---------------------------------------------------------------------------
# Sample payload + cache plumbing (shared by every provider)
# ---------------------------------------------------------------------------

def _sample_payload(series: pd.Series) -> dict[str, Any]:
    n = int(len(series))
    nulls = int(series.isna().sum())
    s = series.dropna()
    distinct = int(s.nunique())

    sample_vals: list[Any]
    if distinct <= MAX_SAMPLES_PER_COLUMN:
        sample_vals = sorted(s.unique().tolist(), key=lambda v: str(v))
    else:
        sample_vals = s.sample(
            min(MAX_SAMPLES_PER_COLUMN, len(s)), random_state=0
        ).tolist()
    sample_vals = [
        (None if pd.isna(v) else
         (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)))
        for v in sample_vals
    ]

    stats: dict[str, Any] = {"rows": n, "nulls": nulls, "distinct": distinct}
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
        cache_path.write_text(
            json.dumps(cache, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        pass


def _coerce_suggestion(raw: dict, *, provider: str = "") -> LLMSuggestion | None:
    name = raw.get("name") or raw.get("column")
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
        provider=provider or str(raw.get("provider", "")),
    )


# ---------------------------------------------------------------------------
# Provider base + concrete providers
# ---------------------------------------------------------------------------

class BaseProvider:
    name: str = "base"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def enrich(
        self,
        df: pd.DataFrame,
        columns: list[str],
        *,
        target_col: str,
        csv_name: str,
        cache_ns: dict,
        model_id: str,
    ) -> tuple[dict[str, LLMSuggestion], int]:
        """Return ``({column: suggestion}, api_calls)``."""
        raise NotImplementedError


# ── Embedding provider ────────────────────────────────────────────────────

ROLE_PROTOTYPES = {
    "age": "patient age, age in years, years old, age category, age group, "
           "age bracket",
    "gender": "patient sex, gender, male or female, biological sex",
    "symptom": "clinical symptom reported by the patient, presenting sign, "
               "polyuria, polydipsia, chest pain, weakness, blurred vision, "
               "weight loss, itching, fatigue, paresis, stiffness, alopecia",
    "lab": "laboratory test result, blood test, vital sign, biomarker, "
           "blood pressure, cholesterol, glucose, hemoglobin, creatinine, "
           "bilirubin, ALT, AST, albumin, BMI, heart rate, voice biomarker, "
           "jitter, shimmer, dysphonia measure",
    "comorbidity": "comorbid disease history, pre-existing condition, prior "
                   "diagnosis, hypertension, diabetes, heart disease, "
                   "stroke history, kidney disease, anemia",
    "lifestyle": "lifestyle behavior, smoking status, alcohol consumption, "
                 "physical activity, exercise, diet, occupation",
    "geography": "residence location, urban or rural area, geographic region, "
                 "country, city, postal code",
    "descriptor": "demographic descriptor, socioeconomic status, education "
                  "level, marital status, income bracket, insurance coverage, "
                  "self-rated health",
}


class EmbeddingProvider(BaseProvider):
    """Cosine-similarity role classification with sentence-transformers.

    Only fills the ``role`` field; description and mapping are left for later
    providers. Confidence is MEDIUM when the top cosine similarity is above
    :data:`EMBEDDING_ACCEPT_THRESHOLD`, LOW otherwise (and the column drops
    through to the next provider).
    """

    name = "embedding"

    def __init__(self) -> None:
        self._model = None
        self._proto_vecs = None

    def available(self) -> tuple[bool, str]:
        try:
            import sentence_transformers  # noqa: F401
            return True, ""
        except ImportError:
            return False, "sentence-transformers not installed (pip install sentence-transformers)"

    def _load(self, model_id: str) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        prompts = list(ROLE_PROTOTYPES.values())
        self._proto_vecs = self._model.encode(
            prompts, normalize_embeddings=True, show_progress_bar=False,
        )
        self._proto_roles = list(ROLE_PROTOTYPES.keys())

    def enrich(
        self,
        df: pd.DataFrame,
        columns: list[str],
        *,
        target_col: str,
        csv_name: str,
        cache_ns: dict,
        model_id: str,
    ) -> tuple[dict[str, LLMSuggestion], int]:
        import numpy as np

        self._load(model_id)
        results: dict[str, LLMSuggestion] = {}

        # One forward pass for all columns. Combine the column name with a few
        # sample values so cryptic column names get a hint from the data.
        queries = []
        for col in columns:
            if col not in df.columns:
                queries.append(col)
                continue
            payload = _sample_payload(df[col])
            samples = ", ".join(str(v) for v in payload["samples"][:5])
            queries.append(f"column name: {col}; example values: {samples}")

        vecs = self._model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False,
        )
        sims = vecs @ self._proto_vecs.T  # (n_cols, n_roles)

        for col, row in zip(columns, sims):
            order = np.argsort(row)[::-1]
            best_idx = int(order[0])
            second_idx = int(order[1])
            best_score = float(row[best_idx])
            margin = best_score - float(row[second_idx])
            if best_score < EMBEDDING_ACCEPT_THRESHOLD:
                continue
            if margin < EMBEDDING_MIN_MARGIN:
                continue  # ambiguous; let next provider try
            role = self._proto_roles[best_idx]
            conf = "MEDIUM" if best_score < 0.72 else "HIGH"
            results[col] = LLMSuggestion(
                column=col,
                role=role,
                description=None,
                mapping=None,
                confidence=conf,
                reason=(f"embedding cosine={best_score:.2f} to {role!r} "
                        f"(margin={margin:.2f})"),
                provider=self.name,
            )
        return results, 0


# ── Ollama provider ──────────────────────────────────────────────────────

class OllamaProvider(BaseProvider):
    """Local SLM via the ollama REST API at 127.0.0.1:11434."""

    name = "ollama"

    def available(self) -> tuple[bool, str]:
        if not ollama_reachable():
            return False, f"ollama not reachable at {OLLAMA_HOST}:{OLLAMA_PORT} — start with `ollama serve`"
        return True, ""

    def _build_prompt(self, csv_name: str, target_col: str, items: list[dict]) -> str:
        # Same shape as the Gemini prompt, but a strict response format
        # because ollama's JSON mode is more permissive than Gemini's schema.
        return (
            "You are a clinical-data column classifier. The columns below come from a "
            f"healthcare dataset named {csv_name!r} whose prediction target is {target_col!r}. "
            "For EACH input column, produce one JSON object with these exact fields:\n"
            f'  - "name": echo the input column name\n'
            f'  - "role": one of {sorted(ALLOWED_ROLES)}, or the literal string "null" if unknown\n'
            '  - "description": one short clinical sentence (under 25 words)\n'
            '  - "mapping": a JSON-encoded string like "{\\"Yes\\": 1, \\"No\\": 0}" '
            'ONLY if the column needs value normalisation; otherwise the empty string ""\n'
            '  - "confidence": "HIGH" | "MEDIUM" | "LOW" (your own assessment)\n'
            '  - "reason": brief justification (under 20 words)\n\n'
            'Return a JSON object with a single key "columns" whose value is the array. '
            'If you cannot judge a column, set role="null" and confidence="LOW".\n\n'
            "Input columns:\n"
            + json.dumps(items, indent=2, default=str)
        )

    def _call_ollama(self, model: str, prompt: str, timeout: float = 300.0) -> str:
        url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                # Cap the context window so ollama doesn't reserve the model's
                # full 128K KV cache up-front (Phi-3.5 mini → ~50 GB without
                # this; only 3 GB with num_ctx=8192).
                "num_ctx": DEFAULT_OLLAMA_NUM_CTX,
                # Stop runaway generation — structured JSON output for a
                # 40-column batch is well under 6 K tokens.
                "num_predict": DEFAULT_OLLAMA_NUM_PREDICT,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")

    def enrich(
        self,
        df: pd.DataFrame,
        columns: list[str],
        *,
        target_col: str,
        csv_name: str,
        cache_ns: dict,
        model_id: str,
    ) -> tuple[dict[str, LLMSuggestion], int]:
        results: dict[str, LLMSuggestion] = {}
        if not columns:
            return results, 0

        items: list[dict] = []
        item_hashes: dict[str, str] = {}
        cached_hits = 0

        for col in columns:
            if col not in df.columns:
                continue
            payload = _sample_payload(df[col])
            h = _sample_hash({"col": col, **payload})
            item_hashes[col] = h
            cached = cache_ns.get(h)
            if cached:
                sug = _coerce_suggestion(cached, provider=self.name)
                if sug:
                    results[col] = sug
                    cached_hits += 1
                    continue
            items.append({"name": col, **payload})

        if not items:
            return results, 0

        api_calls = 0
        for start in range(0, len(items), MAX_COLUMNS_PER_REQUEST):
            chunk = items[start:start + MAX_COLUMNS_PER_REQUEST]
            prompt = self._build_prompt(csv_name, target_col, chunk)
            try:
                text = self._call_ollama(model_id, prompt)
                api_calls += 1
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # network/process failure — let the orchestrator fall through
                raise RuntimeError(f"ollama call failed: {exc}") from exc

            parsed = _safe_json_load(text)
            if isinstance(parsed, dict) and "columns" in parsed:
                parsed = parsed["columns"]
            if not isinstance(parsed, list):
                continue
            for raw in parsed:
                if not isinstance(raw, dict):
                    continue
                sug = _coerce_suggestion(raw, provider=self.name)
                if sug is None:
                    continue
                results[sug.column] = sug
                h = item_hashes.get(sug.column)
                if h:
                    cache_ns[h] = {
                        "name": sug.column,
                        "role": sug.role,
                        "description": sug.description,
                        "mapping": sug.mapping,
                        "confidence": sug.confidence,
                        "reason": sug.reason,
                    }
        return results, api_calls


def _safe_json_load(text: str) -> Any:
    """Tolerant JSON loader — ollama sometimes wraps the array in extra text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the outermost {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Try outermost [...]
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


# ── Gemini provider (extracted from the old single-provider flow) ────────

class GeminiProvider(BaseProvider):
    name = "gemini"

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("GEMINI_API_KEY"):
            return False, "GEMINI_API_KEY env var not set"
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            return False, "google-generativeai not installed (pip install google-generativeai)"
        return True, ""

    def _build_prompt(self, csv_name: str, target_col: str, items: list[dict]) -> str:
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

    def enrich(
        self,
        df: pd.DataFrame,
        columns: list[str],
        *,
        target_col: str,
        csv_name: str,
        cache_ns: dict,
        model_id: str,
    ) -> tuple[dict[str, LLMSuggestion], int]:
        import google.generativeai as genai

        results: dict[str, LLMSuggestion] = {}
        if not columns:
            return results, 0

        items: list[dict] = []
        item_hashes: dict[str, str] = {}
        cached_hits = 0
        for col in columns:
            if col not in df.columns:
                continue
            payload = _sample_payload(df[col])
            h = _sample_hash({"col": col, **payload})
            item_hashes[col] = h
            cached = cache_ns.get(h)
            if cached:
                sug = _coerce_suggestion(cached, provider=self.name)
                if sug:
                    results[col] = sug
                    cached_hits += 1
                    continue
            items.append({"name": col, **payload})

        if not items:
            return results, 0

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model_id)

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
            prompt = self._build_prompt(csv_name, target_col, chunk)
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
            parsed = _safe_json_load(text)
            if not isinstance(parsed, list):
                continue
            for raw in parsed:
                if not isinstance(raw, dict):
                    continue
                sug = _coerce_suggestion(raw, provider=self.name)
                if sug is None:
                    continue
                results[sug.column] = sug
                h = item_hashes.get(sug.column)
                if h:
                    cache_ns[h] = {
                        "name": sug.column,
                        "role": sug.role,
                        "description": sug.description,
                        "mapping": sug.mapping,
                        "confidence": sug.confidence,
                        "reason": sug.reason,
                    }
        return results, api_calls


PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "embedding": EmbeddingProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _resolve_model_id(preset_name: str, override: str | None) -> str:
    if override:
        return override
    return PRESETS[preset_name].model if preset_name in PRESETS else ""


def enrich_low_confidence(
    df: pd.DataFrame,
    low_confidence_columns: list[str],
    *,
    target_col: str,
    csv_name: str,
    cache_dir: Path,
    slm_preset: str = "auto",
    privacy_mode: str = "balanced",
    slm_model_override: str | None = None,
    disable_embedding: bool = False,
    # legacy alias (Gemini-only callers from older code paths)
    model_name: str | None = None,
) -> tuple[dict[str, LLMSuggestion], str]:
    """Run the configured provider chain over the LOW-confidence column set.

    Returns ``({column: best LLMSuggestion}, status_line)``. Status is a
    one-line CLI summary describing what actually ran.
    """
    if not low_confidence_columns:
        return {}, "no low-confidence columns to enrich"

    # Back-compat: callers that still pass ``model_name=...`` (the old Gemini
    # path) get treated as ``--slm cloud --privacy cloud-only``.
    if model_name and slm_preset == "auto":
        slm_preset = "cloud"
        privacy_mode = "cloud-only"

    chain_ids = chain_for(slm_preset, privacy_mode, disable_embedding=disable_embedding)
    resolved_preset = resolve_preset(slm_preset, privacy_mode)

    if not chain_ids:
        return {}, f"skipped: no providers available (preset={slm_preset}, privacy={privacy_mode})"

    cache_path = cache_dir / CACHE_FILENAME
    cache = _load_cache(cache_path)

    results: dict[str, LLMSuggestion] = {}
    status_parts: list[str] = []

    remaining = list(low_confidence_columns)

    for provider_id in chain_ids:
        if not remaining:
            break
        provider_cls = PROVIDER_CLASSES.get(provider_id)
        if provider_cls is None:
            continue
        provider = provider_cls()
        ok, reason = provider.available()
        if not ok:
            status_parts.append(f"{provider_id}-skipped({reason})")
            continue

        # Each provider picks its own model id:
        #   embedding -> always BGE-small (preset embed-only model)
        #   ollama    -> the preset's model (or override)
        #   gemini    -> 'cloud' preset's model (or override only for cloud)
        if provider_id == "embedding":
            model_id = PRESETS["embed-only"].model
        elif provider_id == "ollama":
            model_id = _resolve_model_id(resolved_preset, slm_model_override)
            if not model_id:
                model_id = PRESETS["small"].model
        elif provider_id == "gemini":
            model_id = (slm_model_override if resolved_preset == "cloud" else None) \
                or PRESETS["cloud"].model
        else:
            model_id = ""

        ns_key = f"{csv_name}::{provider_id}::{model_id}"
        cache_ns = cache.setdefault(ns_key, {})

        try:
            partial, api_calls = provider.enrich(
                df,
                remaining,
                target_col=target_col,
                csv_name=csv_name,
                cache_ns=cache_ns,
                model_id=model_id,
            )
        except Exception as exc:  # noqa: BLE001 — never let a provider kill scaffold
            status_parts.append(f"{provider_id}-error({type(exc).__name__}:{exc})")
            continue

        # Merge: keep the best confidence per column across providers.
        conf_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        for col, sug in partial.items():
            cur = results.get(col)
            if cur is None or conf_rank[sug.confidence] < conf_rank[cur.confidence]:
                results[col] = sug

        # Columns that came back with a usable role drop out of the residual.
        fully_resolved = {c for c, s in partial.items() if s.role is not None}
        remaining = [c for c in remaining if c not in fully_resolved]

        status_parts.append(
            f"{provider_id}({model_id}):{len(partial)} cols, {api_calls} call(s)"
        )

    _save_cache(cache_path, cache)
    status = "; ".join(status_parts) or "no enrichment performed"
    if remaining:
        status += f"; still LOW: {len(remaining)}"
    return results, status
