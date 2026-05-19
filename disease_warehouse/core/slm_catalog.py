"""Catalog of named SLM presets + auto-detection helpers.

A *preset* binds a friendly name (``small``, ``small-bio``, ...) to a concrete
``(backend, model, recommended_quant)`` triple. The provider chain in
:mod:`disease_warehouse.core.llm_enrich` looks up presets here so the CLI
surface stays small (``--slm small``) while still letting power users override
the underlying model id (``--slm-model my-tag:latest``).

Tiers (footprint figures are 4-bit GGUF unless noted):

    embed-only    BAAI/bge-small-en-v1.5         ~130 MB      role classification only
    tiny          qwen2.5:1.5b-instruct-q4_K_M   ~1.1 GB      low-end CPU
    small         phi3.5:mini                    ~2.5 GB      default — sweet spot
    small-bio     biomistral:7b                  ~4.5 GB      biomarker-heavy datasets
    mid           qwen2.5:7b-instruct-q4_K_M     ~4.5 GB      broader multilingual
    mid-bio       meditron:7b-q5_K_M             ~5.5 GB      highest local biomedical
    cloud         gemini-2.5-flash               network      online, low-privacy ok
    off           (no enrichment)                0            deterministic only
    auto          resolved at runtime            varies       see ``resolve_preset``

The ``auto`` resolver probes ollama at 127.0.0.1:11434 and falls back to
cloud (if ``GEMINI_API_KEY`` set) or ``off``.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))


@dataclass(frozen=True)
class Preset:
    name: str
    backend: str            # 'embedding' | 'ollama' | 'gemini' | 'noop'
    model: str              # backend-specific id (ollama tag, HF repo, etc.)
    description: str
    approx_size_gb: float


PRESETS: dict[str, Preset] = {
    "off": Preset(
        name="off",
        backend="noop",
        model="",
        description="No LLM enrichment — deterministic heuristics only.",
        approx_size_gb=0.0,
    ),
    "embed-only": Preset(
        name="embed-only",
        backend="embedding",
        model="BAAI/bge-small-en-v1.5",
        description="Sentence-transformers role classification only. Free, ~130 MB, CPU-fast.",
        approx_size_gb=0.13,
    ),
    "tiny": Preset(
        name="tiny",
        backend="ollama",
        model="qwen2.5:1.5b",
        description="Smallest workable generative SLM. Good on low-end CPU / no-GPU laptops.",
        approx_size_gb=1.0,
    ),
    "small": Preset(
        name="small",
        backend="ollama",
        model="phi3.5",
        description="Default generalist SLM (3.8B Q4). Strong JSON output, fast on RTX 4060.",
        approx_size_gb=2.3,
    ),
    "small-bio": Preset(
        name="small-bio",
        backend="ollama",
        model="biomistral:7b",
        description="Biomedical-tuned 7B. Best on biomarker-heavy datasets (Parkinson voice, hepatology).",
        approx_size_gb=4.5,
    ),
    "mid": Preset(
        name="mid",
        backend="ollama",
        model="qwen2.5:7b",
        description="Broader multilingual coverage. ~4.5 GB.",
        approx_size_gb=4.5,
    ),
    "mid-bio": Preset(
        name="mid-bio",
        backend="ollama",
        model="meditron:7b",
        description="Local biomedical 7B (Meditron). ALWAYS pin :7b — bare 'meditron' pulls the 70B variant.",
        approx_size_gb=4.0,
    ),
    "cloud": Preset(
        name="cloud",
        backend="gemini",
        model="gemini-2.5-flash",
        description="Google Gemini Flash. Requires GEMINI_API_KEY and network egress.",
        approx_size_gb=0.0,
    ),
}

# Valid CLI choices; 'auto' is a virtual preset resolved at runtime.
PRESET_CHOICES = ["auto", "off", "embed-only", "tiny", "small", "small-bio",
                  "mid", "mid-bio", "cloud"]

PRIVACY_CHOICES = ["strict", "balanced", "cloud-only"]


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"Unknown SLM preset {name!r}. Valid: {sorted(PRESETS)}")
    return PRESETS[name]


def ollama_reachable(timeout: float = 0.3) -> bool:
    """Cheap TCP probe — true if something is listening on the ollama port."""
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_preset(preset: str, privacy: str) -> str:
    """Map ``auto`` → a concrete preset based on what's installed/reachable.

    Order:
        1. If ollama is reachable, prefer ``small`` (with embed pre-pass).
        2. Else if Gemini key is set and privacy allows it, ``cloud``.
        3. Else ``embed-only`` if sentence-transformers is importable.
        4. Else ``off``.
    """
    if preset != "auto":
        return preset
    if ollama_reachable():
        return "small"
    if privacy != "strict" and has_gemini_key():
        return "cloud"
    if has_sentence_transformers():
        return "embed-only"
    return "off"


def chain_for(
    preset: str,
    privacy: str,
    *,
    disable_embedding: bool = False,
) -> list[str]:
    """Return the ordered list of backend ids to invoke for this (preset, privacy).

    Backends downstream of the first that fully resolves a column are not
    re-invoked for that column (they only see the residual LOW set).

    * ``strict`` never includes ``gemini``.
    * ``cloud-only`` is just ``[gemini]`` (legacy behavior).
    * Otherwise ``embedding`` is prepended when sentence-transformers is
      importable, so role classification gets a free first pass before any
      generative call. Pass ``disable_embedding=True`` to skip it — useful
      for clean A/B benchmarks (deterministic vs deterministic+SLM vs
      deterministic+embed+SLM).
    """
    resolved = resolve_preset(preset, privacy)
    chain: list[str] = []

    if privacy == "cloud-only":
        if has_gemini_key():
            chain = ["gemini"]
    elif resolved == "off":
        chain = []
    elif resolved == "cloud":
        if has_sentence_transformers():
            chain.append("embedding")
        if has_gemini_key():
            chain.append("gemini")
    elif resolved == "embed-only":
        if has_sentence_transformers():
            chain.append("embedding")
        if privacy != "strict" and has_gemini_key():
            chain.append("gemini")
    else:
        # generative ollama tier
        if has_sentence_transformers():
            chain.append("embedding")
        chain.append("ollama")
        if privacy != "strict" and has_gemini_key():
            chain.append("gemini")

    if disable_embedding:
        chain = [p for p in chain if p != "embedding"]
    return chain
