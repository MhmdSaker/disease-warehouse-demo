# Disease Warehouse — Profile-Driven Healthcare Data Warehouse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MhmdSaker/disease-warehouse-demo/blob/main/disease_warehouse_colab.ipynb)

A **dataset-agnostic** ETL engine that turns any healthcare CSV into a SQLite
star-schema warehouse — without writing Python. Each disease is onboarded as
a single YAML profile; the engine handles cleaning, hierarchy building,
conformed dimensions, per-disease facts, and Parquet exports.

## Quick start

**Click the "Open In Colab" badge above** — runs the entire pipeline end-to-end
in your browser. No local install required.

Then read [`INSTRUCTIONS_FOR_PROFESSOR.md`](INSTRUCTIONS_FOR_PROFESSOR.md) for
the 5-minute setup walkthrough.

## What's in the warehouse

7 pre-registered disease profiles, ~580k total rows across per-disease fact
tables sharing conformed dimensions:

| Disease | Rows | Domain |
|---|---|---|
| diabetes | 520 | metabolic |
| stroke | 5,109 | cardiovascular |
| heart_disease | 303 | cardiovascular |
| cardio | 70,000 | cardiovascular |
| cdc_diabetes | 253,680 | metabolic |
| cdc_heart_disease | 253,680 | cardiovascular |
| chronic_kidney_disease | 400 | general |

## How to onboard a new dataset

Three commands, no Python edits:

```bash
# 1. Auto-profile a raw CSV → produces a runnable YAML
python -m disease_warehouse scaffold datasets/your_file.csv --name your_slug

# 2. (Optional) skim the YAML and tweak # REVIEW comments
$EDITOR disease_warehouse/profiles/your_slug.yaml

# 3. Build into the warehouse
python -m disease_warehouse build --profile your_slug
```

The scaffold combines six inference layers, applied in order until a column
is resolved with high enough confidence:

1. **Statistical inference** (dtype, cardinality, identifier detection)
2. **Medical-keyword pattern matching** (~80 patterns covering hepatology,
   nephrology, cardiology, endocrinology terms)
3. **Numeric value-range fingerprinting** (BP, glucose, BMI, HR, cholesterol, age)
4. **Multilingual + unit-suffix aliases** (`edad`, `sexo`, `_mmhg`, `_mg_dl`, ...)
5. **Embedding role classifier** *(optional, free, offline)* — BGE-small with
   a margin-based acceptance filter. Resolves columns whose names are
   semantically clear but absent from the keyword dictionary, before any
   generative model is consulted. Requires `sentence-transformers`.
6. **Generative SLM / LLM fallback** for whatever is still LOW-confidence:
   - **Local SLM via ollama** (Phi-3.5, Qwen2.5, BioMistral, Meditron — picked
     by `--slm <preset>`). Runs fully offline once the model is pulled.
   - **Gemini 2.5 Flash** as cloud fallback when allowed. Requires
     `GEMINI_API_KEY` in your environment or `.env` file.

## SLM presets and privacy modes

```bash
python -m disease_warehouse scaffold datasets/your_file.csv --name your_slug \
    --slm small --privacy strict
```

| Preset | Backend | Model | Footprint | Notes |
|---|---|---|---|---|
| `off` | — | — | 0 | Deterministic layers only. |
| `embed-only` | sentence-transformers | BAAI/bge-small-en-v1.5 | ~130 MB | Role classification, no generative call. |
| `tiny` | ollama | `qwen2.5:1.5b` | ~1 GB | Low-end CPU / no-GPU laptops. |
| `small` *(default for `auto`)* | ollama | `phi3.5` | ~2.3 GB | Sweet spot — RTX 4060 or higher-end CPU. |
| `small-bio` | ollama | `biomistral:7b` | ~4.5 GB | Biomarker-heavy datasets (Parkinson voice, hepatology). |
| `mid` | ollama | `qwen2.5:7b` | ~4.5 GB | Broader multilingual coverage. |
| `mid-bio` | ollama | `meditron:7b` | ~4 GB | Local biomedical 7B. |
| `cloud` | google-generativeai | `gemini-2.5-flash` | network | Legacy behavior — cloud only. |
| `auto` | — | resolves at runtime | varies | Local if ollama reachable, else cloud if key set, else embed-only, else off. |

| Privacy | Effect on chain |
|---|---|
| `strict` | Local providers only — never calls Gemini, even if a key is set. |
| `balanced` *(default)* | Local first; Gemini fallback for whatever stays LOW. |
| `cloud-only` | Gemini only — legacy `--use-llm` behavior. |

Other useful flags on `scaffold`:

- `--slm-model <id>` — override the preset's default ollama tag (e.g. a custom
  `Modelfile`-derived model).
- `--slm-no-embed` — skip the embedding pre-pass even when sentence-transformers
  is installed (clean A/B benchmarks).
- `--use-llm` / `--no-llm` — back-compat aliases for `--slm cloud --privacy cloud-only`
  and `--slm off` respectively.

The legacy two-flag interface (`--use-llm` / `--no-llm`) keeps working unchanged
— existing scripts and the Colab notebook continue to run with no edits.

## Local install

```bash
pip install -r requirements.txt
cp .env.example .env  # paste your GEMINI_API_KEY here (optional)
python -m disease_warehouse list-profiles
python -m disease_warehouse build
```

To use the local-SLM path: install [ollama](https://ollama.com),
`ollama pull phi3.5` (or any preset above), then run `scaffold` normally —
the provider chain auto-detects the daemon. To use the embedding pre-pass:
`pip install sentence-transformers`.

Outputs land in `disease_warehouse/outputs/` — SQLite warehouse, Parquet
exports, data dictionary, lineage report, and star-schema markdown.

## Documentation

- [`INSTRUCTIONS_FOR_PROFESSOR.md`](INSTRUCTIONS_FOR_PROFESSOR.md) — 5-min Colab walkthrough
