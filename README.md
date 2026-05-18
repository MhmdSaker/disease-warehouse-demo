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
python -m disease_warehouse scaffold datasets/your_file.csv --name your_slug --use-llm

# 2. (Optional) skim the YAML and tweak # REVIEW comments
$EDITOR disease_warehouse/profiles/your_slug.yaml

# 3. Build into the warehouse
python -m disease_warehouse build --profile your_slug
```

The scaffold combines five inference layers:
- Statistical inference (dtype, cardinality, identifier detection)
- Medical-keyword pattern matching (~80 patterns covering hepatology,
  nephrology, cardiology, endocrinology terms)
- Numeric value-range fingerprinting (BP, glucose, BMI, HR, cholesterol, age)
- Multilingual + unit-suffix aliases (`edad`, `sexo`, `_mmhg`, `_mg_dl`, ...)
- **Gemini 2.5 Flash fallback** for anything still LOW-confidence — requires
  `GEMINI_API_KEY` in your environment or `.env` file

## Local install

```bash
pip install -r requirements.txt
cp .env.example .env  # paste your GEMINI_API_KEY here
python -m disease_warehouse list-profiles
python -m disease_warehouse build
```

Outputs land in `disease_warehouse/outputs/` — SQLite warehouse, Parquet
exports, data dictionary, lineage report, and star-schema markdown.

## Documentation

- [`INSTRUCTIONS_FOR_PROFESSOR.md`](INSTRUCTIONS_FOR_PROFESSOR.md) — 5-min Colab walkthrough
