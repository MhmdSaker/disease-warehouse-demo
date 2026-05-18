# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A data-mining / data-warehousing project around healthcare datasets, with **two parallel pipelines**:

1. **Legacy pipeline** (`etl_pipeline/`, `warehouse/`) — hardcoded for two specific datasets (diabetes screening + stroke prediction). Drives the defense materials in `00_PRESENTATION_READY/` and `presentation_readiness/`, the notebooks in `eda/` and `models/`, and is verified by `final_check.py`. **Do not break this pipeline** when making changes; the defense package depends on it.

2. **Generic engine** (`disease_warehouse/`) — profile-driven, dataset-agnostic. Each disease is onboarded as a YAML file in `disease_warehouse/profiles/`. The engine auto-collects metadata, cleans, builds a star schema with conformed dimensions + per-disease facts, and emits both a SQLite gold warehouse and Parquet files. This is the path to use when adding any new disease/dataset.

Both pipelines run independently and don't share runtime code. The legacy one writes to `warehouse/health_warehouse.db`; the generic engine writes to `disease_warehouse/outputs/gold.db`.

## Commands

### Setup

```bash
pip install -r requirements.txt
```

Python ≥3.9. Repo path on Windows contains Arabic characters, so always invoke with `python -X utf8` to avoid `cp1252` `UnicodeEncodeError` when printing tool output.

### Generic engine (preferred for new work)

```bash
# Discover every YAML profile in disease_warehouse/profiles/
python -X utf8 -m disease_warehouse list-profiles

# Build the gold warehouse from every profile
python -X utf8 -m disease_warehouse build

# Build only one disease
python -X utf8 -m disease_warehouse build --profile diabetes

# Inspect auto-profiled metadata for a YAML without writing the DB
python -X utf8 -m disease_warehouse inspect disease_warehouse/profiles/stroke.yaml

# Auto-profile a CSV into a complete YAML (inference + medical heuristics)
python -X utf8 -m disease_warehouse scaffold datasets/new_disease.csv --name kidney
```

Outputs land in `disease_warehouse/outputs/`:

- `gold.db` — SQLite warehouse (conformed dims + per-disease facts)
- `schema.sql` — generated DDL (review this to see the star schema for current profiles)
- `parquet/<table>.parquet` — columnar exports for Power BI / pandas / Spark
- `data_dictionary.json` — combined declared + inferred column metadata
- `lineage_report.json` — per-profile cleaning lineage
- `cleaned/<profile>_cleaned.csv` — silver-layer cleaned CSVs

### Legacy pipeline (still required for defense)

```bash
python etl_pipeline/load_and_inspect.py     # Phase 0: inspect raw CSVs
python etl_pipeline/clean.py                # Phase 1A: clean diabetes+stroke
python etl_pipeline/warehouse.py            # Phase 1B: build SQLite warehouse
python reports/export_dashboard_data.py     # Generate Power BI CSVs
python final_check.py                       # Verify reconciliation, FKs, counts
python presentation_readiness/collect_presentation_evidence.py
```

Notebook regeneration (only if the `.ipynb` files need rebuilding from `.py` sources):

```bash
python eda/create_eda_notebook.py
python models/create_feature_engineering_notebook.py
python models/create_modeling_notebook.py
python warehouse/create_sql_notebook.py
```

There is **no `pytest` suite**. Verification of the legacy pipeline is performed by `final_check.py`, which exits non-zero on any failed reconciliation.

## How to add a new disease dataset (generic engine)

The whole point of `disease_warehouse/` is that you never edit Python to add a disease. Steps:

1. Drop the raw CSV into `datasets/` (or any path).
2. Create `disease_warehouse/profiles/<name>.yaml` declaring:
   - `name` — slug; becomes `fact_<name>` table.
   - `source.path` — relative to repo root.
   - `target` — column name, positive/negative values, what to rename it to (default `label`).
   - `columns` — one entry per source column with `type` (numeric/binary/nominal/ordinal/identifier), optional `role` (age/gender/symptom/lab/comorbidity/lifestyle/geography/descriptor), optional `mapping`, optional `description`, optional `drop: true`, optional `imputation`.
   - `cleaning.drop_rows_where` — declarative row removal (e.g. sparse minority categories).
   - `hierarchies` — derived ordinal columns (age bands, glucose bands, BMI bands, etc.); the engine builds them automatically after imputation.
3. Run `python -X utf8 -m disease_warehouse build`. The new disease is now in `dim_dataset`, has its own `fact_<name>` table, a `vw_<name>_full` join view, and appears in `vw_disease_summary`. Conformed dimensions are reused — no schema migration required.

### Auto-profiling (recommended for new datasets)

`python -X utf8 -m disease_warehouse scaffold <csv> [--name slug] [--target col]` runs the inference engine in `core/profile_inference.py` and emits a complete YAML profile, including:

- delimiter sniff (CSV / `;` / TSV / `|`)
- target detection (by name pattern: `class`, `label`, `target`, `<disease>_binary`, `num`, etc., or fallback to last low-cardinality column)
- target encoding inference (Yes/No, Positive/Negative, 1/0; multi-class targets get a binarization mapping like `{0:0, 1:1, 2:1, 3:1, 4:1}`)
- per-column type classification (identifier/binary/ordinal/nominal/numeric)
- role assignment from a curated medical keyword dictionary (`age`, `sex`/`gender`, `bmi`, `ap_hi`/`trestbps`/`systolic`, `chol`, `glucose`/`hba1c`, `cp`/`exang`, `smoke`/`alco`/`active`, `hypertension`/`heart_disease`/`stroke`, ...)
- value-mapping inference from data (Yes/No → 1/0, Male/Female → 1/0, Urban/Rural → 1/0, ...)
- standard clinical hierarchies attached automatically when matching columns exist: `age_bracket`, `bmi_band`, `bp_band` (AHA stages), `chol_band` (AHA tiers), `glucose_band`
- imputation suggestion (median for numeric, constant `"Unknown"` for categorical) when nulls are present
- domain validation whitelist for nominal/ordinal columns with ≤ 10 distinct values
- sanity checks that flag REVIEW: age values that look like days, gender encoded `{1, 2}` instead of `{0, 1}`, sparse minorities

Every decision is tagged HIGH / MEDIUM / LOW confidence and a report is printed when scaffolding. **LOW-confidence decisions are emitted as inline `# REVIEW: <reason>` comments in the YAML** — the file is still buildable as-is, but those lines may need editing. For the existing six profiles, auto-profiling against a fresh CSV regenerates a near-identical YAML and the resulting warehouse has the same row counts and label distributions as the hand-written versions.

Tested coverage: re-generating `heart_disease.yaml` from `heart_disease_cleveland.csv` produces zero LOW-confidence decisions; auto-cardio flags the days→years and ambiguous gender encoding correctly.

## Architecture: legacy pipeline

```
datasets/*.csv  →  etl_pipeline/clean.py  →  datasets/cleaned/*.csv
                                                  ↓
                                          etl_pipeline/warehouse.py
                                                  ↓
                                      warehouse/health_warehouse.db
                                      (uses warehouse/schema.sql)
                                                  ↓
                                eda/, models/, warehouse/03_sql_analysis.ipynb
                                                  ↓
                            reports/export_dashboard_data.py → Power BI CSVs
                                                  ↓
                                          final_check.py (verification)
```

Defining traits:
- **Hardcoded** column lists and cleaning rules in `etl_pipeline/clean.py` (DIABETES_SYMPTOM_COLS, STROKE_WORK_TYPES, etc.).
- **Two facts** (`fact_diabetes_screening`, `fact_stroke_risk`) with a single shared `dim_patient_record` that has a `source_dataset` discriminator column.
- **Grain rule**: `patient_record_id` is a *warehouse surrogate*, not a real cross-dataset patient identifier. Diabetes rows and stroke rows are NOT linked patients.
- **Governance objects**: `dim_etl_snapshot` (time-variant load version), `metadata_data_dictionary`, `metadata_lineage`, `etl_audit_log`, plus views `vw_data_quality_metrics` and `vw_lineage_reconciliation` — these are presented during defense as evidence of warehouse governance.
- **Class imbalance reality**: stroke positives are 4.87% of cleaned rows; accuracy is misleading. Models in `models/04_modeling.ipynb` use class-weight / resampling.
- **Documented data decisions**: BMI imputed by age-bracket median (not global median); one `gender='Other'` stroke row removed (documented, target reconciled after removal); duplicates audited but not deleted.

`data_layers/` mirrors the medallion pattern (1_bronze_raw_data, 2_silver_cleaned_data, 3_gold_warehouse) but is a presentation artifact — the actual runtime path is `etl_pipeline/` → `warehouse/`.

`archive_old_scope/` contains autism-screening files from an earlier scope. Excluded from main pipeline intentionally; do not re-introduce.

## Architecture: generic engine (`disease_warehouse/`)

```
disease_warehouse/
├── profiles/                   ← arg-list YAML registry (drop a file, it's onboarded)
│   ├── diabetes.yaml
│   └── stroke.yaml
├── core/
│   ├── profile.py              ← YAML schema + dataclasses (DatasetProfile, ColumnSpec, ...)
│   ├── registry.py             ← discover_profiles() — walks the folder
│   ├── metadata.py             ← profile_dataset() — declared+inferred ColumnMetadata
│   ├── cleaner.py              ← clean_dataset() — profile-driven cleaning pipeline
│   ├── schema_builder.py       ← build_schema_sql() — generates DDL from profile list
│   └── loader.py               ← WarehouseBuilder — orchestrates the whole build
├── cli.py / __main__.py        ← `python -m disease_warehouse {build|list-profiles|inspect|scaffold}`
└── outputs/                    ← generated artifacts (gold.db, parquet/, data_dictionary.json, …)
```

Pipeline order inside `WarehouseBuilder.build()`:

1. Load every profile YAML and validate.
2. For each profile, read raw CSV and run `profile_dataset()` (auto-metadata over RAW values, so distinct counts and stats reflect what the source actually contained).
3. For each profile, run `clean_dataset()`:
   strip strings → drop rows by rules → drop columns → **imputation** → **build hierarchies** → value mappings → domain validation → target reconciliation. (Imputation must precede hierarchy building — otherwise band columns inherit NaNs from un-imputed source columns. The stroke `bmi_band` regression caught this during initial testing.)
4. Generate full schema DDL via `build_schema_sql()` and write it to `outputs/schema.sql` for review, then execute it against a fresh `gold.db`.
5. Insert one `dim_etl_snapshot` row, populate lookup dims (`dim_age_bracket`, `dim_gender`), insert one `dim_dataset` row per profile.
6. For each profile, append rows to the shared `dim_patient_record` (with FK to `dim_dataset` so cohorts are joinable), then bulk-insert into the per-disease `fact_<name>` table.
7. Insert `metadata_data_dictionary` from the auto-profile results, `metadata_lineage` from the cleaning lineage, `etl_audit_log` from per-profile audits.
8. Export every table to Parquet via `pandas.read_sql_query` → `to_parquet`.

### Star-schema shape produced by the engine

**Conformed dims (shared across all diseases):**
- `dim_etl_snapshot` — load version / time-variant key
- `dim_dataset` — registry: one row per onboarded profile (`dataset_id`, name, domain, fact_table, source_path)
- `dim_patient_record` — one row per source record; carries `age`, `gender_id`, `age_bracket`; FK to `dim_dataset`
- `dim_age_bracket`, `dim_gender` — lookup dims (no NULL gender — uses `gender_id = -1` Unknown)

**Per-disease facts (one per profile):**
- `fact_<name>` — columns derived from `profile.fact_columns()` (excludes target, dropped cols, age, gender — those live on `dim_patient_record`). Includes derived hierarchy columns (e.g. `glucose_band`, `bmi_band`) and the `label` column.

**Per-disease views**: `vw_<name>_full` joins fact → patient_record → gender → dataset.
**Cross-disease view**: `vw_disease_summary` — UNION-ALL count/positive_rate per fact table.

### Profile YAML contract (must-know fields)

```yaml
name: <slug>                                  # becomes fact_<slug>
display_name: <human title>
domain: <metabolic|cardiovascular|...>        # free-text grouping
description: |
  multi-line dataset description

source:
  path: datasets/<file>.csv                   # relative to repo root
  format: csv
  na_values: ['?', '', 'N/A', ...]

target:
  column: <raw column name>
  rename_to: label                            # always 'label' in the warehouse
  positive_value: <raw label>                 # e.g. 'Positive' for diabetes, 1 for stroke
  negative_value: <raw label>
  positive_int: 1
  negative_int: 0

columns:
  - source: <raw column name>
    type: numeric|binary|nominal|ordinal|identifier
    role: age|gender|symptom|lab|comorbidity|lifestyle|geography|descriptor
    mapping: {Yes: 1, No: 0}                  # optional value translation
    domain: [val1, val2, ...]                 # optional allowed set for nominal
    description: <free text>
    drop: true                                # optional, e.g. for source ids
    imputation:                               # optional per-column
      strategy: grouped_median|median|constant
      group_source_column: age                # for grouped_median
      group_bands: [{label: '0-18', upper_exclusive: 18}, ...]
      fallback: global_median

cleaning:
  strip_whitespace: true
  drop_rows_where:
    - column: <raw>
      operator: equals|in|lt|gt
      value: <v>
      ci: true                                # case-insensitive match
      note: <audit note>

hierarchies:
  - source_column: <raw>
    new_column: <derived band column>
    bands:
      - {label: '<name>', upper_exclusive: <float or .inf>}
```

### Engine invariants worth knowing

- **Roles `age` and `gender` are special.** Columns with these roles get pulled onto `dim_patient_record` and are NOT placed in the per-disease fact table. Every profile must declare exactly one `age` role and one `gender` role for the warehouse to populate `dim_patient_record` correctly.
- **Target column is excluded from `fact_columns()` automatically.** The target is renamed to `label` and lives only on the fact table.
- **Hierarchies named `age_bracket` are special.** The engine puts `age_bracket` on `dim_patient_record` instead of on the fact table.
- **Imputation runs before hierarchy building** (see step 3 above). A `bmi`-imputation rule's `group_bands` are independent of the warehouse's `age_bracket` hierarchy — declare imputation bands explicitly inside the column's `imputation:` block.
- **Profile discovery is alphabetical** (`sorted()`). Duplicate `name:` slugs across YAMLs raise on `discover_profiles()`.
- **Engine never modifies raw CSVs.** Outputs go only to `disease_warehouse/outputs/`.

## Cross-cutting conventions

- Always pass `python -X utf8 ...` on this repo. The working directory contains non-ASCII characters that break `cp1252` stdout.
- Never re-introduce autism-screening files into the main pipeline; they are archived intentionally.
- Defense numbers that **must** reconcile (legacy pipeline): diabetes 520 rows / 320 pos / 200 neg; stroke 5,110 raw → 5,109 cleaned / 249 pos / 4,860 neg. The generic engine reproduces these same numbers in `fact_diabetes` / `fact_stroke`.
- When changing the generic engine, do **not** touch `etl_pipeline/` or `warehouse/schema.sql` — they back the defense package and the notebooks. The two pipelines are intentionally decoupled.
