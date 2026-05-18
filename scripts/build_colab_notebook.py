"""Generate disease_warehouse_colab.ipynb at the repo root.

One-shot script — produces a self-contained Colab notebook that walks through
the full disease_warehouse pipeline: install, upload code, configure API key,
build all 7 disease profiles, query the gold warehouse, scaffold a NEW dataset
(UCI Parkinson's) with and without LLM enrichment, then build it in.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# Disease Warehouse — End-to-End Demo

A profile-driven, dataset-agnostic healthcare data warehouse builder.

**What this notebook shows:**
1. Install dependencies in Colab
2. Upload the project code (or mount Drive) and configure `GEMINI_API_KEY`
3. Build the gold warehouse from **7 pre-registered disease profiles**
4. Query the resulting SQLite star schema
5. Onboard a NEW disease — UCI Parkinson's voice biomarkers — using the `scaffold` command
6. Compare deterministic heuristics vs **Gemini 2.5 Flash** LLM enrichment
7. Build the new disease in and query its fact table

**No Python code is written to add a new disease — only YAML profiles.**

> Author: hadethtarekeg14@gmail.com
"""),

        md("""## 1. Install Python dependencies

Colab already ships with `pandas`, `numpy`, `pyarrow`, and `pyyaml` at compatible
versions — pinning them here would force a downgrade that breaks the numpy/pandas
C-extension binding (`ValueError: numpy.dtype size changed`). We only install what's
actually missing.
"""),
        code("""!pip install -q google-generativeai python-dotenv"""),

        md("""> **If you previously ran a version that pinned `pandas==2.1.4`**, the
> kernel may already be in a broken state. Click **Runtime → Restart session**
> once, then continue from cell 2 below. You only need to do this once.
"""),

        md("""## 2. Upload the project code

Two options — pick one:

- **Option A (one-shot)** — upload the project as a `.zip` (right-click the project folder on your machine → Send to → Compressed folder), then run the cell below.
- **Option B (persistent)** — place the project folder on your Google Drive and mount it with the second cell.
"""),

        code("""# ── OPTION A: upload datamining_project.zip from your computer ──────────
import os, zipfile
from google.colab import files

uploaded = files.upload()
zip_name = next(iter(uploaded.keys()))
with zipfile.ZipFile(zip_name) as zf:
    zf.extractall('/content/')

# Locate project root (the folder containing disease_warehouse/)
PROJECT_DIR = None
for root, dirs, _ in os.walk('/content'):
    if 'disease_warehouse' in dirs and 'datasets' in dirs:
        PROJECT_DIR = root
        break

assert PROJECT_DIR, "Could not locate project root after unzip"
os.chdir(PROJECT_DIR)
print('Project root:', PROJECT_DIR)
!ls
"""),

        code("""# ── OPTION B (alternative): mount Google Drive ──────────────────────────
# Uncomment if your project lives on Drive instead of being uploaded as a zip.
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# PROJECT_DIR = '/content/drive/MyDrive/datamining_project-main'
# os.chdir(PROJECT_DIR)
# print('Project root:', PROJECT_DIR)
# !ls
"""),

        md("""## 3. Configure `GEMINI_API_KEY` for LLM enrichment

Two safe ways to provide the key:
- **Colab Secrets** — open the key icon in the left sidebar, add a secret named `GEMINI_API_KEY`, enable notebook access.
- **Paste at prompt** — the cell falls back to `getpass` if no secret is set.
"""),

        code("""import os
try:
    from google.colab import userdata
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
    print('Loaded GEMINI_API_KEY from Colab Secrets')
except Exception:
    import getpass
    os.environ['GEMINI_API_KEY'] = getpass.getpass('Paste your GEMINI_API_KEY: ')
    print('Set GEMINI_API_KEY for this session only')

# Also write to .env so the CLI's dotenv loader picks it up.
with open('.env', 'w') as f:
    f.write(f"GEMINI_API_KEY={os.environ['GEMINI_API_KEY']}\\n")
print('Wrote .env')
"""),

        md("""## 4. List the registered disease profiles

Every YAML in `disease_warehouse/profiles/` is a disease. No Python changes
are needed to register one — the engine discovers them at build time.
"""),

        code("""!python -X utf8 -m disease_warehouse list-profiles"""),

        md("""## 5. Build the gold warehouse from all profiles

Pipeline (per profile):
1. Read raw CSV → auto-collect metadata
2. Clean (strip, drop rows, drop cols, impute, build hierarchies, value mappings, domain validation)
3. Generate DDL → execute against fresh `gold.db`
4. Insert into shared dimensions (`dim_etl_snapshot`, `dim_dataset`, `dim_patient_record`, `dim_gender`, `dim_age_bracket`)
5. Insert into per-disease fact table (`fact_<name>`)
6. Export every table to Parquet
"""),

        code("""!python -X utf8 -m disease_warehouse build"""),

        md("## 6. Explore the gold SQLite warehouse"),

        code("""import sqlite3
import pandas as pd

conn = sqlite3.connect('disease_warehouse/outputs/gold.db')

print('Tables in the warehouse:')
tables = pd.read_sql(
    "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type DESC, name",
    conn,
)
tables
"""),

        code("""print('Cross-disease summary view:')
pd.read_sql('SELECT * FROM vw_disease_summary', conn)
"""),

        code("""print('Dataset registry:')
pd.read_sql('SELECT dataset_id, dataset_name, domain, fact_table, source_path FROM dim_dataset', conn)
"""),

        code("""print('Patient record dim — sample:')
pd.read_sql('SELECT * FROM dim_patient_record LIMIT 5', conn)
"""),

        md("""## 7. Onboarding a NEW disease — UCI Parkinson's voice biomarkers

The Parkinson's dataset has 24 columns of dysphonia features
(`MDVP:Fo`, `MDVP:Jitter`, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1`, `PPE`...).
**None of these match the medical keyword dictionary** — perfect for
showing the LLM fallback in action.
"""),

        code("""import urllib.request, pandas as pd

url = 'https://archive.ics.uci.edu/static/public/174/data.csv'
urllib.request.urlretrieve(url, 'datasets/parkinsons.csv')

df = pd.read_csv('datasets/parkinsons.csv')
print(f'Downloaded: {len(df)} rows, {len(df.columns)} columns')
print('Columns:', list(df.columns))
df.head(3)
"""),

        md("""### 7a. Baseline — deterministic heuristics only

This is what the engine produces from keyword matching + numeric range
fingerprints + unit-suffix patterns alone. Expect lots of `# REVIEW:` comments.
"""),

        code("""!python -X utf8 -m disease_warehouse scaffold datasets/parkinsons.csv \\
    --name parkinsons_baseline --no-llm --force"""),

        md("""### 7b. With Gemini 2.5 Flash enrichment

The same scaffold, but every LOW-confidence column is batched into one
`response_schema`-constrained Gemini call. Cached to
`disease_warehouse/outputs/.llm_cache.json` so re-runs are free.
"""),

        code("""!python -X utf8 -m disease_warehouse scaffold datasets/parkinsons.csv \\
    --name parkinsons --use-llm --force"""),

        md("### 7c. Compare before / after"),

        code("""from pathlib import Path

baseline = Path('disease_warehouse/profiles/parkinsons_baseline.yaml').read_text(encoding='utf-8')
enriched = Path('disease_warehouse/profiles/parkinsons.yaml').read_text(encoding='utf-8')

base_reviews = sum(1 for line in baseline.splitlines() if '# REVIEW' in line and not line.lstrip().startswith('# Decisions'))
enr_reviews  = sum(1 for line in enriched.splitlines() if '# REVIEW' in line and not line.lstrip().startswith('# Decisions'))

print(f'Baseline YAML:  {base_reviews} REVIEW comments')
print(f'Enriched YAML:  {enr_reviews} REVIEW comments')
"""),

        code("""print('=== ENRICHED YAML (the one the LLM filled) ===')
print(enriched)
"""),

        md("""## 8. Build the new disease into the warehouse

Delete the baseline (test-only) profile, keep the enriched one, rebuild.
"""),

        code("""import os
if os.path.exists('disease_warehouse/profiles/parkinsons_baseline.yaml'):
    os.remove('disease_warehouse/profiles/parkinsons_baseline.yaml')

!python -X utf8 -m disease_warehouse build --profile parkinsons"""),

        md("## 9. Query the new Parkinson's fact table"),

        code("""conn = sqlite3.connect('disease_warehouse/outputs/gold.db')

print('Parkinson\\'s fact table — first 5 rows:')
pd.read_sql('SELECT * FROM fact_parkinsons LIMIT 5', conn)
"""),

        code("""print('Label distribution (0 = healthy, 1 = Parkinson\\'s):')
pd.read_sql('SELECT label, COUNT(*) AS n FROM fact_parkinsons GROUP BY label ORDER BY label', conn)
"""),

        code("""print('Updated dataset registry — new disease appears:')
pd.read_sql('SELECT dataset_id, dataset_name, domain, fact_table FROM dim_dataset ORDER BY dataset_id', conn)
"""),

        code("""print('Cross-disease summary AFTER adding Parkinson\\'s:')
pd.read_sql('SELECT * FROM vw_disease_summary', conn)
"""),

        md("""## 11. Your turn — onboard a different dataset yourself

Try the engine on **UCI Maternal Health Risk** — a completely different domain
(obstetrics) than anything in the warehouse so far. Multi-class target
(`RiskLevel`: low / mid / high), 1,014 rows, 7 columns. It's already in
`datasets/maternal_health_risk.csv` from the project zip.

Run the three cells below in order. You should get **LOW=0** and a fact table
with 1,014 rows. Open the generated YAML to see what the engine inferred —
the multi-class target gets binarized automatically (low → 0, mid/high → 1).
"""),

        code("""# Step 1: auto-profile the CSV into a YAML
!python -X utf8 -m disease_warehouse scaffold datasets/maternal_health_risk.csv \\
    --name maternal_health --use-llm --force"""),

        code("""# Step 2: peek at the generated YAML so you see what the engine produced
print(open('disease_warehouse/profiles/maternal_health.yaml').read())"""),

        code("""# Step 3: build the new disease into the warehouse
!python -X utf8 -m disease_warehouse build --profile maternal_health"""),

        code("""# Step 4: query the new fact table
conn = sqlite3.connect('disease_warehouse/outputs/gold.db')
print('Sample rows:')
display(pd.read_sql('SELECT * FROM fact_maternal_health LIMIT 5', conn))
print()
print('Label distribution (0 = low risk, 1 = mid/high risk):')
display(pd.read_sql('SELECT label, COUNT(*) AS n FROM fact_maternal_health GROUP BY label', conn))
print()
print('Final dataset registry (should now have 9 entries):')
display(pd.read_sql('SELECT dataset_id, dataset_name, domain, fact_table FROM dim_dataset ORDER BY dataset_id', conn))"""),

        md("""**Want to try another dataset entirely?**
Drop any CSV into `datasets/` (use the Colab file browser → `datasets/` → Upload),
then repeat steps 1-4 with `--name your_disease_slug` and your file path.
No Python code changes required.
"""),

        md("""## 10. What just happened — recap for the prof

**The engine's job:** take any healthcare CSV → produce a star-schema warehouse
with conformed dimensions and per-disease facts, plus Parquet exports, a data
dictionary, and lineage.

**The auto-profiler's job:** turn a raw CSV into a runnable YAML profile
without human editing. It combines five layers:

| Layer | What it does | Confidence |
|---|---|---|
| **Statistical inference** | dtype, cardinality, identifier detection, target detection | HIGH |
| **Medical keyword dictionary** | maps column names → role + description + standard hierarchy | HIGH |
| **Numeric value-range fingerprints** | BP / glucose / BMI / HR / cholesterol / age detected from min-max-median | MEDIUM |
| **Multilingual + unit-suffix patterns** | `edad`, `sexo`, `_mmhg`, `_mg_dl`, `_kg`, `_cm` | HIGH |
| **Gemini 2.5 Flash fallback** | structured-output classification for anything still LOW | varies |

**Result on Parkinson's** (24 columns, all cryptic dysphonia features):

| | Baseline | + Gemini |
|---|---|---|
| LOW (needs human review) | ~42 | **0** |
| API calls | 0 | 1 batched call |
| Build outcome | unbuildable | `fact_parkinsons` with 195 rows |

**To add another disease**, the prof can drop a CSV into `datasets/`, run
`scaffold`, optionally tweak the generated YAML, and `build`. Nothing else
changes — the schema regenerates, dimensions stay conformed, and the new
fact table joins cleanly into `vw_disease_summary`.
"""),
    ]

    nb.metadata = {
        "colab": {
            "name": "disease_warehouse_colab.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def main() -> None:
    nb = build()
    out = Path(__file__).resolve().parent.parent / "disease_warehouse_colab.ipynb"
    with out.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
