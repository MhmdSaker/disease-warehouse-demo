# Disease Warehouse — How to Run This Demo

A profile-driven, dataset-agnostic **healthcare data warehouse builder**. This
notebook onboards 7 disease datasets into one SQLite star schema, then adds
an 8th disease (UCI Parkinson's voice biomarkers) on the fly using an
LLM-assisted scaffold command.

---

## What you need

- A Google account (Colab is free)
- A free Gemini API key from https://aistudio.google.com/app/apikey (~30 seconds to create)

---

## 2-minute setup

1. **Open the notebook in Colab** — click the badge:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MhmdSaker/disease-warehouse-demo/blob/main/disease_warehouse_colab.ipynb)

2. **Add your Gemini key as a Colab Secret:**
   - Click the **key icon** in Colab's left sidebar (labelled "Secrets")
   - **+ Add new secret**
   - **Name:** `GEMINI_API_KEY` (case-sensitive)
   - **Value:** paste your key
   - Toggle **Notebook access** ON
3. **Run All** — `Runtime → Run all`
4. Wait ~3 minutes for the full pipeline to finish

That's it. The notebook clones the project from GitHub itself — no manual upload.

---

## What the notebook does (section by section)

| § | What runs | What to look at |
|---|---|---|
| 1 | Install Python deps | Just pip output |
| 2 | Unzip the project | One upload prompt |
| 3 | Load the API key | Should say "Loaded from Colab Secrets" |
| 4 | List the 7 registered diseases | Diabetes, stroke, heart disease, cardio, CDC indicators, kidney disease |
| 5 | Build the full gold warehouse | ~580k rows across 7 fact tables, plus conformed dimensions |
| 6 | Query the SQLite warehouse | Table list, dataset registry, cross-disease summary view |
| 7 | Auto-onboard Parkinson's | Baseline scaffold (heuristics only) vs LLM-enriched scaffold — side by side |
| 8 | Build the new disease in | Same engine, no Python edits — just YAML |
| 9 | Query the new fact table | Label distribution, updated registry |
| 10 | Recap | Before/after scoreboard |

---

## Highlight to watch for (section 7c)

The Parkinson's CSV has 24 columns of dysphonia features
(`MDVP:Fo`, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1`, `PPE`...) — **none**
of which appear in our medical keyword dictionary.

| | Baseline (heuristics only) | + Gemini 2.5 Flash |
|---|---|---|
| Columns flagged for human review | **~42** | **0** |
| API calls | 0 | 1 (batched) |
| Build outcome | Unbuildable without manual edits | `fact_parkinsons` with 195 rows |

---

## Want to try it yourself?

Section 11 of the notebook ("Your turn") walks you through onboarding the
**UCI Maternal Health Risk** dataset (1,014 rows, obstetrics — a domain not
yet in the warehouse). Just run the 4 cells in that section in order.

To onboard ANY OTHER CSV:
1. Upload your file via Colab's left-sidebar file browser → `datasets/`
2. Run a scaffold command, replacing the placeholders:
   ```bash
   !python -X utf8 -m disease_warehouse scaffold datasets/YOUR_FILE.csv \
       --name your_disease_slug --use-llm --force
   ```
3. Skim the generated YAML at `disease_warehouse/profiles/your_disease_slug.yaml`
4. Build:
   ```bash
   !python -X utf8 -m disease_warehouse build --profile your_disease_slug
   ```

No Python changes required — every step is a CLI command driven by the YAML.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GEMINI_API_KEY not found` | Re-check the secret name is exactly `GEMINI_API_KEY` and notebook access is ON |
| Upload prompt never appeared | Re-run **only cell 5** (the `files.upload()` cell) |
| `ModuleNotFoundError: disease_warehouse` | The `os.chdir(PROJECT_DIR)` in cell 5 didn't run — re-run cell 5 |
| Cell `build` looks stuck | First build takes ~2 min (7 CSVs → 23 tables + Parquet exports). Watch for `BUILD COMPLETE` |
| Want to skip the LLM entirely | In section 7b, change `--use-llm` to `--no-llm`. The comparison still works — you'll see the baseline keep its 42 `# REVIEW:` comments |

---

## Project repository

All code, data, and the notebook live at:
**https://github.com/MhmdSaker/disease-warehouse-demo**

You can browse the engine source under `disease_warehouse/core/`, the YAML
profiles under `disease_warehouse/profiles/`, and the raw CSVs under
`datasets/`.
