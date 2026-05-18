"""Auto-profile a CSV into a usable disease_warehouse YAML profile.

The inference engine combines:

* **Statistical inference** — delimiter sniffing, type classification (numeric /
  binary / ordinal / nominal / identifier), distinct-count + null + range
  stats, identifier detection.
* **Medical-domain heuristics** — a curated keyword dictionary maps column
  names (age, sex, bmi, ap_hi, smoke, hypertension, ...) to roles, standard
  hierarchies, and template descriptions.
* **Value-pattern matching** — recognises Yes/No, Positive/Negative, Male/Female,
  Urban/Rural, etc., and emits the appropriate ``mapping:`` block.

Every decision is tagged with a confidence level (HIGH / MEDIUM / LOW). The
emitted YAML carries inline ``# REVIEW:`` markers next to low-confidence
decisions so the operator knows exactly what the engine guessed.

The output is a *dict* (and a list of decision notes) — a separate emitter
in :mod:`disease_warehouse.core.yaml_writer` turns it into commented YAML.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"


# ---------------------------------------------------------------------------
# 1. Medical-keyword dictionary
# ---------------------------------------------------------------------------
# Each entry maps a regex pattern (matched against a snake_cased column name)
# to: role, suggested description, optional hierarchy_key, optional category.
# Patterns are case-insensitive and matched in order; the first hit wins.

@dataclass(frozen=True)
class ColumnPattern:
    pattern: str
    role: str | None
    description: str
    hierarchy_key: str | None = None
    category: str = ""

    def matches(self, normalized_name: str) -> bool:
        return re.fullmatch(self.pattern, normalized_name, re.IGNORECASE) is not None


COLUMN_PATTERNS: list[ColumnPattern] = [
    # ─── demographics / IDs ─────────────────────────────────────────────────
    ColumnPattern(r"id|patient[_-]?id|case[_-]?id|record[_-]?id|subject[_-]?id",
                  None, "Source-side identifier (dropped, replaced by warehouse surrogate)."),
    ColumnPattern(r"age|age[_-]?years?|years?[_-]?old",
                  "age", "Patient age in years.", "age_bracket"),
    ColumnPattern(r"age[_-]?group|age[_-]?category|age[_-]?bracket",
                  "age", "Age group / bracket (ordinal). Map to midpoint years if BRFSS-style 1-13 code.", "age_bracket"),
    ColumnPattern(r"sex|gender",
                  "gender", "Patient sex / gender."),

    # ─── labs / vitals ──────────────────────────────────────────────────────
    ColumnPattern(r"bmi|body[_-]?mass[_-]?index",
                  "lab", "Body Mass Index (kg/m²).", "bmi_band"),
    ColumnPattern(r"trestbps|ap[_-]?hi|systolic|sbp|bp[_-]?sys",
                  "lab", "Systolic blood pressure (mm Hg).", "bp_band"),
    ColumnPattern(r"ap[_-]?lo|diastolic|dbp|bp[_-]?dia",
                  "lab", "Diastolic blood pressure (mm Hg)."),
    ColumnPattern(r"chol|cholesterol|total[_-]?cholesterol|tc",
                  "lab", "Serum cholesterol.", "chol_band"),
    ColumnPattern(r"hdl|ldl|triglycerides?|trig",
                  "lab", "Lipid panel measurement."),
    ColumnPattern(r"glucose|gluc|avg[_-]?glucose[_-]?level|fasting[_-]?glucose|hba1c|a1c",
                  "lab", "Glucose / glycemic measurement.", "glucose_band"),
    ColumnPattern(r"fbs|fasting[_-]?blood[_-]?sugar",
                  "lab", "Fasting blood sugar flag (> 120 mg/dL)."),
    ColumnPattern(r"thalach|max[_-]?heart[_-]?rate|peak[_-]?hr",
                  "lab", "Maximum heart rate achieved (stress test)."),
    ColumnPattern(r"oldpeak|st[_-]?depression",
                  "lab", "ST depression induced by exercise relative to rest."),
    ColumnPattern(r"restecg|resting[_-]?ecg",
                  "lab", "Resting ECG result."),
    ColumnPattern(r"slope|st[_-]?slope",
                  "lab", "Slope of peak exercise ST segment."),
    ColumnPattern(r"ca|num[_-]?major[_-]?vessels",
                  "lab", "Number of major vessels (0–3) coloured by fluoroscopy."),
    ColumnPattern(r"thal|thalassemia",
                  "lab", "Thallium stress test result (3=normal, 6=fixed, 7=reversible)."),
    ColumnPattern(r"height|height[_-]?cm",
                  "lab", "Patient height (cm)."),
    ColumnPattern(r"weight|weight[_-]?kg",
                  "lab", "Patient weight (kg)."),

    # ─── hepatology / liver-function panel (UCI ILPD idioms) ──────────────
    ColumnPattern(r"tb|total[_-]?bilirubin|t[_-]?bilirubin",
                  "lab", "Total bilirubin (mg/dL); composite liver-function marker."),
    ColumnPattern(r"db|direct[_-]?bilirubin|d[_-]?bilirubin|conjugated[_-]?bilirubin",
                  "lab", "Direct (conjugated) bilirubin (mg/dL)."),
    ColumnPattern(r"alkphos|alp|alkaline[_-]?phosphatase|alk[_-]?phos",
                  "lab", "Alkaline phosphatase (IU/L); raised in cholestasis."),
    ColumnPattern(r"sgpt|alt|alanine[_-]?aminotransferase|alanine[_-]?transaminase",
                  "lab", "ALT / SGPT — alanine aminotransferase (IU/L); hepatocyte injury marker."),
    ColumnPattern(r"sgot|ast|aspartate[_-]?aminotransferase|aspartate[_-]?transaminase",
                  "lab", "AST / SGOT — aspartate aminotransferase (IU/L); hepatocyte injury marker."),
    ColumnPattern(r"tp|total[_-]?protein|total[_-]?proteins|proteins",
                  "lab", "Total protein in serum (g/dL)."),
    ColumnPattern(r"alb|albumin",
                  "lab", "Serum albumin (g/dL); synthesised by the liver."),
    ColumnPattern(r"a[_-]?g[_-]?ratio|albumin[_-]?globulin[_-]?ratio|globulin[_-]?ratio",
                  "lab", "Albumin / globulin ratio; reflects liver-synthesis function."),

    # ─── nephrology / renal labs (UCI Chronic Kidney Disease idioms) ───────
    ColumnPattern(r"bp",   "lab", "Blood pressure (mm Hg).", "bp_band"),
    ColumnPattern(r"sg",   "lab", "Urine specific gravity (ordinal: 1.005, 1.010, 1.015, 1.020, 1.025)."),
    ColumnPattern(r"al",   "lab", "Urine albumin (ordinal 0–5)."),
    ColumnPattern(r"su",   "lab", "Urine sugar (ordinal 0–5)."),
    ColumnPattern(r"rbc",  "lab", "Red blood cells, qualitative microscopy (normal / abnormal)."),
    ColumnPattern(r"pc",   "lab", "Pus cells, qualitative microscopy (normal / abnormal)."),
    ColumnPattern(r"pcc",  "lab", "Pus cell clumps in urine (present / notpresent)."),
    ColumnPattern(r"ba",   "lab", "Bacteria in urine (present / notpresent)."),
    ColumnPattern(r"bgr",  "lab", "Blood glucose, random (mg/dL).", "glucose_band"),
    ColumnPattern(r"bu",   "lab", "Blood urea (mg/dL); renal-function marker."),
    ColumnPattern(r"sc",   "lab", "Serum creatinine (mg/dL); primary renal-function marker."),
    ColumnPattern(r"sod",  "lab", "Serum sodium (mEq/L)."),
    ColumnPattern(r"pot",  "lab", "Serum potassium (mEq/L)."),
    ColumnPattern(r"hemo", "lab", "Hemoglobin (g/dL)."),
    ColumnPattern(r"pcv",  "lab", "Packed cell volume / hematocrit (%)."),
    ColumnPattern(r"wbcc", "lab", "White blood cell count (cells/cmm)."),
    ColumnPattern(r"rbcc", "lab", "Red blood cell count (millions/cmm)."),
    ColumnPattern(r"dm",   "comorbidity", "Diabetes mellitus (yes / no)."),
    ColumnPattern(r"cad",  "comorbidity", "Coronary artery disease (yes / no)."),
    ColumnPattern(r"ane",  "comorbidity", "Anemia (yes / no)."),
    ColumnPattern(r"pe",   "symptom",     "Pedal edema (yes / no)."),
    ColumnPattern(r"appet", "descriptor", "Appetite (good / poor)."),

    # ─── comorbidities ──────────────────────────────────────────────────────
    ColumnPattern(r"hypertension|highbp|high[_-]?bp|htn",
                  "comorbidity", "Self-reported hypertension / high blood pressure."),
    ColumnPattern(r"highchol|high[_-]?cholesterol|hyperlipidemia",
                  "comorbidity", "Self-reported high cholesterol."),
    ColumnPattern(r"heart[_-]?disease|heartdiseaseorattack|chd|cvd|myocardial[_-]?infarction|mi",
                  "comorbidity", "Coronary heart disease / myocardial infarction history."),
    ColumnPattern(r"stroke",
                  "comorbidity", "History of stroke."),
    ColumnPattern(r"diabetes|diabetic",
                  "comorbidity", "Diabetes status / history."),
    ColumnPattern(r"kidney|ckd|renal",
                  "comorbidity", "Kidney / renal disease indicator."),

    # ─── symptoms ───────────────────────────────────────────────────────────
    ColumnPattern(r"polyuria",      "symptom", "Excessive urination (osmotic-diuresis symptom of diabetes)."),
    ColumnPattern(r"polydipsia",    "symptom", "Excessive thirst."),
    ColumnPattern(r"polyphagia",    "symptom", "Excessive hunger."),
    ColumnPattern(r"cp|chest[_-]?pain",                              "symptom", "Chest pain type."),
    ColumnPattern(r"exang|exercise[_-]?induced[_-]?angina",          "symptom", "Exercise-induced angina."),
    ColumnPattern(r"weakness|fatigue|tiredness",                     "symptom", "Generalised weakness or fatigue."),
    ColumnPattern(r"sudden[_-]?weight[_-]?loss|weight[_-]?loss",     "symptom", "Sudden / unintentional weight loss."),
    ColumnPattern(r"itching|pruritus",                               "symptom", "Itching / pruritus."),
    ColumnPattern(r"irritability|mood[_-]?lability",                 "symptom", "Mood lability / irritability."),
    ColumnPattern(r"visual[_-]?blurring|blurred[_-]?vision",         "symptom", "Visual blurring."),
    ColumnPattern(r"delayed[_-]?healing|wound[_-]?healing",          "symptom", "Slow / delayed wound healing."),
    ColumnPattern(r"partial[_-]?paresis|paresis",                    "symptom", "Partial muscle weakness."),
    ColumnPattern(r"muscle[_-]?stiffness",                           "symptom", "Muscle stiffness."),
    ColumnPattern(r"alopecia|hair[_-]?loss",                         "symptom", "Hair loss."),
    ColumnPattern(r"obesity",                                        "symptom", "Self-reported obesity flag."),
    ColumnPattern(r"genital[_-]?thrush|candidiasis",                 "symptom", "Genital candidiasis."),
    ColumnPattern(r"diffwalk|difficulty[_-]?walking",                "symptom", "Difficulty walking or climbing stairs."),

    # ─── lifestyle ──────────────────────────────────────────────────────────
    ColumnPattern(r"smoker|smoke|smoking|smoking[_-]?status|tobacco",
                  "lifestyle", "Smoking status."),
    ColumnPattern(r"alco|alcohol|drinker|drinking|hvyalcoholconsump",
                  "lifestyle", "Alcohol consumption."),
    ColumnPattern(r"active|physactivity|physical[_-]?activity|exercise",
                  "lifestyle", "Physical activity / exercise."),
    ColumnPattern(r"fruits?",     "lifestyle", "Fruit consumption (≥1×/day)."),
    ColumnPattern(r"veggies|vegetables?", "lifestyle", "Vegetable consumption (≥1×/day)."),
    ColumnPattern(r"diet",        "lifestyle", "Dietary pattern."),
    ColumnPattern(r"work[_-]?type|occupation|job", "lifestyle", "Occupation category."),

    # ─── geography ──────────────────────────────────────────────────────────
    ColumnPattern(r"residence|residence[_-]?type|location|urban|rural",
                  "geography", "Urban / rural residence."),

    # ─── descriptors / SES ──────────────────────────────────────────────────
    ColumnPattern(r"education|edu",       "descriptor", "Education level."),
    ColumnPattern(r"income|earnings",     "descriptor", "Income bracket."),
    ColumnPattern(r"marital|married|ever[_-]?married", "descriptor", "Marital status."),
    ColumnPattern(r"genhlth|general[_-]?health|self[_-]?rated[_-]?health", "descriptor", "Self-rated general health."),
    ColumnPattern(r"menthlth|mental[_-]?health", "descriptor", "Days of poor mental health in past 30."),
    ColumnPattern(r"physhlth|physical[_-]?health", "descriptor", "Days of poor physical health in past 30."),
    ColumnPattern(r"anyhealthcare|healthcare|insurance|coverage", "descriptor", "Healthcare coverage indicator."),
    ColumnPattern(r"nodocbccost", "descriptor", "Could not see a doctor due to cost."),
    ColumnPattern(r"cholcheck|chol[_-]?check", "descriptor", "Cholesterol check within past 5 years."),

    # ─── multilingual aliases (common European medical terms) ──────────────
    ColumnPattern(r"edad|alter|age?",                     "age",      "Patient age in years (multilingual alias).", "age_bracket"),
    ColumnPattern(r"sexo|geschlecht|sexe",                "gender",   "Patient sex / gender (multilingual alias)."),
    ColumnPattern(r"peso|gewicht|poids",                  "lab",      "Patient weight (kg) (multilingual alias)."),
    ColumnPattern(r"altura|estatura|grosse|taille",       "lab",      "Patient height (cm) (multilingual alias)."),
    ColumnPattern(r"presion|presi_n|pression|druck",      "lab",      "Blood pressure (multilingual alias).", "bp_band"),
    ColumnPattern(r"colesterol|cholesterin",              "lab",      "Cholesterol (multilingual alias).", "chol_band"),
    ColumnPattern(r"glucosa|glucos|zucker",               "lab",      "Glucose (multilingual alias).", "glucose_band"),
    ColumnPattern(r"fumador|raucher|fumeur",              "lifestyle","Smoking status (multilingual alias)."),

    # ─── unit-suffix idioms (when authors stick units onto column names) ──
    ColumnPattern(r".*_mm_?hg|.*_mmhg",                   "lab",      "Pressure measurement (mm Hg) inferred from unit suffix.", "bp_band"),
    ColumnPattern(r".*_mg_?dl|.*_mgdl",                   "lab",      "Lab value (mg/dL) inferred from unit suffix."),
    ColumnPattern(r".*_mmol_?l|.*_mmoll",                 "lab",      "Lab value (mmol/L) inferred from unit suffix."),
    ColumnPattern(r".*_iu_?l|.*_u_?l|.*_iul",             "lab",      "Enzyme activity (IU/L or U/L) inferred from unit suffix."),
    ColumnPattern(r".*_g_?dl|.*_gdl",                     "lab",      "Concentration (g/dL) inferred from unit suffix."),
    ColumnPattern(r".*_meq_?l",                           "lab",      "Electrolyte (mEq/L) inferred from unit suffix."),
    ColumnPattern(r".*_bpm|.*_per_?min",                  "lab",      "Rate (per minute) inferred from unit suffix."),
    ColumnPattern(r".*_kg",                               "lab",      "Mass (kg) inferred from unit suffix."),
    ColumnPattern(r".*_cm|.*_mm",                         "lab",      "Length (cm/mm) inferred from unit suffix."),
]


# ---------------------------------------------------------------------------
# Numeric value-range fingerprints — used as a MEDIUM-confidence fallback
# when the column name alone gives us nothing. A fingerprint passes if both
# the observed range AND median fall inside the expected window. Order
# matters: more specific (narrower) patterns first.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeFingerprint:
    role: str
    description: str
    hierarchy_key: str | None
    # Min/max bound the observed *full* range; med_lo/med_hi bound the median.
    min_lo: float
    min_hi: float
    max_lo: float
    max_hi: float
    med_lo: float
    med_hi: float
    label: str

    def matches(self, mn: float, mx: float, med: float) -> bool:
        return (self.min_lo <= mn <= self.min_hi
                and self.max_lo <= mx <= self.max_hi
                and self.med_lo <= med <= self.med_hi)


RANGE_FINGERPRINTS: list[RangeFingerprint] = [
    # systolic BP: typical adult range 90-200, median ~120-140
    RangeFingerprint("lab", "Systolic blood pressure (inferred from value range).", "bp_band",
                     min_lo=60, min_hi=110, max_lo=140, max_hi=260, med_lo=105, med_hi=160, label="systolic-BP"),
    # diastolic BP: 50-120, median ~70-90
    RangeFingerprint("lab", "Diastolic blood pressure (inferred from value range).", None,
                     min_lo=30, min_hi=70, max_lo=80, max_hi=140, med_lo=60, med_hi=100, label="diastolic-BP"),
    # BMI: 13-60, median ~22-35
    RangeFingerprint("lab", "Body Mass Index (inferred from value range).", "bmi_band",
                     min_lo=10, min_hi=20, max_lo=28, max_hi=80, med_lo=20, med_hi=40, label="BMI"),
    # fasting/random glucose mg/dL: 40-500, median 80-180
    RangeFingerprint("lab", "Glucose / glycemic measurement (inferred from value range).", "glucose_band",
                     min_lo=20, min_hi=80, max_lo=140, max_hi=600, med_lo=70, med_hi=200, label="glucose"),
    # total cholesterol mg/dL: 100-400, median 150-260
    RangeFingerprint("lab", "Cholesterol (inferred from value range).", "chol_band",
                     min_lo=80, min_hi=160, max_lo=200, max_hi=500, med_lo=150, med_hi=280, label="cholesterol"),
    # heart rate bpm: 40-220, median 60-110
    RangeFingerprint("lab", "Heart rate (inferred from value range).", None,
                     min_lo=30, min_hi=70, max_lo=100, max_hi=240, med_lo=55, med_hi=120, label="heart-rate"),
    # age (years): 0-120, median 20-80 — used only when role wasn't already 'age'
    RangeFingerprint("age", "Patient age in years (inferred from value range).", "age_bracket",
                     min_lo=0, min_hi=20, max_lo=40, max_hi=120, med_lo=20, med_hi=80, label="age-years"),
]


def _fingerprint_numeric(series: pd.Series) -> RangeFingerprint | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or len(s) < 10:
        return None
    # Use 5th/95th percentiles instead of raw min/max so a single out-of-range
    # outlier (e.g. HeartRate=7 in an otherwise 60-90 column, or a sentinel
    # like -999) doesn't knock the column out of its true fingerprint window.
    mn, mx = float(s.quantile(0.05)), float(s.quantile(0.95))
    med = float(s.median())
    if mn == mx:
        return None
    for fp in RANGE_FINGERPRINTS:
        if fp.matches(mn, mx, med):
            return fp
    return None


TARGET_NAME_PATTERNS = [
    r"^class$",
    r"^label$",
    r"^target$",
    r"^outcome$",
    r"^diagnosis$",
    r".*_binary$",
    r"^diabetes$",
    r"^stroke$",
    r"^cardio$",
    r"^heartdiseaseorattack$",
    r"^num$",                       # UCI heart disease
    r"^selector$",                  # UCI ILPD
    r"^dataset$",                   # some BRFSS-derived files
    r"^status$",                    # UCI Parkinson's
    r"^y$",
    r".*_label$",
    r".*_outcome$",
]


# ---------------------------------------------------------------------------
# 2. Value-pattern recognisers (for mapping inference)
# ---------------------------------------------------------------------------

KNOWN_BINARY_MAPPINGS: list[tuple[set[str], dict[str, int], str]] = [
    # NOTE: template keys are stored lowercase; the actual cased values from
    # the data are substituted in at match time so the produced mapping fires
    # against the real values regardless of case ('Yes' vs 'yes' vs 'YES').
    ({"yes", "no"},                {"yes": 1, "no": 0},                "Yes/No"),
    ({"true", "false"},            {"true": 1, "false": 0},            "True/False"),
    ({"male", "female"},           {"male": 1, "female": 0},           "Male/Female"),
    ({"m", "f"},                   {"m": 1, "f": 0},                   "M/F"),
    ({"positive", "negative"},     {"positive": 1, "negative": 0},     "Positive/Negative"),
    ({"present", "absent"},        {"present": 1, "absent": 0},        "Present/Absent"),
    ({"present", "notpresent"},    {"present": 1, "notpresent": 0},    "Present/NotPresent"),
    ({"urban", "rural"},           {"urban": 1, "rural": 0},           "Urban/Rural"),
    ({"normal", "abnormal"},       {"abnormal": 1, "normal": 0},       "Normal/Abnormal"),
    ({"good", "poor"},             {"good": 0, "poor": 1},             "Good/Poor (poor=worse=positive)"),
    ({"ckd", "notckd"},            {"ckd": 1, "notckd": 0},            "ckd/notckd"),
    ({"disease", "healthy"},       {"disease": 1, "healthy": 0},       "disease/healthy"),
    ({"benign", "malignant"},      {"malignant": 1, "benign": 0},      "Benign/Malignant (M=B Wisconsin BC)"),
    ({"b", "m"},                   {"m": 1, "b": 0},                   "B/M (Wisconsin breast cancer)"),
]


# ---------------------------------------------------------------------------
# 3. Standard clinical hierarchies (emitted when matching column present)
# ---------------------------------------------------------------------------

STANDARD_HIERARCHIES: dict[str, list[dict[str, Any]]] = {
    "age_bracket": [
        {"label": "0-17",  "upper_exclusive": 18},
        {"label": "18-39", "upper_exclusive": 40},
        {"label": "40-59", "upper_exclusive": 60},
        {"label": "60-79", "upper_exclusive": 80},
        {"label": "80+",   "upper_exclusive": ".inf"},
    ],
    "bmi_band": [
        {"label": "underweight", "upper_exclusive": 18.5},
        {"label": "normal",      "upper_exclusive": 25},
        {"label": "overweight",  "upper_exclusive": 30},
        {"label": "obese",       "upper_exclusive": ".inf"},
    ],
    "bp_band": [
        {"label": "normal",      "upper_exclusive": 120},
        {"label": "elevated",    "upper_exclusive": 130},
        {"label": "stage-1-htn", "upper_exclusive": 140},
        {"label": "stage-2-htn", "upper_exclusive": 180},
        {"label": "crisis",      "upper_exclusive": ".inf"},
    ],
    "chol_band": [
        {"label": "desirable",  "upper_exclusive": 200},
        {"label": "borderline", "upper_exclusive": 240},
        {"label": "high",       "upper_exclusive": ".inf"},
    ],
    "glucose_band": [
        {"label": "normal",         "upper_exclusive": 100},
        {"label": "elevated",       "upper_exclusive": 126},
        {"label": "diabetic-range", "upper_exclusive": ".inf"},
    ],
}


# ---------------------------------------------------------------------------
# 4. The inference engine
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    column: str
    field: str           # e.g. 'type', 'role', 'mapping'
    value: Any
    confidence: str      # HIGH / MEDIUM / LOW
    reason: str

    def is_review(self) -> bool:
        return self.confidence == LOW


@dataclass
class InferenceResult:
    profile: dict
    decisions: list[Decision] = field(default_factory=list)
    delimiter: str = ","
    notes: list[str] = field(default_factory=list)

    def review_items(self) -> list[Decision]:
        return [d for d in self.decisions if d.is_review()]


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower())


def _looks_int(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    return bool((s % 1 == 0).all())


def _match_pattern(column_name: str) -> ColumnPattern | None:
    norm = _normalize(column_name)
    for pat in COLUMN_PATTERNS:
        if pat.matches(norm):
            return pat
    return None


def _sniff_delimiter(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter, HIGH
    except csv.Error:
        # Fall back: pick whichever delimiter splits the first line into the
        # most columns.
        first_line = sample.splitlines()[0] if sample else ""
        best = max([",", ";", "\t", "|"], key=lambda d: first_line.count(d))
        return best, MEDIUM


def _detect_target(
    df: pd.DataFrame,
    explicit: str | None,
) -> tuple[str, str, str]:
    """Returns (target_column, confidence, reason)."""
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit target {explicit!r} not in CSV columns")
        return explicit, HIGH, "explicit --target flag"

    # 1. Match by name pattern.
    name_hits: list[tuple[str, int]] = []
    for i, col in enumerate(df.columns):
        norm = _normalize(col)
        for j, pat in enumerate(TARGET_NAME_PATTERNS):
            if re.fullmatch(pat, norm, re.IGNORECASE):
                name_hits.append((col, j))  # earlier pattern = stronger
                break
    if name_hits:
        name_hits.sort(key=lambda x: x[1])
        col = name_hits[0][0]
        return col, HIGH, f"column name matches target pattern"

    # 2. Fall back: last column with low cardinality (≤ 5) wins.
    candidates = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if 1 < nunique <= 5:
            candidates.append((col, nunique))
    if candidates:
        # Prefer the last one (conventional position) and binary > multi-class.
        candidates.sort(key=lambda x: (x[1], -list(df.columns).index(x[0])))
        col = candidates[0][0]
        return col, MEDIUM, f"low-cardinality column near end (distinct={candidates[0][1]})"

    raise ValueError("Could not detect target column; pass --target explicitly")


def _infer_target_encoding(
    series: pd.Series,
    column_pattern: ColumnPattern | None,
) -> dict[str, Any]:
    """Returns target spec keys: positive_value, negative_value, positive_int, negative_int."""
    distinct = sorted(series.dropna().unique().tolist(), key=lambda v: str(v))
    if len(distinct) == 2:
        a, b = distinct
        # Medical convention: binary integer targets coded {1, 2} treat 1=disease.
        # This is the dominant encoding in older UCI clinical datasets (ILPD,
        # Statlog Heart, Hepatitis). When the encoding really is reversed the
        # user can flip positive_value / negative_value in the YAML manually.
        try:
            int_set = {int(a), int(b)}
        except (TypeError, ValueError):
            int_set = None
        if int_set == {1, 2}:
            return {"positive_value": 1, "negative_value": 2,
                    "positive_int": 1, "negative_int": 0}

        a_lower, b_lower = str(a).lower(), str(b).lower()
        # Common positive/negative pairs
        for positives, negatives in [
            ({"1", "yes", "positive", "true", "present", "ckd", "disease", "malignant"},
             {"0", "no", "negative", "false", "absent", "notckd", "healthy", "benign"}),
        ]:
            if a_lower in negatives and b_lower in positives:
                return {"positive_value": b, "negative_value": a,
                        "positive_int": 1, "negative_int": 0}
            if a_lower in positives and b_lower in negatives:
                return {"positive_value": a, "negative_value": b,
                        "positive_int": 1, "negative_int": 0}
        # not-prefix heuristic: 'not<x>' is the negative side
        not_pair = _try_not_prefix_pair(a_lower, b_lower)
        if not_pair:
            positive_lower, negative_lower = not_pair
            positive = a if a_lower == positive_lower else b
            negative = a if a_lower == negative_lower else b
            return {"positive_value": positive, "negative_value": negative,
                    "positive_int": 1, "negative_int": 0}
        # Default: assume larger value = positive (1 > 0, "Yes" > "No" lexicographically)
        positive, negative = (b, a) if str(b) > str(a) else (a, b)
        return {"positive_value": positive, "negative_value": negative,
                "positive_int": 1, "negative_int": 0}

    if len(distinct) > 2 and all(_looks_int(pd.Series([d])) for d in distinct):
        # Multi-class integer (e.g., UCI heart num 0..4) — propose binarization
        return {"positive_value": 1, "negative_value": 0,
                "positive_int": 1, "negative_int": 0,
                "_binarize_mapping": {int(d): (1 if int(d) > 0 else 0) for d in distinct}}

    if len(distinct) > 2:
        # Multi-class STRING target (e.g. RiskLevel: low/mid/high risk).
        # Identify the negative class by clinical-name pattern; everything
        # else collapses to positive. This handles the common case where the
        # baseline class ("low", "normal", "healthy", "absent") is unambiguous
        # but the "positive" side has multiple severity grades.
        negative_hints = re.compile(
            r"\b(low|normal|healthy|benign|negative|absent|no|none|mild)\b",
            re.IGNORECASE,
        )
        positive_hints = re.compile(
            r"\b(high|severe|positive|death|died|disease|abnormal|critical|malignant)\b",
            re.IGNORECASE,
        )
        negatives = [v for v in distinct if negative_hints.search(str(v))]
        positives = [v for v in distinct if positive_hints.search(str(v))]
        if negatives and len(negatives) < len(distinct):
            neg_set = set(negatives)
            mapping = {v: (0 if v in neg_set else 1) for v in distinct}
            return {"positive_value": [v for v in distinct if v not in neg_set][0],
                    "negative_value": negatives[0],
                    "positive_int": 1, "negative_int": 0,
                    "_binarize_mapping": mapping}
        if positives and len(positives) < len(distinct):
            pos_set = set(positives)
            mapping = {v: (1 if v in pos_set else 0) for v in distinct}
            return {"positive_value": positives[0],
                    "negative_value": [v for v in distinct if v not in pos_set][0],
                    "positive_int": 1, "negative_int": 0,
                    "_binarize_mapping": mapping}
        # No hint matched — fall through to the dumb default.

    return {"positive_value": distinct[-1], "negative_value": distinct[0],
            "positive_int": 1, "negative_int": 0}


def _classify_column(
    df: pd.DataFrame,
    col: str,
    target: str,
) -> tuple[str, str, str]:
    """Returns (attribute_type, confidence, reason)."""
    series = df[col]
    n = len(series.dropna())
    distinct = int(series.nunique(dropna=True))

    if distinct == n and n > 50:
        # Identifier heuristic only fires for non-float columns. Continuous
        # floats (lab values, biomarkers, signal-processing features) routinely
        # have one unique value per row by nature — they aren't IDs.
        if not pd.api.types.is_float_dtype(series):
            return "identifier", HIGH, "distinct == row count"

    if distinct == 2:
        return "binary", HIGH, "exactly 2 distinct values"

    if pd.api.types.is_numeric_dtype(series):
        if _looks_int(series) and distinct <= 8:
            return "ordinal", MEDIUM, f"integer-valued with {distinct} distinct values"
        return "numeric", HIGH, "continuous numeric column"

    # Object dtype
    if distinct <= 20:
        return "nominal", HIGH, f"{distinct} distinct string values"
    return "nominal", LOW, f"high-cardinality string column ({distinct} distinct values)"


def _propose_mapping(series: pd.Series) -> tuple[dict | None, str, str]:
    """Try to match a series of string values against the known-binary table.

    The known-binary table stores template keys in **lowercase**. When a
    pattern matches we rebuild the mapping using the actual cased values
    that appear in the data, so the resulting ``mapping:`` block fires
    correctly during cleaning regardless of source casing.
    """
    if pd.api.types.is_numeric_dtype(series):
        return None, HIGH, ""

    # Map lowercased → actual cased value present in the data.
    cased_by_lower: dict[str, Any] = {}
    for v in series.dropna().unique():
        cased_by_lower.setdefault(str(v).strip().lower(), v)
    distinct = set(cased_by_lower.keys())

    for known_set, mapping_template, label in KNOWN_BINARY_MAPPINGS:
        if distinct == known_set:
            out: dict = {}
            for lower_key, target_int in mapping_template.items():
                actual = cased_by_lower.get(lower_key)
                if actual is None:
                    continue
                out[actual] = target_int
            if len(out) == 2:
                return out, HIGH, f"matches {label} pattern"

    if len(distinct) == 2:
        # Honest fall-through: try a 'not-' prefix heuristic before giving up.
        a, b = sorted(distinct)
        # If one value is 'not<other>' or 'no<other>' that's the negative.
        not_pair = _try_not_prefix_pair(a, b)
        if not_pair:
            positive_lower, negative_lower = not_pair
            return {
                cased_by_lower[positive_lower]: 1,
                cased_by_lower[negative_lower]: 0,
            }, HIGH, "not-prefix heuristic ('not<x>' is the negative)"
        return {cased_by_lower[a]: 0, cased_by_lower[b]: 1}, LOW, \
            f"binary string values {sorted(distinct)} but no known pattern"
    return None, HIGH, ""


def _try_not_prefix_pair(a: str, b: str) -> tuple[str, str] | None:
    """Return (positive_lower, negative_lower) if one looks like 'not<other>'."""
    for short, long in [(a, b), (b, a)]:
        if not long.startswith("not"):
            continue
        tail = long[3:].lstrip("_- ").lower()
        if tail == short.lower():
            return short, long
    # Also handle 'no_x' / 'no-x' / 'nox' patterns
    for short, long in [(a, b), (b, a)]:
        if long.lower() in {"no", short.lower()}:
            continue
        if long.lower().startswith("no") and long.lower()[2:].lstrip("_- ") == short.lower():
            return short, long
    return None


def _propose_imputation(series: pd.Series) -> dict | None:
    if series.isna().sum() == 0:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return {"strategy": "median"}
    return {"strategy": "constant", "value": "Unknown"}


def _propose_hierarchies(profile_columns: list[dict]) -> list[dict]:
    hierarchies: list[dict] = []
    seen: set[str] = set()
    for col in profile_columns:
        h_key = col.get("_hierarchy_key")
        if not h_key or h_key in seen:
            continue
        seen.add(h_key)
        hierarchies.append({
            "source_column": col["source"],
            "new_column": h_key,
            "bands": STANDARD_HIERARCHIES[h_key],
        })
    return hierarchies


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

def infer_profile(
    csv_path: str | Path,
    name: str | None = None,
    target: str | None = None,
    source_path_for_yaml: str | None = None,
    use_llm: bool = False,
    llm_cache_dir: Path | None = None,
) -> InferenceResult:
    """Auto-generate a profile dict from a raw CSV.

    Parameters
    ----------
    use_llm
        When ``True``, columns still tagged LOW-confidence after the
        deterministic pass are sent to Gemini for role/description suggestions.
        Requires ``GEMINI_API_KEY`` in the environment and the
        ``google-generativeai`` package; silently no-ops if either is missing.
    llm_cache_dir
        Directory under which ``.llm_cache.json`` lives. Defaults to the CSV's
        parent directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    delimiter, delim_conf = _sniff_delimiter(csv_path)
    df = pd.read_csv(csv_path, sep=delimiter, na_values=["?", "", " ", "N/A", "n/a", "NA"], keep_default_na=True)
    # Mirror the cleaner's strip_whitespace pass so inference sees the same
    # values the build will see (otherwise trailing tabs / spaces produce
    # spurious "third value" effects in target detection and domain inference).
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    decisions: list[Decision] = [
        Decision("<file>", "delimiter", delimiter, delim_conf,
                 f"sniffed from header line"),
    ]

    name = name or csv_path.stem.lower().replace("-", "_").replace(" ", "_")
    decisions.append(Decision("<file>", "name", name, HIGH if name == csv_path.stem else MEDIUM,
                              "derived from filename" if not target else "user-supplied"))

    target_col, target_conf, target_reason = _detect_target(df, target)
    decisions.append(Decision(target_col, "target", target_col, target_conf, target_reason))

    target_pattern = _match_pattern(target_col)
    target_spec = _infer_target_encoding(df[target_col], target_pattern)
    binarize_map = target_spec.pop("_binarize_mapping", None)

    # ─── Per-column inference ─────────────────────────────────────────────
    profile_columns: list[dict] = []
    for col in df.columns:
        entry: dict[str, Any] = {"source": col}
        norm = _normalize(col)
        pattern = _match_pattern(col)

        attr_type, type_conf, type_reason = _classify_column(df, col, target_col)
        entry["type"] = attr_type
        decisions.append(Decision(col, "type", attr_type, type_conf, type_reason))

        # role: from keyword dict; identifier from type; numeric-range fallback
        fingerprint: RangeFingerprint | None = None
        if attr_type == "identifier":
            entry["role"] = None  # identifiers don't get a role
            entry["drop"] = True
            decisions.append(Decision(col, "drop", True, HIGH, "identifier column"))
        elif pattern and pattern.role:
            entry["role"] = pattern.role
            decisions.append(Decision(col, "role", pattern.role, HIGH, "keyword match"))
        else:
            # Keyword miss — try a numeric value-range fingerprint before giving up.
            if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
                fingerprint = _fingerprint_numeric(df[col])
            if fingerprint is not None:
                entry["role"] = fingerprint.role
                decisions.append(Decision(
                    col, "role", fingerprint.role, MEDIUM,
                    f"numeric value-range fingerprint matched {fingerprint.label!r} — verify",
                ))
            else:
                entry["role"] = None
                if col != target_col:
                    decisions.append(Decision(col, "role", None, LOW,
                                              "no keyword match — please assign role manually"))

        # Age-unit sanity check: years should be roughly in [0, 150]
        if entry.get("role") == "age" and pd.api.types.is_numeric_dtype(df[col]):
            ages = pd.to_numeric(df[col], errors="coerce").dropna()
            if not ages.empty:
                max_age = float(ages.max())
                if max_age > 150 and max_age < 50000:
                    entry["scale_factor"] = 1.0 / 365.25
                    decisions.append(Decision(
                        col, "scale_factor", entry["scale_factor"], LOW,
                        f"age max={max_age:.0f} suggests DAYS — auto-applied scale_factor 1/365.25 "
                        "(verify before build, or remove if values are actually years)",
                    ))
                elif max_age > 13 and max_age <= 150:
                    pass  # plausible years
                elif max_age <= 13:
                    decisions.append(Decision(
                        col, "scale_factor", None, LOW,
                        f"age max={max_age:.0f} is too low for years — may be a BRFSS-style "
                        "age-group code (1-13). Consider adding mapping: {1: 21, 2: 27, ...}",
                    ))

        # Gender-encoding sanity check: warn if gender values aren't the expected 0/1 or Male/Female
        if entry.get("role") == "gender":
            distinct_vals = set(str(v).strip() for v in df[col].dropna().unique())
            normalized = {v.lower() for v in distinct_vals}
            if normalized in ({"0", "1"}, {"male", "female"}, {"m", "f"}):
                pass  # already standard
            elif normalized == {"1", "2"}:
                decisions.append(Decision(
                    col, "mapping", {1: 0, 2: 1}, LOW,
                    "gender values {1, 2} are non-standard — verify which is Male before mapping "
                    "(Kaggle cardio uses 1=Female, 2=Male; some surveys use 1=Male, 2=Female)",
                ))
                # Don't auto-insert mapping because direction is ambiguous; just flag it.
            elif len(distinct_vals) > 2:
                decisions.append(Decision(
                    col, "drop_rows_where", None, LOW,
                    f"gender has {len(distinct_vals)} distinct values {sorted(distinct_vals)} — "
                    "consider drop_rows_where to remove sparse minorities (e.g. 'Other')",
                ))

        # description: template or REVIEW
        if pattern:
            entry["description"] = pattern.description
            decisions.append(Decision(col, "description", "template", HIGH, "keyword match"))
        elif col == target_col:
            entry["description"] = "Engine-managed target column."
            decisions.append(Decision(col, "description", "target", HIGH, "target column"))
        elif fingerprint is not None:
            entry["description"] = fingerprint.description
            decisions.append(Decision(col, "description", "fingerprint", MEDIUM, "value-range fingerprint"))
        else:
            entry["description"] = f"TODO — describe {col}"
            decisions.append(Decision(col, "description", "TODO", LOW, "no keyword match"))

        # mapping: from value patterns
        mapping, map_conf, map_reason = _propose_mapping(df[col])
        if mapping is not None:
            entry["mapping"] = mapping
            decisions.append(Decision(col, "mapping", mapping, map_conf, map_reason))

        # multi-class target binarization mapping
        if col == target_col and binarize_map is not None:
            entry["mapping"] = binarize_map
            decisions.append(Decision(col, "mapping", binarize_map, HIGH,
                                      "multi-class target — collapsed to 0/1 by sign"))

        # domain whitelist for small nominal columns — but skip the target
        # column because its values get replaced by the engine-managed label
        # encoding, so validating the raw domain post-mapping would fail.
        if (col != target_col
                and attr_type in {"nominal", "ordinal"}
                and df[col].nunique(dropna=True) <= 10):
            domain = sorted(df[col].dropna().unique().tolist(), key=lambda v: str(v))
            entry["domain"] = domain
            decisions.append(Decision(col, "domain", domain, HIGH,
                                      f"{len(domain)} distinct values whitelisted"))

        # imputation when nulls exist
        imputation = _propose_imputation(df[col])
        if imputation:
            entry["imputation"] = imputation
            decisions.append(Decision(col, "imputation", imputation, MEDIUM,
                                      f"{int(df[col].isna().sum())} nulls present"))

        # remember the column-level hierarchy hint for the hierarchies block
        if pattern and pattern.hierarchy_key:
            entry["_hierarchy_key"] = pattern.hierarchy_key
        elif fingerprint is not None and fingerprint.hierarchy_key:
            entry["_hierarchy_key"] = fingerprint.hierarchy_key

        profile_columns.append(entry)

    # ─── LLM enrichment for remaining LOW-confidence columns ─────────────
    if use_llm:
        # Import lazily so the module stays usable without google-generativeai.
        from disease_warehouse.core.llm_enrich import enrich_low_confidence

        # Skip columns already marked drop:true — there's no point asking the
        # LLM for a role/description we'll never use.
        dropped_cols = {
            entry["source"] for entry in profile_columns if entry.get("drop")
        }
        low_conf_cols = sorted({
            d.column for d in decisions
            if d.is_review()
            and d.column != "<file>"
            and d.column != target_col
            and d.column not in dropped_cols
        })
        cache_dir = Path(llm_cache_dir) if llm_cache_dir else csv_path.parent
        suggestions, status = enrich_low_confidence(
            df,
            low_conf_cols,
            target_col=target_col,
            csv_name=csv_path.name,
            cache_dir=cache_dir,
        )
        decisions.append(Decision("<file>", "llm", status, HIGH, status))

        # Apply suggestions: only fill in fields the deterministic pass left empty
        # or marked LOW. Never overwrite a HIGH/MEDIUM decision.
        for entry in profile_columns:
            col = entry["source"]
            sug = suggestions.get(col)
            if sug is None:
                continue
            llm_conf = {"HIGH": HIGH, "MEDIUM": MEDIUM, "LOW": LOW}[sug.confidence]

            if sug.role and not entry.get("role"):
                entry["role"] = sug.role
                decisions.append(Decision(
                    col, "role", sug.role, llm_conf,
                    f"LLM suggestion ({sug.confidence}): {sug.reason}",
                ))
            if sug.description and entry.get("description", "").startswith("TODO"):
                entry["description"] = sug.description
                decisions.append(Decision(
                    col, "description", "llm", llm_conf,
                    f"LLM suggestion ({sug.confidence}): {sug.reason}",
                ))
            if sug.mapping and "mapping" not in entry:
                # Only attach the mapping if every key actually appears in the
                # raw distinct values — otherwise it'd be a no-op or trip
                # domain validation downstream.
                raw_vals = {str(v) for v in df[col].dropna().unique()}
                clean = {k: v for k, v in sug.mapping.items() if k in raw_vals}
                if clean:
                    entry["mapping"] = clean
                    decisions.append(Decision(
                        col, "mapping", clean, llm_conf,
                        f"LLM suggestion ({sug.confidence}): {sug.reason}",
                    ))

    # ─── Build the profile dict ───────────────────────────────────────────
    hierarchies = _propose_hierarchies(profile_columns)
    # Strip the private hint key now that hierarchies have been built
    for entry in profile_columns:
        entry.pop("_hierarchy_key", None)
        # Drop None roles for cleaner YAML
        if entry.get("role") is None:
            entry.pop("role", None)

    if source_path_for_yaml is None:
        source_path_for_yaml = csv_path.as_posix()

    profile: dict[str, Any] = {
        "name": name,
        "display_name": name.replace("_", " ").title(),
        "domain": "general",
        "description": f"Auto-generated profile for {csv_path.name}.\nReview '# REVIEW:' markers before running build.",
        "source": {
            "path": source_path_for_yaml,
            "format": "csv",
        },
        "target": {
            "column": target_col,
            "rename_to": "label",
            **{k: v for k, v in target_spec.items() if not k.startswith("_")},
        },
        "columns": profile_columns,
        "cleaning": {
            "strip_whitespace": True,
            "drop_rows_where": [],
        },
        "hierarchies": hierarchies,
    }
    if delimiter != ",":
        profile["source"]["delimiter"] = delimiter
    if df[target_col].isna().sum() == 0 and not any(c.get("imputation") for c in profile_columns):
        pass  # no na_values customization needed

    return InferenceResult(
        profile=profile,
        decisions=decisions,
        delimiter=delimiter,
    )
