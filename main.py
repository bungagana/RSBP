#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_xai_complete.py
Full pipeline dengan analisis fitur detail
"""

import os
import sys
import math
import datetime
import warnings
warnings.filterwarnings("ignore")

import joblib
import json
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# optional libs
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD
    RDFLIB_AVAILABLE = True
except Exception:
    RDFLIB_AVAILABLE = False

try:
    from owlready2 import get_ontology, Thing, DataProperty
    OWLREADY_AVAILABLE = True
except Exception:
    OWLREADY_AVAILABLE = False

# Neo4j (py2neo)
try:
    from py2neo import Graph as NeoGraph, Node
    PY2NEO_AVAILABLE = True
except Exception:
    PY2NEO_AVAILABLE = False

# ---------------- CONFIG ----------------
CSV_PATH = os.getenv("CSV_PATH", "Mali_Cohort_Study.csv")
OUT_DIR = os.getenv("OUT_DIR", "./hybrid_xai_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Neo4j env vars (optional)
NEO4J_URI = os.getenv("NEO4J_URI", None)
NEO4J_USER = os.getenv("NEO4J_USER", None)
NEO4J_PASS = os.getenv("NEO4J_PASS", None)

TARGETS = ["total_rutf", "LOS"]  # must be present in CSV

# ---------------- utils ----------------
def now_ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(*args, level="INFO"):
    ts = now_ts()
    print(f"[{ts}] [{level}]", *args)
    sys.stdout.flush()

def make_ohe():
    # compatibility across sklearn versions
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def safe_literal(val):
    if val == "" or pd.isna(val):
        return None
    if not RDFLIB_AVAILABLE:
        return val
    try:
        f = float(val)
        if float(int(f)) == f:
            return Literal(int(f), datatype=XSD.integer)
        else:
            return Literal(f, datatype=XSD.double)
    except Exception:
        return Literal(str(val))

# ---------------- 1) load & basic cleaning ----------------
log("1) Loading dataset...", "INFO")
if not os.path.exists(CSV_PATH):
    log(f"CSV not found at {CSV_PATH}. Set CSV_PATH environment variable or put file there.", "ERROR")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
original_shape = df.shape
log(f"Dataset shape: {original_shape}", "INFO")
log(f"Original columns: {list(df.columns)}", "INFO")

# drop columns with >80% missing
initial_columns = df.shape[1]
df = df.dropna(axis=1, thresh=0.2 * len(df))
columns_after_missing_drop = df.shape[1]
columns_dropped_missing = initial_columns - columns_after_missing_drop
log(f"After drop >80% missing: columns = {df.shape[1]} (dropped {columns_dropped_missing} columns)", "INFO")

# ensure targets present
for t in TARGETS:
    if t not in df.columns:
        log(f"Target '{t}' missing — abort.", "ERROR"); sys.exit(1)

# conservative list of leakage columns to drop (customize if needed)
leakage_candidates = [
    'rutf_cured', 'non_response', 'defaulted', 'missed_0_visit',
    'recovered_los', 'avg_daily_weight_gain', 'muac_gain_velo',
    'recovered', 'outcome', 'discharge_status',
    'weight_gain_cured', 'rutf_received', 'days_to_recovery',
    'AnonID', 'anonid', 'id'
]
to_drop = [c for c in leakage_candidates if c in df.columns]
if to_drop:
    log(f"Dropping leakage/ID cols: {to_drop}", "WARN")
    df = df.drop(columns=to_drop)

# keep rows that have at least one target
initial_rows = len(df)
df = df[~(df[TARGETS[0]].isnull() & df[TARGETS[1]].isnull())].reset_index(drop=True)
rows_after_target_clean = len(df)
rows_dropped_target_missing = initial_rows - rows_after_target_clean
log(f"Rows after removing both-target-missing: {len(df)} (dropped {rows_dropped_target_missing} rows)", "INFO")

# ---------------- 2) feature engineering ----------------
log("2) Feature engineering...", "INFO")
engineered_features = []
if {'adm_kg', 'adm_cm'}.issubset(df.columns):
    df['bmi_proxy'] = df['adm_kg'] / ((df['adm_cm'] / 100).replace(0, np.nan) ** 2 + 1e-9)
    engineered_features.append('bmi_proxy')
if {'adm_muac', 'adm_age'}.issubset(df.columns):
    df['muac_age_ratio'] = df['adm_muac'] / (df['adm_age'] + 1e-9)
    engineered_features.append('muac_age_ratio')
if {'adm_muac', 'adm_whz06'}.issubset(df.columns):
    df['severe_combo'] = ((df['adm_muac'] < 115).astype(int) + (df['adm_whz06'] < -3).astype(int))
    engineered_features.append('severe_combo')
if {'adm_muac', 'adm_age'}.issubset(df.columns):
    df['muac_x_age'] = df['adm_muac'] * df['adm_age']
    engineered_features.append('muac_x_age')

log(f"Engineered features: {engineered_features}", "INFO")

# ---------------- 3) prepare features & split ----------------
log("3) Prepare features & train/test split...", "INFO")
features = [c for c in df.columns if c not in TARGETS]

# drop very high-cardinality string columns (likely free-text)
drop_candidates = [c for c in features if df[c].dtype == 'object' and df[c].nunique() > 200]
if drop_candidates:
    log(f"Dropping high-cardinality free-text cols: {drop_candidates}", "WARN")
    features = [c for c in features if c not in drop_candidates]

X = df[features].copy()
y = df[TARGETS].copy()

# Detailed feature analysis
log("=== DETAILED FEATURE ANALYSIS ===", "INFO")
log(f"Total features available: {len(features)}", "INFO")
log(f"Features list: {features}", "INFO")

# Analyze data types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [c for c in features if c not in numeric_cols]

log(f"Numeric features ({len(numeric_cols)}): {numeric_cols}", "INFO")
log(f"Categorical features ({len(cat_cols)}): {cat_cols}", "INFO")

# Check for missing values
missing_summary = X.isnull().sum()
high_missing_features = missing_summary[missing_summary > 0]
if len(high_missing_features) > 0:
    log(f"Features with missing values: {dict(high_missing_features)}", "WARN")

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
log(f"Data split: Train {X_train.shape} -> {len(X_train)} rows, Test {X_test.shape} -> {len(X_test)} rows", "OK")
log(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features", "INFO")
log(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features", "INFO")

# ---------------- 4) preprocessing ----------------
log("4) Building preprocessing pipeline...", "INFO")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", make_ohe())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", cat_transformer, cat_cols)
], remainder="drop")

preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# feature names
feature_names = numeric_cols.copy()
if len(cat_cols) > 0:
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names.extend(ohe_names)
        log(f"One-hot encoded features added: {len(ohe_names)}", "INFO")
    except Exception as e:
        log(f"Error getting OHE names: {e}", "WARN")
        ohe_names = []

log(f"Final processed feature count: {len(feature_names)}", "INFO")
log(f"First 20 feature names: {feature_names[:20]}", "INFO")

# Save feature analysis to file
feature_analysis = {
    "original_dataset_shape": original_shape,
    "columns_dropped_missing": columns_dropped_missing,
    "rows_dropped_target_missing": rows_dropped_target_missing,
    "leakage_columns_dropped": to_drop,
    "high_cardinality_dropped": drop_candidates,
    "engineered_features": engineered_features,
    "final_features_count": len(features),
    "numeric_features": numeric_cols,
    "categorical_features": cat_cols,
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "processed_feature_names": feature_names
}

with open(os.path.join(OUT_DIR, "feature_analysis.json"), "w") as f:
    json.dump(feature_analysis, f, indent=2)

log("Saved feature analysis to feature_analysis.json", "OK")

# ---------------- 5) model selection & train ----------------
log("5) Model selection & training...", "INFO")
if CATBOOST_AVAILABLE:
    log("Using CatBoostRegressor", "OK")
    base = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, verbose=0, random_state=42)
elif XGBOOST_AVAILABLE:
    log("Using XGBoost", "OK")
    base = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, verbosity=0, n_jobs=1)
else:
    log("CatBoost/XGBoost not found — using RandomForest fallback", "WARN")
    base = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)

multi = MultiOutputRegressor(base, n_jobs=-1)
multi.fit(X_train_proc, y_train.values)
log("Model training completed.", "OK")

# Analyze feature importance per target
log("=== FEATURE IMPORTANCE ANALYSIS ===", "INFO")
for i, target in enumerate(TARGETS):
    estimator = multi.estimators_[i]
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]  # Top 10 features
        log(f"Top 10 features for {target}:", "INFO")
        for idx in top_indices:
            if idx < len(feature_names):
                log(f"  {feature_names[idx]}: {importances[idx]:.4f}", "INFO")

# save artifacts
joblib.dump({"model": multi, "preprocessor": preprocessor, "feature_names": feature_names}, 
            os.path.join(OUT_DIR, "multi_model_joblib.pkl"))
log("Saved model + preprocessor to", OUT_DIR, "OK")

# ---------------- 6) evaluate ----------------
log("6) Evaluating on test set...", "INFO")
y_pred = multi.predict(X_test_proc)
y_pred_rutf = np.clip(y_pred[:, 0], 0, None)
y_pred_los = np.clip(y_pred[:, 1], 0, None)

for i, target in enumerate(TARGETS):
    r2 = r2_score(y_test[target], y_pred[:, i])
    mae = mean_absolute_error(y_test[target], y_pred[:, i])
    rmse = math.sqrt(mean_squared_error(y_test[target], y_pred[:, i]))
    log(f"{target:<10} | R2: {r2:.4f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}", "OK")

# create pred_df
X_test_raw = X_test.reset_index(drop=True).copy()
pred_df = X_test_raw.copy()
pred_df["pred_total_rutf"] = y_pred_rutf
pred_df["pred_LOS"] = y_pred_los
pred_df.to_csv(os.path.join(OUT_DIR, "predictions_test_set.csv"), index=False)
log("Saved test predictions to", os.path.join(OUT_DIR, "predictions_test_set.csv"), "OK")

# ---------------- 7) SHAP explanations (optional) ----------------
shap_values = None
if SHAP_AVAILABLE:
    log("7) Computing SHAP values (TreeExplainer if available)...", "INFO")
    try:
        est0 = multi.estimators_[0]
        explainer = shap.TreeExplainer(est0)
        shap_values_raw = explainer.shap_values(X_test_proc)
        shap_values = shap_values_raw
        log("SHAP values computed.", "OK")
        
        # Analyze top SHAP features overall
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[0])
        else:
            shap_arr = np.array(shap_values)
            
        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
        top_shap_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        log("Top 10 features by mean |SHAP| value:", "INFO")
        for idx in top_shap_indices:
            if idx < len(feature_names):
                log(f"  {feature_names[idx]}: {mean_abs_shap[idx]:.4f}", "INFO")
                
    except Exception as e:
        log("SHAP computation failed:", e, "WARN")
        shap_values = None
else:
    log("SHAP not available — skipping SHAP.", "WARN")

def top_shap_features_for_row(row_idx: int, topk: int = 3) -> List[tuple]:
    if shap_values is None:
        try:
            est0 = multi.estimators_[0]
            if hasattr(est0, "feature_importances_"):
                imp = np.array(est0.feature_importances_)
                top_idx = np.argsort(imp)[-topk:][::-1]
                return [(feature_names[i], float(imp[i])) for i in top_idx]
        except Exception:
            return []
        return []
    if isinstance(shap_values, list):
        arr = np.array(shap_values[0])
    else:
        arr = np.array(shap_values)
    row_sv = arr[row_idx]
    abs_sv = np.abs(row_sv)
    top_idx = np.argsort(abs_sv)[-topk:][::-1]
    return [(feature_names[i], float(row_sv[i])) for i in top_idx]

# ---------------- 8) RDF/Turtle export + OWL labeling ----------------
TTL_PATH = os.path.join(OUT_DIR, "sam_ontology.ttl")
OWL_PATH = os.path.join(OUT_DIR, "sam_ontology.owl")
severity_labels = {}

if RDFLIB_AVAILABLE:
    log("8) Building RDF/Turtle ontology (rdflib)...", "INFO")
    g = Graph()
    EX = Namespace("http://example.org/sam#")
    g.bind("sam", EX)
    g.bind("rdfs", RDFS)

    for cls in ["Patient", "Admission", "Anthropometry", "Treatment", "Outcome", "Severity"]:
        g.add((EX[cls], RDF.type, RDFS.Class))

    for idx, row in pred_df.reset_index().iterrows():
        pid = f"P{idx}"
        p = EX[pid]
        g.add((p, RDF.type, EX.Patient))

        adm = EX[f"Admission_{pid}"]; g.add((adm, RDF.type, EX.Admission)); g.add((p, EX.hasAdmission, adm))
        anth = EX[f"Anthro_{pid}"]; g.add((anth, RDF.type, EX.Anthropometry)); g.add((adm, EX.hasAnthropometry, anth))
        tr = EX[f"Treatment_{pid}"]; g.add((tr, RDF.type, EX.Treatment)); g.add((p, EX.receivedTreatment, tr))

        for c in X_test_raw.columns:
            v = row[c]
            if pd.isna(v) or v == "":
                continue
            lit = safe_literal(v)
            if lit is None:
                continue
            if str(c).startswith("adm_") or c in ("adm_age", "adm_sex", "age_cat_1", "age_cat_2"):
                prop = EX[str(c)]; g.add((prop, RDF.type, RDF.Property)); g.add((adm, prop, lit))
            elif c in ("adm_muac", "adm_kg", "adm_cm", "adm_whz06", "adm_waz06", "adm_haz06"):
                prop = EX[str(c)]; g.add((prop, RDF.type, RDF.Property)); g.add((anth, prop, lit))
            else:
                prop = EX[str(c)]; g.add((prop, RDF.type, RDF.Property)); g.add((p, prop, lit))

        g.add((EX['pred_total_rutf'], RDF.type, RDF.Property)); g.add((tr, EX['pred_total_rutf'], Literal(float(row['pred_total_rutf']))))
        g.add((EX['pred_LOS'], RDF.type, RDF.Property)); g.add((tr, EX['pred_LOS'], Literal(float(row['pred_LOS']))))

    g.add((EX['severity_rule_muac'], RDFS.comment, Literal("Severity: MUAC < 115 & WHZ < -3 -> Severe")))
    g.serialize(destination=TTL_PATH, format="turtle")
    log("TTL saved to", TTL_PATH, "OK")
else:
    log("rdflib not available — skipping TTL export.", "WARN")

# OWL labeling (owlready2) or procedural fallback
if OWLREADY_AVAILABLE:
    log("Creating OWL individuals + labeling (owlready2)...", "INFO")
    onto = get_ontology("http://example.org/sam.owl")
    with onto:
        class Patient(Thing): pass
        class hasMUAC(DataProperty): pass
        class hasWHZ(DataProperty): pass
        class classifiedAs(DataProperty): pass

    for idx, row in pred_df.reset_index().iterrows():
        name = f"Patient_{idx}"
        inst = onto.Patient(name)
        try:
            if 'adm_muac' in X_test_raw.columns and not pd.isna(X_test_raw.loc[idx,'adm_muac']):
                inst.hasMUAC = [float(X_test_raw.loc[idx,'adm_muac'])]
        except:
            pass
        try:
            if 'adm_whz06' in X_test_raw.columns and not pd.isna(X_test_raw.loc[idx,'adm_whz06']):
                inst.hasWHZ = [float(X_test_raw.loc[idx,'adm_whz06'])]
        except:
            pass
        # procedural rule -> label
        try:
            muac_v = float(inst.hasMUAC[0]) if inst.hasMUAC else None
            whz_v = float(inst.hasWHZ[0]) if inst.hasWHZ else None
            lbl = "Unknown"
            if muac_v is not None and whz_v is not None:
                if muac_v < 115 and whz_v < -3:
                    lbl = "Severe"
                elif muac_v < 115 or whz_v < -2:
                    lbl = "Moderate"
                else:
                    lbl = "Mild"
            inst.classifiedAs = [lbl]
            severity_labels[name] = lbl
        except:
            severity_labels[name] = "Unknown"
    try:
        onto.save(file=OWL_PATH)
        log("OWL saved to", OWL_PATH, "OK")
    except Exception:
        log("OWL saved failed but labels computed.", "WARN")
else:
    log("owlready2 not available — using procedural severity labeling.", "WARN")
    for idx, row in pred_df.reset_index().iterrows():
        name = f"Patient_{idx}"
        muac_v = None; whz_v = None
        try:
            if 'adm_muac' in X_test_raw.columns and not pd.isna(X_test_raw.loc[idx,'adm_muac']):
                muac_v = float(X_test_raw.loc[idx,'adm_muac'])
        except:
            muac_v = None
        try:
            if 'adm_whz06' in X_test_raw.columns and not pd.isna(X_test_raw.loc[idx,'adm_whz06']):
                whz_v = float(X_test_raw.loc[idx,'adm_whz06'])
        except:
            whz_v = None
        lbl = "Unknown"
        if muac_v is not None and whz_v is not None:
            if muac_v < 115 and whz_v < -3:
                lbl = "Severe"
            elif muac_v < 115 or whz_v < -2:
                lbl = "Moderate"
            else:
                lbl = "Mild"
        severity_labels[name] = lbl

# ---------------- 9) Neo4j ingestion & similarity (optional) ----------------
neo_similar_cases = {}
if PY2NEO_AVAILABLE and NEO4J_URI and NEO4J_USER and NEO4J_PASS:
    try:
        log("9) Connecting to Neo4j...", "INFO")
        neo = NeoGraph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        log("Connected to Neo4j — ingesting patients...", "OK")
        # clear graph (CAUTION: will delete all data in DB)
        try:
            neo.run("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        for idx, row in pred_df.reset_index().iterrows():
            pid = f"P{idx}"
            pnode = Node("Patient", pid=pid, pred_RUTF=float(row['pred_total_rutf']), pred_LOS=float(row['pred_LOS']))
            if 'adm_muac' in X_test_raw.columns and not pd.isna(X_test_raw.loc[idx,'adm_muac']):
                pnode['muac'] = float(X_test_raw.loc[idx,'adm_muac'])
            if 'adm_whz06' in X_test_raw.columns and not pd.isna(X_test_raw.loc[idx,'adm_whz06']):
                pnode['whz'] = float(X_test_raw.loc[idx,'adm_whz06'])
            neo.create(pnode)

        # similarity: find patients with MUAC within +/-5 mm
        for idx, row in pred_df.reset_index().iterrows():
            pid = f"P{idx}"
            try:
                muac_val = float(row['adm_muac']) if 'adm_muac' in row and not pd.isna(row['adm_muac']) else None
                if muac_val is None:
                    neo_similar_cases[pid] = []
                    continue
                q = f"""
                MATCH (p:Patient {{pid:'{pid}'}}), (o:Patient)
                WHERE o.pid <> '{pid}' AND exists(o.muac) AND abs(toFloat(o.muac)-{muac_val}) <= 5
                RETURN o.pid as pid, o.pred_RUTF as rutf, o.pred_LOS as los LIMIT 5
                """
                res = list(neo.run(q))
                neo_similar_cases[pid] = [r['pid'] for r in res]
            except Exception:
                neo_similar_cases[pid] = []
        log("Neo4j ingestion & similarity done.", "OK")
    except Exception as e:
        log("Neo4j step failed: " + str(e), "ERROR")
        neo_similar_cases = {}
else:
    log("Neo4j not configured or py2neo not installed — skipping Neo4j part.", "WARN")

# ---------------- 10) Caption generator (humanized) ----------------
log("10) Generating human-friendly captions...", "INFO")

# helper rule functions
def muac_severity(muac):
    try:
        muac = float(muac)
    except:
        return None
    if muac < 115:
        return "Severe"
    if muac < 125:
        return "Moderate"
    return "Mild"

def whz_severity(whz):
    try:
        whz = float(whz)
    except:
        return None
    if whz < -3:
        return "Severe"
    if whz < -2:
        return "Moderate"
    return "Mild"

def triage_action(muac, oedema, whz, predicted_rutf):
    try:
        oed = int(oedema) if oedema is not None and not pd.isna(oedema) else 0
    except:
        oed = 0
    if oed == 1:
        return "⚠️ Oedema terdeteksi — rujuk/beri perawatan segera."
    if muac is not None:
        if float(muac) < 115 or (whz is not None and float(whz) < -3):
            return "🚨 Kriteria SAM: alokasikan RUTF segera dan monitor ketat."
    if predicted_rutf is not None and predicted_rutf > 150:
        return "🔴 Prediksi kebutuhan tinggi — prioritaskan alokasi."
    return "🟢 Pantau reguler; alokasikan sesuai ketersediaan."

def extract_top_features(idx, topk=3):
    items = top_shap_features_for_row(idx, topk)
    res = []
    for f, v in items:
        # prefer raw value if column exists
        if f in X_test_raw.columns:
            try:
                val = X_test_raw.loc[idx, f]
                res.append(f"{f}={val}")
            except:
                res.append(f"{f}")
        else:
            res.append(f"{f}")
    return res

pred_df['caption'] = ""
pred_df['severity'] = ""
pred_df['priority_score'] = (pred_df['pred_total_rutf'] - pred_df['pred_total_rutf'].min()) / (pred_df['pred_total_rutf'].max() - pred_df['pred_total_rutf'].min() + 1e-9)
top_idxs = pred_df.sort_values('priority_score', ascending=False).head(20).index.tolist()

human_rows = []
for i in pred_df.index:
    pid = f"P{i}"
    prutf = float(pred_df.loc[i, 'pred_total_rutf'])
    plos = float(pred_df.loc[i, 'pred_LOS'])
    muac = pred_df.loc[i, 'adm_muac'] if 'adm_muac' in pred_df.columns else None
    whz = pred_df.loc[i, 'adm_whz06'] if 'adm_whz06' in pred_df.columns else None
    age = pred_df.loc[i, 'adm_age'] if 'adm_age' in pred_df.columns else None
    oedema = pred_df.loc[i, 'adm_oedema_YN'] if 'adm_oedema_YN' in pred_df.columns else 0

    sev = severity_labels.get(f"Patient_{i}", None)
    if not sev:
        # fallback: compute from muac/whz
        sev_muac = muac_severity(muac) if muac is not None else None
        sev_whz = whz_severity(whz) if whz is not None else None
        if sev_muac == "Severe" or sev_whz == "Severe" or (oedema is not None and int(oedema) == 1):
            sev = "Severe"
        elif sev_muac == "Moderate" or sev_whz == "Moderate":
            sev = "Moderate"
        else:
            sev = "Mild"
    top_feats = extract_top_features(i, topk=3)
    sim_cases = neo_similar_cases.get(pid, []) if neo_similar_cases else []
    sim_txt = ", ".join(sim_cases[:3]) if sim_cases else "tidak ada kasus historis sangat mirip"

    reason_parts = []
    if muac is not None and not pd.isna(muac):
        try:
            mu = float(muac)
            if mu < 115:
                reason_parts.append(f"MUAC rendah {int(mu)} mm (kriteria SAM)")
            elif mu < 125:
                reason_parts.append(f"MUAC {int(mu)} mm (malnutrisi sedang)")
            else:
                reason_parts.append(f"MUAC {int(mu)} mm")
        except:
            pass
    if whz is not None and not pd.isna(whz):
        try:
            wv = float(whz)
            if wv < -3:
                reason_parts.append(f"WHZ {wv:.1f} (severe)")
            elif wv < -2:
                reason_parts.append(f"WHZ {wv:.1f} (moderate)")
            else:
                reason_parts.append(f"WHZ {wv:.1f}")
        except:
            pass
    if oedema is not None and not pd.isna(oedema):
        try:
            if int(oedema) == 1:
                reason_parts.insert(0, "oedema terdeteksi (kondisi kritis)")
        except:
            pass
    if age is not None and not pd.isna(age):
        try:
            reason_parts.append(f"usia {int(float(age))} bulan")
        except:
            pass

    reason_text = "; ".join(reason_parts) if reason_parts else (", ".join(top_feats) if top_feats else "indikator klinis")
    action = triage_action(muac, oedema, whz, prutf)

    human_caption = (f"Pasien {pid}: diprediksi membutuhkan ~{int(round(prutf))} sachet RUTF selama ~{int(round(plos))} hari. "
                     f"Alasan: {reason_text}. Fitur penting: {', '.join(top_feats) if top_feats else '-'}."
                     f" Rekomendasi: {action} (bukti: {sim_txt}).")

    pred_df.at[i, 'caption'] = human_caption
    pred_df.at[i, 'severity'] = sev

# save captions CSV
pred_df.to_csv(os.path.join(OUT_DIR, "predictions_with_captions.csv"), index=False)
log("Saved captions CSV to", os.path.join(OUT_DIR, "predictions_with_captions.csv"), "OK")

# also save a brief guidance file for high-need patients
guid_list = []
for i in pred_df.sort_values('pred_total_rutf', ascending=False).head(50).index.tolist():
    guid_list.append(pred_df.loc[i, 'caption'])
with open(os.path.join(OUT_DIR, "predictions_guidance.txt"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(guid_list))
log("Saved guidance text to", os.path.join(OUT_DIR, "predictions_guidance.txt"), "OK")

# sample prints
log("\n=== Sample captions (top 5 by predicted RUTF) ===")
for i in pred_df.sort_values('pred_total_rutf', ascending=False).head(5).index.tolist():
    log(f"P{i}: {pred_df.loc[i,'caption']}", "INFO")

# Final summary
log("\n=== FINAL SUMMARY ===", "INFO")
log(f"Original dataset: {original_shape[0]} rows, {original_shape[1]} columns", "INFO")
log(f"Final dataset: {len(df)} rows, {len(df.columns)} columns", "INFO")
log(f"Columns dropped (missing): {columns_dropped_missing}", "INFO")
log(f"Rows dropped (target missing): {rows_dropped_target_missing}", "INFO")
log(f"Leakage columns dropped: {len(to_drop)}", "INFO")
log(f"High-cardinality columns dropped: {len(drop_candidates)}", "INFO")
log(f"Engineered features: {len(engineered_features)}", "INFO")
log(f"Final features used: {len(features)}", "INFO")
log(f"Training set: {X_train.shape[0]} samples", "INFO")
log(f"Test set: {X_test.shape[0]} samples", "INFO")
log(f"Processed features after encoding: {len(feature_names)}", "INFO")

log("Pipeline finished. Outputs in folder:", OUT_DIR, "OK")