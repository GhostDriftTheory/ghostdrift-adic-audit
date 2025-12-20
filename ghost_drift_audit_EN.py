

# =========================================================
# Ghost Drift Audit v8.0 (Scientific Integrity / The Masterpiece)
#
# 【Update: v8.0 "Scientific Integrity"】
# Following Manny's final order, this is the final delivery version that completes the code-level implementation of
# 'trust assurance' and 'risk isolation' required for commercial operation.
#
# 1. Strict Default & Profile (strict entry gate):
#    - PROFILE defaults to "commercial"; options: "paper" and "demo".
#    - DEMO_MODE is enabled only when PROFILE == "demo".
#    - In commercial/paper modes, the script hard-stops on any data-contract violation to prevent issuing an untrustworthy certificate.
#
# 2. Split Integrity (reproducible split guarantee):
#    - commercial: ratio split, but stamps the boundary datetime into the certificate for reproducibility.
#    - paper: forces a fixed-date split to fully freeze experimental conditions.
#
# 3. Policy Separation (no goal-seeking backsolving):
#    - Separates threshold selection into "Scientific Fixed" vs "Ops Budget".
#    - paper: parameters are frozen over the test period to eliminate P-hacking suspicion.
#    - commercial: allows rolling calibration for real operations, and records it explicitly in the certificate.
#
# [Certificate: definition of issuance (strict)]
# - ✅ OK: within Scientific criteria and model health is sound
# - ⚠ NG: criteria breach (cap hit) / model degradation (drift or loses to naive) / distribution shift (shift)
# - 🚧 DEMO (NO VERDICT): demo mode does not issue a verdict
# =========================================================

# Safety net when LightGBM is not installed (e.g., Colab)
# !pip -q install lightgbm

import os
import json
import glob
# Provide alias for glob to avoid import shadowing in local scopes. Some functions may import
# glob as a different alias (e.g., glob_module). Defining this alias globally ensures
# consistent reference and avoids NameError if local imports are modified or removed later.
glob_module = glob
import csv
import hashlib
import sys
import platform
import re
from textwrap import dedent
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb

# Type Aliases for better readability
AuditConfig = Dict[str, Any]
Fingerprints = Dict[str, str]

# ==========================================
# 0.1 Code Identity (Deterministic Certificate ID)
CODE_VERSION = "v8.0.3"  # bump this when code changes (used in certificate_id)
def _compute_code_sha256() -> str:
    """
    Compute SHA256 of the current code file for integrity.
    Prefer hashing the source file if __file__ is available; otherwise hash key functions.
    This avoids accidental reuse of fixed CODE_VERSION hash which is not scientifically sufficient.
    """
    try:
        file_path = globals().get("__file__", None)
        if isinstance(file_path, str) and os.path.exists(file_path):
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
    except Exception:
        pass
    try:
        import inspect
        # Fall back to hashing key function sources if file is inaccessible
        parts = {}
        for fn_name in ("load_and_preprocess_data", "train_lgb_model", "train_model", "run_audit", "main"):
            fn = globals().get(fn_name)
            if callable(fn):
                parts[fn_name] = inspect.getsource(fn)
        if parts:
            j = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(j.encode("utf-8")).hexdigest()
    except Exception:
        pass
    # Final fallback
    return hashlib.sha256(CODE_VERSION.encode("utf-8")).hexdigest()
CODE_SHA256  = _compute_code_sha256()

# ==========================================
# 0. Operational configuration (config)
# ==========================================
AUDIT_CONFIG: AuditConfig = {
    # ---------------------------------------------------------
    # [System Profile]
    # ---------------------------------------------------------
# [Operational default] Default is commercial (external CSV required; strict gate). demo is for synthetic/training only (NO VERDICT).
    # Strict Default: PROFILE='commercial' (requires external CSV). PROFILE='paper' freezes split/tau. PROFILE='demo' allows synthetic (NO VERDICT).
    'PROFILE':           "commercial",  # "commercial" | "paper" | "demo"
    'DEMO_MODE':         None,          # DEPRECATED (legacy mirror only). Derived from PROFILE; do not set.

    # Scientific-grade enforcement flag. When True (default), strict reproducibility rules apply
    # and synthetic/demo-only features (such as net fetch or column imputation) are disabled.
    'SCIENTIFIC_GRADE':   True,

    # ---------------------------------------------------------
    # [Data Source]
    # ---------------------------------------------------------
    'USE_SYNTHETIC_DATA':       False,   # True: synthetic (fixed seed) / False: real data (prefer local CSV)
    
# Candidate input filenames to load (priority order)
    'EXTERNAL_CSV_PATHS':       [
        # Weather CSV candidates. The first match in this list takes precedence.  
        # For scientific or commercial use, prefer explicit weather-only files.  
        "electric_load_weather.csv",
        # Additional encodings/variants of the same file (utf8/shiftjis/cp932).  
        "electric_load_weather_utf8.csv",
        "electric_load_weather_shiftjis.csv",
        "electric_load_weather_cp932.csv",
        # JMA weather CSV (raw format).  
        "jma_weather.csv",
        # Legacy/demo-specific combined datasets. These remain for backward compatibility  
        # and will be considered after the above candidates.  
        "【Weather】202401-202404.csv", # [Demo] JMA weather data
        "demand_weather_merged.csv",    # [Alt] alternative name pattern
        "target_data.csv",              # [Generic]
    ],

# Two-file operation (demand CSV). Network download is OFF by default to close the reproducibility surface for science/commercial.
    'POWER_USAGE_CSV_PATHS': [
        "power_usage.csv",
        "*power_usage*.csv",
    ],
    'ALLOW_NET_DEMAND_FETCH': False,
    'DEMAND_MERGE_HOW': "inner",
    'MIN_MERGE_OVERLAP_RATIO': 0.98,
    
    'SYNTHETIC_SEED':           42,      # RNG seed for synthetic data generation
    
# [Paper Mode Settings] Paper mode forces a fixed-date split for reproducibility.
    'PAPER_SPLIT_DATES': {
    'TEST_START_DATE':  '2024-01-01',   # Test starts here (future window)
    'CALIB_START_DATE': '2023-10-01',   # Calibration starts here (recent past)
# Earlier than that is Fit.
    },

# [Budget & Sensitivity] Operational budget and sensitivity settings
    'TARGET_EVENTS_PER_WEEK':   1,       # Allowed event budget per week (ops workload upper bound)
    'ROLLING_CALIB_DAYS':       7,       # Rolling calibration window (use last N days as baseline)
    'TAU_CAP_RATIO':            1.5,     # Cap multiplier over Scientific tau (prevents self-justifying relaxation)
    
# [Algorithm Parameters] Core detection parameters
    'Q0':                        0.999,  # Scientific fixed quantile (pre-registered)
    'THR_MIN_N':              6,       # Minimum sample size (Hour×Weekend) to ensure statistical reliability
    'W':                         5,      # Persistence window: number of points to look back
    'K':                         3,      # Persistence count: number of points within W that must exceed tau
    'LOW_RATIO':                 0.98,   # Hysteresis release ratio (prevents stuck-on)
    'COOLDOWN_HOURS':           24,      # Cooldown time to merge events (avoid splitting one incident)

# [Fortification] Auto-tuning and model health checks
    "AUTO_TUNE_WK":              True,   # Auto-tune W and K to the data
    "W_CANDIDATES":              [3,5,7,9], # Candidate W values for auto-tuning
    "K_CANDIDATES":              [2,3,4],   # Candidate K values for auto-tuning
    
# [Health Gate] Thresholds for model degradation checks
    "BEACON_VS_NAIVE_RMSE_RATIO_ALERT": 1.05,    # Beacon(AI) RMSE / Naive RMSE > 1.05 → NG
    "MODEL_DRIFT_WINDOW_DAYS":   14,     # Recent window length (days) used for drift check
    "MODEL_DRIFT_RMSE_RATIO_ALERT": 1.25,        # Recent RMSE / Calib RMSE > 1.25 → NG

# [Strict Scientific] Detect distribution shift (variance change)
    "SCORE_SHIFT_WINDOW_DAYS": 14,
    "SCORE_SHIFT_Q":             0.95,   # Upper quantile to measure score "roughness" (95th percentile)
    "SCORE_SHIFT_RATIO_ALERT": 1.25,              # Recent Q95 / Calib Q95 > 1.25 → NG
}

# ==========================================
# 1. UI/UX Helper Functions (Enhanced)
# ==========================================
def print_banner() -> None:
    """Top-of-screen: display hero message and basic info"""
    profile = AUDIT_CONFIG["PROFILE"]
    ext_used = AUDIT_CONFIG.get("_EXTERNAL_CSV_USED")
    
    if AUDIT_CONFIG['USE_SYNTHETIC_DATA']:
        data_mode = "Synthetic (Fixed Seed)"
        repro_note = "→ With a fixed seed, every run produces the same certificate."
    else:
        filename = os.path.basename(ext_used) if ext_used else "External CSV"
        data_mode = f"External CSV ({filename})"
        repro_note = "→ We stamp the input data SHA256 into the certificate, guaranteeing reproducibility for that data."

    # Switch message per profile
    if profile == "demo":
        profile_msg = "🚧 DEMO MODE (Training/Simulation Only - No Official Certificate)"
    elif profile == "paper":
        profile_msg = "🎓 PAPER MODE (Frozen Parameters / Fixed Split / Strict Reproducibility)"
    else:
        profile_msg = "🏢 COMMERCIAL MODE (Rolling Calibration / Split Spec Recorded)"

    msg = dedent(f"""
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Ghost Drift Audit v8.0 (Scientific Integrity)                          │
    │ ────────────────────────────────────────────────────────────────────── │
    │ [Hero] Record ops gaps that AI cannot explain as a reasoned "certificate" in the ledger.         │
    │                                                                        │
    │ [Sub]  Not ordinary outlier detection (3σ is "value outliers"; ADIC is "broken assumptions").    │
    │        Output does not end on screen (issues certificate JSON + ledger CSV).                     │
    │ ────────────────────────────────────────────────────────────────────── │
    │ [Profile] {profile_msg:<53} │
    │ [Data]    {data_mode:<53} │
    │    {repro_note:<68}│
    │ [Gate] Health Gate Active / Budget Filter Active / Full Integrity      │
    └────────────────────────────────────────────────────────────────────────┘
    """).strip("\n")
    print(msg)

def print_section(title: str) -> None:
    """Print a section divider"""
    print("\n" + "─" * 72)
    print(title)
    print("─" * 72)

def print_business_view(cert: Dict[str, Any], paths: Dict[str, str]) -> None:
    """Business View: show only the conclusion and next action (default view for business users)"""
    hs = cert["human_summary"]
    cert_id = cert["certificate_id"]
    issued = cert["issued_at_utc"]
    
    print("\n" + "━" * 72)
    print(f"【Business View】 {cert['badge']}")
    print("━" * 72)
    print(f"Verdict       : {hs['verdict']} ({hs['verdict_reason']})")
    print(f"One Liner     : {hs['one_liner']}")
    print(f"Next Action : {hs['next_action']}")
    print("-" * 72)
    print(f"Cert ID       : {cert_id}")
    print(f"Issued At     : {issued}")
    print(f"Profile       : {cert['config']['PROFILE']}")
    print(f"Download      : {paths['json']} / {paths['csv']}")
    print("━" * 72)

def print_scientific_view(cert: Dict[str, Any]) -> None:
    """Scientific View: show detailed metrics and evidence data (detailed view for scientists/auditors)"""
    m = cert["metrics"]
    fp = cert["fingerprints"]
    scope = cert["human_summary"]["scope_of_certificate"]
    split_spec = cert.get("split_spec", {})
    
    print("\n" + "─" * 72)
    print("【Scientific View】 (Evidence & Integrity Check)")
    print("─" * 72)
    
    print("[1] Scope & Split Specification")
    print(f"  - Profile      : {cert['config']['PROFILE']}")
    print(f"  - Fit Start    : {split_spec.get('fit_start')}")
    print(f"  - Calib Start  : {split_spec.get('calib_start')}")
    print(f"  - Test Start   : {split_spec.get('test_start')} (Boundaries Frozen)")
    for line in scope:
        print(f"  - {line}")
    
    print("\n[2] Thresholds & Policy")
    print(f"  - Tau Policy   : {cert.get('tau_policy', 'N/A')}")
    print(f"  - Baseline Tau : {m.get('baseline_tau', 0):.4f} (Scientific Fixed Base)")
    print(f"  - W (Window)   : {m.get('W')}")
    print(f"  - K (Hits)     : {m.get('K')}")
    
    print("\n[3] Budget Suppression Report")
    print(f"  - Cap Hit Days : {m.get('cap_hit_days')} (Critical if > 0)")
    print(f"  - Ghost Events : {m.get('ghost_events')} (Scientific Total)")
    if "ghost_events_budget" in m:
        print(f"  - Suppressed    : {m.get('suppressed_events_by_budget')} events / {m.get('suppressed_hours_by_budget')} hours (Hidden by Budget)")

    print("\n[4] Health Gate Metrics")
    if "beacon_rmse" in m:
        ratio = m.get('beacon_vs_naive_rmse_ratio', 0)
        thresh = AUDIT_CONFIG['BEACON_VS_NAIVE_RMSE_RATIO_ALERT']
        print(f"  - Beacon RMSE  : {m.get('beacon_rmse'):.2f}")
        print(f"  - Naive RMSE    : {m.get('naive_rmse'):.2f} (Seasonality-only Reference)")
        print(f"  - Ratio (B/N)  : {ratio:.3f} (Threshold: >{thresh})")
    
    if "drift_rmse_ratio" in m:
        ratio = m.get('drift_rmse_ratio', 0)
        thresh = AUDIT_CONFIG['MODEL_DRIFT_RMSE_RATIO_ALERT']
        print(f"  - Drift Ratio  : {ratio:.3f} (Recent/Calib RMSE, Threshold: >{thresh})")
    
    if "score_shift_ratio" in m:
        ratio = m.get('score_shift_ratio', 0)
        q_val = int(m.get('score_shift_q', 0.95) * 100)
        thresh = AUDIT_CONFIG['SCORE_SHIFT_RATIO_ALERT']
        print(f"  - Score Shift  : {ratio:.3f} (Q{q_val} Ratio, Threshold: >{thresh})")

    print("\n[5] Fingerprints (Integrity)")
    print(f"  - Data SHA256   : {fp.get('data_sha256')}")
    print(f"  - Config SHA256 : {fp.get('config_sha256')}")
    print(f"  - Split SHA256  : {fp.get('split_sha256')}")
    print(f"  - Code SHA256   : {fp.get('code_sha256')}")
    print(f"  - Env SHA256    : {fp.get('env_sha256')}")
    if fp.get("source_csv_sha256"):
        print(f"  - Source SHA256: {fp.get('source_csv_sha256')} (Raw CSV Check)")
    print(f"  - Cert SHA256   : {cert.get('certificate_sha256')}")
    print("─" * 72 + "\n")

# ==========================================
# 2. IO / Ledger (Hash & Save)
# ==========================================
def _sha256_bytes(b: bytes) -> str:
    """Generate a SHA256 hash from bytes"""
    return hashlib.sha256(b).hexdigest()

def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 from file bytes (memory-efficient)"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _sha256_json(obj: Any) -> str:
    """Generate a stable hash from a JSON object (keys sorted)"""
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(s.encode("utf-8"))

def _get_environment_info() -> Dict[str, str]:
    """Execution environment fingerprint (minimal set; pip-freeze equivalent)"""
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
        "lightgbm": getattr(lgb, "__version__", "unknown"),
        "matplotlib": getattr(plt.matplotlib, "__version__", "unknown"),
    }

def _utc_now_iso() -> str:
    """Return the current UTC time in ISO format"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _fingerprint_dataframe(df: pd.DataFrame, cols: Optional[List[str]] = None) -> str:
    """
    Data 'fingerprint' generation (full-dataset hash - Strict Integrity).
    Compute a hash over all rows/cols (or selected cols) of the input DataFrame,
    to fully guarantee data identity.
    """
    if cols is None:
        cols = list(df.columns)
# Safely ignore columns that do not exist
    valid_cols = [c for c in cols if c in df.columns]
    
# Hash the entire dataset (pd.util.hash_pandas_object)
# To account for row-order consistency, sort by DATETIME before hashing (if available)
    sub = df[valid_cols].copy()
    if "DATETIME" in sub.columns:
        sub = sub.sort_values("DATETIME", kind="mergesort")
        
    row_hash = pd.util.hash_pandas_object(sub, index=False).values
    
    h = hashlib.sha256()
    h.update(row_hash.tobytes())
    
# Include metadata (row count, column schema, missing-value profile) in the hash
    meta = {
        "rows": int(len(df)),
        "cols": valid_cols,
        "dtypes": {c: str(df[c].dtype) for c in valid_cols},
        "nulls": {c: int(df[c].isna().sum()) for c in valid_cols},
    }
    h.update(json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    
    return h.hexdigest()

def certificate_to_ledger_row(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Create one ledger row (CSV): fix column order for human readability"""
    hs = cert["human_summary"]
    m = cert["metrics"]
    tr = cert.get("top_reason", {})
    
    row = {
        "certificate_id": cert["certificate_id"],
        "issued_at_utc": cert["issued_at_utc"],
        "verdict": hs["verdict"],
        "verdict_reason": hs["verdict_reason"],
        "profile": cert["config"]["PROFILE"],
        "dataset_label": cert["scope"]["dataset_label"],
        "test_range": cert["scope"]["time_range"]["test"],
        "next_action_short": hs["next_action"][:50] + "..." if len(hs["next_action"]) > 50 else hs["next_action"],
        
        # Metrics
        "cap_hit_days": m.get("cap_hit_days", 0),
        "suppressed_hours_by_budget": m.get("suppressed_hours_by_budget", 0),
        "ghost_rate": m.get("ghost_rate", 0),
        
        # Reason
        "reason_major": tr.get("reason_major", ""),
        
        # Hashes (placed later)
        "data_sha256": cert["fingerprints"].get("data_sha256", ""),
        "config_sha256": cert["fingerprints"].get("config_sha256", ""),
        "split_sha256": cert["fingerprints"].get("split_sha256", ""),
        "code_sha256": cert["fingerprints"].get("code_sha256", ""),
        "source_csv_sha256": cert["fingerprints"].get("source_csv_sha256", ""),
        "env_sha256": cert["fingerprints"].get("env_sha256", ""),
        "certificate_sha256": cert.get("certificate_sha256", ""),
    }
# Also include a hash of the ledger row itself
    row["ledger_row_sha256"] = _sha256_json(row)
    return row

# ==========================================
# 0.3 Scientific-Grade Provenance Hardening
# ==========================================
class DataContractError(RuntimeError):
    """Raised when data contracts or integrity conditions are violated in scientific or commercial mode."""
    pass

def _assert_datetime_hourly(series: pd.Series, name: str) -> None:
    """
    Ensure that a datetime-like series is on an exact hourly grid with no minutes or seconds.
    Raises DataContractError if violations are found.
    """
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().any():
        raise DataContractError(f"{name}: DATETIME parse failed (NaT exists).")
    bad = (dt.dt.minute != 0) | (dt.dt.second != 0)
    if bad.any():
        raise DataContractError(f"{name}: DATETIME must be on exact hourly grid (minute/second must be 0).")

def _assert_required_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    """
    Validate that the given DataFrame contains all required columns.
    Raises DataContractError if any are missing.
    """
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise DataContractError(f"{name}: missing required columns: {miss}")

def _assert_unique_sorted_datetime(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure that the DataFrame's DATETIME column is unique and sorted.
    Returns a copy of the DataFrame with DATETIME normalized.
    Raises DataContractError on violation.
    """
    df2 = df.copy()
    df2["DATETIME"] = pd.to_datetime(df2["DATETIME"], errors="coerce")
    if df2["DATETIME"].isna().any():
        raise DataContractError(f"{name}: DATETIME contains NaT after normalization.")
    df2 = df2.drop_duplicates("DATETIME", keep="first").sort_values("DATETIME")
    if df2["DATETIME"].duplicated().any():
        raise DataContractError(f"{name}: DATETIME not unique after de-duplication.")
    return df2

def write_certificate_and_ledger(cert: Dict[str, Any], out_dir: str = "adic_out") -> Dict[str, str]:
    """Write files (JSON certificate & CSV ledger)"""
    os.makedirs(out_dir, exist_ok=True)

# certificate.json (overwrite: latest certificate)
    cert_path = os.path.join(out_dir, "certificate.json")
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, ensure_ascii=False, indent=2)

# ledger.csv (append: history ledger - Standard CSV Module)
    row = certificate_to_ledger_row(cert)
    csv_path = os.path.join(out_dir, "ledger.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return {"json": cert_path, "csv": csv_path}

# ===============================================================
# Business Artifacts Writer
# ===============================================================
def write_business_artifacts(
    test_eval: pd.DataFrame,
    events_df: pd.DataFrame,
    cert: Dict[str, Any],
    target_col: str,
    out_dir: str = "adic_out",
) -> Dict[str, str]:
    """
    Write additional files for business stakeholders: events list, evidence timeseries, and business summary.
    This supplements the core certificate and ledger with artefacts that can be directly used in decision-making.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Events CSV
    events_path = os.path.join(out_dir, "events.csv")
    if events_df is None or events_df.empty:
        pd.DataFrame([], columns=["start", "end", "reason_major", "action", "peak_score"]).to_csv(
            events_path, index=False, encoding="utf-8"
        )
    else:
        events_df.to_csv(events_path, index=False, encoding="utf-8")

    # Evidence timeseries CSV: minimal set of columns for reproducibility
    evidence_path = os.path.join(out_dir, "evidence_timeseries.csv")
    cols = [
        "DATETIME",
        target_col,
        "PRED",
        "PRED_NAIVE",
        "RES",
        "SCORE",
        "REASON",
        "TAU_SCI",
        "TAU_BUDGET",
        "TAU_CAP_HIT",
        "ADIC_GHOST_SCI",
        "ADIC_GHOST_BUDGET",
        "OUTLIER_3SIGMA",
    ]
    cols = [c for c in cols if c in test_eval.columns]
    test_eval.loc[:, cols].to_csv(evidence_path, index=False, encoding="utf-8")

    # Business summary JSON: one-page summary of key outcomes
    bs_path = os.path.join(out_dir, "business_summary.json")
    hs = cert.get("human_summary", {})
    m = cert.get("metrics", {})
    tr = cert.get("top_reason", {})
    summary = {
        "badge": cert.get("badge", ""),
        "verdict": hs.get("verdict", ""),
        "verdict_reason": hs.get("verdict_reason", ""),
        "one_liner": hs.get("one_liner", ""),
        "next_action": hs.get("next_action", ""),
        "cap_hit_days": m.get("cap_hit_days", 0),
        "ghost_events": m.get("ghost_events", 0),
        "suppressed_hours_by_budget": m.get("suppressed_hours_by_budget", 0),
        "ghost_rate": m.get("ghost_rate", 0),
        "reason_major": tr.get("reason_major", ""),
        "certificate_id": cert.get("certificate_id", ""),
        "certificate_sha256": cert.get("certificate_sha256", ""),
    }
    with open(bs_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {"events_csv": events_path, "evidence_csv": evidence_path, "business_json": bs_path}

# ==========================================
# 3. Core Audit Logic & Helper Calculations
# ==========================================

def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root Mean Squared Error calculation helper"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def fit_seasonal_naive(calib_df: pd.DataFrame, target: str, group_cols: Tuple[str, str] = ("HOUR", "IS_WEEKEND")) -> Tuple[Dict[Any, float], float]:
    """
    Train a seasonal naive model (for control experiment):
    To prove that Beacon (LightGBM) is not "too smart and hallucinating",
    build a simple mean-table baseline.
    """
    g = calib_df.groupby(list(group_cols))[target].mean()
    global_mean = float(calib_df[target].mean())
    return g.to_dict(), global_mean

def predict_seasonal_naive(df: pd.DataFrame, mapping: Dict[Any, float], global_mean: float, group_cols: Tuple[str, str] = ("HOUR", "IS_WEEKEND")) -> np.ndarray:
    """Predict with the seasonal naive model"""
    keys = list(zip(df[group_cols[0]].values, df[group_cols[1]].values))
    return np.array([mapping.get(k, global_mean) for k in keys], dtype=float)

def beacon_vs_naive_health(test_df: pd.DataFrame, target: str, ratio_alert: float = 1.05) -> Dict[str, Any]:
    """
    Health Check 1: Beacon(LightGBM) vs Naive
    Check whether the advanced model loses to the naive baseline.
    If it loses, we judge the model as "overfitting" or "mis-specified".
    """
    beacon_rmse = _rmse(test_df[target].values, test_df["PRED"].values)
    naive_rmse  = _rmse(test_df[target].values, test_df["PRED_NAIVE"].values)
    ratio = float(beacon_rmse / max(naive_rmse, 1e-9))
    underperform = int(ratio > ratio_alert)
    return {
        "beacon_rmse": beacon_rmse,
        "naive_rmse": naive_rmse,
        "beacon_vs_naive_rmse_ratio": ratio,
        "beacon_underperform": underperform,
    }

def model_drift_health(calib_subset: pd.DataFrame, test_df: pd.DataFrame, target: str, window_days: int = 14, ratio_alert: float = 1.25) -> Dict[str, Any]:
    """
    Health Check 2: Model Drift (RMSE Degradation)
    Drift check: compare performance between recent window vs calibration window.
    Significant degradation suggests the underlying data distribution has changed.
    """
    rmse_calib = _rmse(calib_subset[target].values, calib_subset["PRED"].values)

    end_t = test_df["DATETIME"].max()
    start_recent = end_t - pd.Timedelta(days=window_days)
    recent = test_df[test_df["DATETIME"] >= start_recent]
# If no recent data exists, treat it as equivalent to Calib (avoid errors)
    rmse_recent = _rmse(recent[target].values, recent["PRED"].values) if len(recent) > 0 else rmse_calib

    ratio = float(rmse_recent / max(rmse_calib, 1e-9))
    drift_alert = int(ratio > ratio_alert)
    return {
        "rmse_calib": rmse_calib,
        "rmse_recent": rmse_recent,
        "drift_rmse_ratio": ratio,
        "model_drift_alert": drift_alert,
        "drift_window_days": int(window_days),
    }

def score_shift_health(calib_scores: np.ndarray, test_df: pd.DataFrame, window_days: int = 14, q: float = 0.95, ratio_alert: float = 1.25) -> Dict[str, Any]:
    """
    Health Check 3: Score Distribution Shift (Heteroscedasticity)
    Distribution-shift check: compare score variance between recent window and calibration window.
    A significant change suggests the model is operating under a different regime.
    """
    calib_scores = np.asarray(calib_scores, dtype=float)
    q_calib = float(np.quantile(calib_scores, q)) if len(calib_scores) > 0 else 1.0

    end_t = test_df["DATETIME"].max()
    start_recent = end_t - pd.Timedelta(days=window_days)
    recent = test_df[test_df["DATETIME"] >= start_recent]
    recent_scores = recent["SCORE"].values.astype(float) if len(recent) > 0 else calib_scores
    
    q_recent = float(np.quantile(recent_scores, q)) if len(recent_scores) > 0 else q_calib

    ratio = float(q_recent / max(q_calib, 1e-9))
    shift_alert = int(ratio > ratio_alert)
    return {
        "score_shift_q": float(q),
        "score_shift_ratio": ratio,
        "score_shift_alert": shift_alert,
        "score_shift_window_days": int(window_days)
    }

def auto_tune_WK(scores_calib: np.ndarray, target_events: int, W_candidates: List[int], K_candidates: List[int], low_ratio: float = 0.98) -> Tuple[int, int, float, int]:
    """
    Parameter Auto-Tuning
    Automatically select the best W (persistence window) and K (hit count) for the data.
    """
    best = None
    for W in W_candidates:
        for K in K_candidates:
# Search the best threshold (tau) for each (W, K) combination
            tau = calibrate_tau_per_week(scores_calib, target_events_per_week=target_events, W=W, K=K, low_ratio=low_ratio)
            
            g, _, _ = pipeline_flags(scores_calib, tau, W=W, K=K, low_ratio=low_ratio)
            e = count_events(g)
            ok = (e <= (target_events * (len(scores_calib)/(24*7) + 1e-9))) # Approximate validation

            key = (0 if ok else 1, float(tau) if ok else float(e), int(W), int(K))
            if (best is None) or (key < best["key"]):
                best = {"W": int(W), "K": int(K), "tau": float(tau), "events": int(e), "key": key}

    if best is None:
        return W_candidates[0], K_candidates[0], 1.0, 0
        
    return best["W"], best["K"], best["tau"], best["events"]

# Action playbook: define the 'next step' per major reason
_ACTION_PLAYBOOK: Dict[str, List[str]] = {
    "SHAPE_GRAD": [
        "Ops log: confirm switching/control/DR interventions",
        "External conditions: check weather (TEMP/SUN) and holidays/events",
        "Data quality: check time misalignment, missing values, duplicates, and units",
    ],
    "LEVEL_RESIDUAL": [
        "Demand level: check baseline step changes (operations / conservation / contract)",
        "Facility / grid: check load transfer, large-customer stop/restart, and maintenance",
        "Data quality: check sensor coefficients and join keys",
    ],
}

def format_action(reason_major: str, cap_hit: bool = False) -> str:
    """Render the action list in a readable format"""
    default = _ACTION_PLAYBOOK.get(reason, ["Check ops log / external conditions / data quality"])
    if cap_hit:
        steps.append("[IMPORTANT] Budget relaxation hit the cap (treated as audit NG)")
    return " / ".join(steps[:3])

def build_threshold_tables(calib_df: pd.DataFrame, group_cols: List[str], value_col: str, q: float = 0.999, min_n: int = 6) -> Tuple[pd.Series, pd.Series, float]:
    """Build a threshold map: compute quantiles per Hour × Weekend type"""
    global_thr = float(calib_df[value_col].quantile(q))
    
    # Group by Hour + Weekend
    tbl_hw = calib_df.groupby(group_cols)[value_col].agg(n='size', thr=lambda s: float(s.quantile(q))).reset_index()
    tbl_hw['thr'] = tbl_hw.apply(lambda r: r['thr'] if r['n'] >= min_n else np.nan, axis=1)
    map_hw = tbl_hw.set_index(group_cols)['thr']
    
    # Group by Hour only (Fallback)
    tbl_h = calib_df.groupby(['HOUR'])[value_col].agg(n='size', thr=lambda s: float(s.quantile(q))).reset_index()
    map_h = tbl_h.set_index('HOUR')['thr']
    
    return map_hw, map_h, global_thr

def map_threshold(df: pd.DataFrame, group_cols: List[str], map_hw: pd.Series, map_h: pd.Series, global_thr: float) -> np.ndarray:
    """Map the appropriate threshold to each row in the dataframe"""
    out = np.empty(len(df), dtype=float)
    for i, row in enumerate(df.itertuples(index=False)):
        hour = getattr(row, 'HOUR')
        is_we = getattr(row, 'IS_WEEKEND')
        thr = map_hw.get((hour, is_we), np.nan)
        if np.isnan(thr):
            thr = map_h.get(hour, np.nan)
            if np.isnan(thr): thr = global_thr
        out[i] = thr
    return out

def pipeline_flags(score: np.ndarray, tau: float, W: int = 5, K: int = 3, low_ratio: float = 0.98, init_state: int = 0, prev_raw_states: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, np.ndarray]:
    """Stream-compatible pipeline"""
    state = init_state
    raw_states = np.zeros(len(score), dtype=int)
    low = tau * low_ratio
    
    # 1. Hysteresis Check (Stream)
    for i, s in enumerate(score):
        if state == 0:
            if s > tau: state = 1
        else:
            if s < low: state = 0
        raw_states[i] = state
    
    # 2. Persistence Check (Combined with previous days)
    if prev_raw_states is not None and len(prev_raw_states) > 0:
        combined = np.concatenate([prev_raw_states, raw_states])
    else:
        combined = raw_states
    
    # Rolling Sum over window W
    rolled = pd.Series(combined).rolling(W, min_periods=1).sum().values
    
    # Filter back to current day's length
    if prev_raw_states is not None:
        # slice off the prepended history
        current_rolled = rolled[len(prev_raw_states):]
    else:
        current_rolled = rolled
        
    final_flags = (current_rolled >= K).astype(int)
    
    # Return flags, final state, and tail history for next day
    tail_len = min(len(combined), W - 1) if W > 1 else 0
    next_history = combined[-tail_len:] if tail_len > 0 else np.array([], dtype=int)
    
    return final_flags, state, next_history

def count_events(flag01: np.ndarray) -> int:
    """Count the number of rising edges (events) in a 0/1 flag series"""
    flag01 = flag01.astype(int)
    if len(flag01) == 0: return 0
    return int(((flag01[1:] - flag01[:-1]) == 1).sum() + (flag01[0] == 1))

def calibrate_tau_per_week(scores_calib: np.ndarray, target_events_per_week: float = 1.0, W: int = 5, K: int = 3, low_ratio: float = 0.98) -> float:
    """
    Calibrate tau so that events_per_week stays within the allowed budget.
    Back-calculate from allowed events/week to total allowed events over the calibration period,
    then search for the quantile that satisfies it.
    """
    hours = max(len(scores_calib), 1)
    weeks = max(hours / (24.0 * 7.0), 1e-9)
    target_events_total = target_events_per_week * weeks

    cand_q = np.linspace(0.90, 0.999, 200)
    for q in cand_q[::-1]:
        tau = float(np.quantile(scores_calib, q))
        g, _, _ = pipeline_flags(scores_calib, tau, W=W, K=K, low_ratio=low_ratio)
        if count_events(g) <= target_events_total:
            return tau
    return float(np.max(scores_calib))

def compute_tau_policies(calib_scores: np.ndarray, config: AuditConfig) -> Tuple[float, float]:
    """
    Separate threshold-selection policies:
    - Scientific tau: fixed quantile q0 (pre-registered, stable claim).
    - Ops tau: budget-satisfying quantile (but capped to avoid self-justifying relaxation).
    """
# 1) Scientific: fixed (pre-registered) — q0 is declared as the paper's experimental condition
    tau_fixed = float(np.quantile(calib_scores, config["THR_Q"]))

# 2) Ops: satisfy the budget (but decide using calibration only)
    tau_budget = calibrate_tau_per_week(
        calib_scores, 
        target_events_per_week=config["TARGET_EVENTS_PER_WEEK"],
        W=config["W"], K=config["K"], low_ratio=config["LOW_RATIO"]
    )
    return tau_fixed, tau_budget

def build_mean_std_tables(calib_df: pd.DataFrame, group_cols: List[str], value_col: str, min_n: int = 6) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, float, float]:
    """Build a mean/std table for the 3σ reference"""
    global_mu = float(calib_df[value_col].mean())
    global_sd = float(np.std(calib_df[value_col].values, ddof=0))
    if not np.isfinite(global_sd) or global_sd <= 0: global_sd = 1e-9

    def safe_std(s): return float(np.std(s.values, ddof=0))

    tbl_hw = calib_df.groupby(group_cols)[value_col].agg(n='size', mu='mean', sd=safe_std).reset_index()
    tbl_h  = calib_df.groupby(['HOUR'])[value_col].agg(n='size', mu='mean', sd=safe_std).reset_index()
    
    for tbl in [tbl_hw, tbl_h]:
        tbl['mu'] = tbl.apply(lambda r: r['mu'] if r['n'] >= min_n else np.nan, axis=1)
        tbl['sd'] = tbl.apply(lambda r: r['sd'] if (r['n'] >= min_n and r['sd'] > 0) else np.nan, axis=1)

    return (tbl_hw.set_index(group_cols)['mu'], tbl_hw.set_index(group_cols)['sd'],
            tbl_h.set_index('HOUR')['mu'],       tbl_h.set_index('HOUR')['sd'],
            global_mu, global_sd)

def map_mean_std(df: pd.DataFrame, group_cols: List[str], maps: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Map mean and standard deviation to each row in the dataframe"""
    map_hw_mu, map_hw_sd, map_h_mu, map_h_sd, global_mu, global_sd = maps
    mu_out = np.empty(len(df), dtype=float)
    sd_out = np.empty(len(df), dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        hour = getattr(row, 'HOUR')
        is_we = getattr(row, 'IS_WEEKEND')
        mu = map_hw_mu.get((hour, is_we), np.nan)
        sd = map_hw_sd.get((hour, is_we), np.nan)
        if not (np.isfinite(mu) and np.isfinite(sd)):
            mu = map_h_mu.get(hour, np.nan)
            sd = map_h_sd.get(hour, np.nan)
            if not (np.isfinite(mu) and np.isfinite(sd)):
                mu, sd = global_mu, global_sd
        mu_out[i] = float(mu)
        sd_out[i] = float(max(sd, 1e-9))
    return mu_out, sd_out

def make_split_spec(df: pd.DataFrame, config: AuditConfig) -> Dict[str, Any]:
    """
    Confirm split spec and return: (train_end, calib_end, audit_start, audit_end, split_spec).
    This is the 'reproducibility anchor' recorded in the certificate.
    In paper mode, dates are fixed; in commercial mode, we compute a boundary datetime from the ratio.
    """
    df = df.sort_values("DATETIME").reset_index(drop=True)
    n = len(df)
    
    if config["PROFILE"] == "paper":
# Fixed-date split (paper mode: uniquely fixed experimental condition)
        paper_dates = config["PAPER_SPLIT_DATES"]
        test_start_dt  = pd.Timestamp(paper_dates["TEST_START_DATE"])
        calib_start_dt = pd.Timestamp(paper_dates["CALIB_START_DATE"])
        fit_start_dt   = df["DATETIME"].min()
        
# Range check
        if not (fit_start_dt < calib_start_dt < test_start_dt):
             raise ValueError(f"Paper mode split dates are invalid or out of range for this data. Data Range: {df['DATETIME'].min()}..{df['DATETIME'].max()}")
             
        spec = {
            "type": "fixed_date",
            "fit_start": fit_start_dt,
            "calib_start": calib_start_dt,
            "test_start": test_start_dt,
        }
    else:
# Ratio split as before (but confirm & record the boundary as a datetime)
        fit_ratio = 0.60
        calib_ratio = 0.20
        
        fit_end_idx = max(1, int(n * fit_ratio))
        calib_end_idx = max(fit_end_idx + 1, int(n * (fit_ratio + calib_ratio)))
        calib_end_idx = min(calib_end_idx, n - 1)
        
        fit_start_dt = df.loc[0, "DATETIME"]
        calib_start_dt = df.loc[fit_end_idx, "DATETIME"] # Calib Start = Fit End Next
        test_start_dt  = df.loc[calib_end_idx, "DATETIME"] # Test Start = Calib End Next
        
        spec = {
            "type": "dynamic_ratio",
            "fit_start": fit_start_dt,
            "calib_start": calib_start_dt,
            "test_start": test_start_dt,
        }

    return spec

def split_time_periods(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame based on the confirmed Split Spec
    """
    print_section(f"[STEP 2] Time Split ({spec['type']})")
    
    fit_start = spec["fit_start"]
    calib_start = spec["calib_start"]
    test_start = spec["test_start"]

    fit = df[df["DATETIME"] < calib_start].copy()
    calib = df[(df["DATETIME"] >= calib_start) & (df["DATETIME"] < test_start)].copy()
    test = df[df["DATETIME"] >= test_start].copy()
    
# Guard when data is insufficient in paper mode, etc.
    if len(fit) == 0 or len(calib) == 0 or len(test) == 0:
        raise ValueError(f"Split resulted in empty dataframe. Check dates or data range. Fit:{len(fit)}, Calib:{len(calib)}, Test:{len(test)}")

    print(f"Time Split: Fit={len(fit)}, Calib={len(calib)}, Test={len(test)}")
    print(f"  - Fit     : {fit['DATETIME'].min().strftime('%Y-%m-%d')} .. {fit['DATETIME'].max().strftime('%Y-%m-%d')}")
    print(f"  - Calib : {calib['DATETIME'].min().strftime('%Y-%m-%d')} .. {calib['DATETIME'].max().strftime('%Y-%m-%d')}")
    print(f"  - Test   : {test['DATETIME'].min().strftime('%Y-%m-%d')} .. {test['DATETIME'].max().strftime('%Y-%m-%d')}")
    
# Convert to strings for saving (for JSON)
    spec["fit_start"] = str(spec["fit_start"])
    spec["calib_start"] = str(spec["calib_start"])
    spec["test_start"] = str(spec["test_start"])
    
    return fit, calib, test

def train_beacon_model(fit: pd.DataFrame, calib: pd.DataFrame, features: List[str], target: str) -> Any:
    """Train a LightGBM model"""
    print_section("[STEP 3] Train Model (LightGBM)")
    train_data = lgb.Dataset(fit[features], label=fit[target])
    valid_data = lgb.Dataset(calib[features], label=calib[target])
    
    # Deterministic Parameters for Commercial Reliability
    params = {
        'objective': 'regression', 
        'metric': 'rmse', 
        'verbose': -1, 
        # Reproducibility Hardening
        'seed': 42,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'data_random_seed': 42,
        'deterministic': True,
        'num_threads': 1,        # Ensure deterministic result
        'force_col_wise': True,  # Reduce overhead
    }
    
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[valid_data], 
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])
    return model

def calculate_ghost_scores(model: Any, calib: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str, config: AuditConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute scores and apply the dynamic threshold map"""
    print_section("[STEP 4] Scoring & Threshold Maps")
    audit_df = pd.concat([calib, test]).sort_values("DATETIME").reset_index(drop=True)
    audit_df["PRED"] = model.predict(audit_df[features])
    
    naive_map, naive_global = fit_seasonal_naive(calib, target)
    audit_df["PRED_NAIVE"] = predict_seasonal_naive(audit_df, naive_map, naive_global)

    D0 = float(calib[target].median())
    calib_dy_abs = np.abs(np.diff(calib[target].values))
    d0 = float(np.median(calib_dy_abs))
    audit_den = np.maximum(audit_df[target].values, D0)

    # Residuals
    audit_df["RES"] = np.abs(audit_df[target].values - audit_df["PRED"].values) / (audit_den + 1e-9)
    audit_df["RES_NAIVE"] = np.abs(audit_df[target].values - audit_df["PRED_NAIVE"].values) / (audit_den + 1e-9)

    # Gradient Residuals
    audit_dy = np.diff(audit_df[target].values, prepend=audit_df[target].values[0])
    audit_dy_pred = np.diff(audit_df["PRED"].values, prepend=audit_df["PRED"].values[0])
    audit_grad_den = np.maximum(np.abs(audit_dy), d0)
    audit_df["GRAD_RES"] = np.abs(audit_dy - audit_dy_pred) / (audit_grad_den + 1e-9)

    group_cols = ["HOUR", "IS_WEEKEND"]
# Build thresholds using the Calib period only (no leakage into the Audit period)
    calib_subset_for_thresh = audit_df[audit_df["DATETIME"] < test["DATETIME"].min()].copy()

    map_hw_res, map_h_res, g_res = build_threshold_tables(calib_subset_for_thresh, group_cols, "RES", q=config["THR_Q"], min_n=config["THR_MIN_N"])
    map_hw_gr, map_h_gr, g_gr = build_threshold_tables(calib_subset_for_thresh, group_cols, "GRAD_RES", q=config["THR_Q"], min_n=config["THR_MIN_N"])

    audit_df["THR_RES"]  = map_threshold(audit_df, group_cols, map_hw_res, map_h_res, g_res)
    audit_df["THR_GRAD"] = map_threshold(audit_df, group_cols, map_hw_gr,  map_h_gr,  g_gr)

    eps = 1e-9
    audit_df["S_RES"]  = audit_df["RES"]       / (audit_df["THR_RES"]  + eps)
    audit_df["S_GRAD"] = audit_df["GRAD_RES"] / (audit_df["THR_GRAD"] + eps)
    audit_df["SCORE"]  = np.maximum(audit_df["S_RES"], audit_df["S_GRAD"])
    audit_df["REASON"] = np.where(audit_df["S_RES"] >= audit_df["S_GRAD"], "LEVEL_RESIDUAL", "SHAPE_GRAD")
    
    calib_subset = audit_df[audit_df["DATETIME"] < test["DATETIME"].min()].copy()
    
    return audit_df, calib_subset

def run_audit_simulation(audit_df: pd.DataFrame, init_tau_fixed: float, init_tau_budget: float, config: AuditConfig, test_start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Audit Simulation with Stream Continuity & Policy Separation
    """
    print_section(f"[STEP 5] Simulation ({config['PROFILE'].upper()} Mode)")
    
    audit_df["ADIC_GHOST_BUDGET"] = 0
    audit_df["ADIC_GHOST_SCI"]    = 0
    audit_df["TAU_BUDGET"]        = np.nan
    audit_df["TAU_SCI"]           = np.nan
    audit_df["TAU_FIXED"]         = init_tau_fixed # Scientific Baseline (Constant)
    audit_df["TAU_CAP_HIT"]       = 0

    test_end_date = audit_df["DATETIME"].max()
    day_cursor = test_start_date
    
    # Continuity States
    state_budget = 0
    hist_budget = np.array([], dtype=int)
    state_sci = 0
    hist_sci = np.array([], dtype=int)
    
    # Frozen Parameter check
    is_frozen = (config["PROFILE"] == "paper")
    
    # Initial Taus
    current_tau_budget = init_tau_budget
    current_tau_sci_cap = init_tau_fixed * config["TAU_CAP_RATIO"]
    current_tau_sci = min(current_tau_budget, current_tau_sci_cap)
    
    if is_frozen:
        print("  -> Parameters Frozen (Paper Mode): Rolling calibration disabled.")

    while day_cursor <= test_end_date:
        calib_start = day_cursor - pd.Timedelta(days=config["ROLLING_CALIB_DAYS"])
        calib_mask = (audit_df["DATETIME"] >= calib_start) & (audit_df["DATETIME"] < day_cursor)
        day_mask = (audit_df["DATETIME"] >= day_cursor) & (audit_df["DATETIME"] < day_cursor + pd.Timedelta(days=1))
        
        # 1. Update Thresholds (Only if not frozen)
        if not is_frozen:
            calib_mask_clean = calib_mask & (audit_df["ADIC_GHOST_SCI"] == 0) & (audit_df["TAU_CAP_HIT"] == 0)
            scores_calib = audit_df.loc[calib_mask_clean, "SCORE"].values

            if len(scores_calib) >= 24 * 7 * 0.5:
                # Recalibrate Ops Budget Tau
                current_tau_budget = calibrate_tau_per_week(
                    scores_calib, 
                    target_events_per_week=config["TARGET_EVENTS_PER_WEEK"], 
                    W=config["W"], K=config["K"], low_ratio=config["LOW_RATIO"]
                )
                # Cap Logic
                current_tau_sci = float(min(current_tau_budget, current_tau_sci_cap))

        # 2. Check Cap Hit
        cap_hit = int(current_tau_budget > current_tau_sci_cap)

        if day_mask.sum() > 0:
            scores_day = audit_df.loc[day_mask, "SCORE"].values
            
            # Apply pipeline (Budget Stream)
            flags_budget, state_budget, hist_budget = pipeline_flags(
                scores_day, current_tau_budget, 
                W=config["W"], K=config["K"], low_ratio=config["LOW_RATIO"], 
                init_state=state_budget, prev_raw_states=hist_budget
            )
            
            # Apply pipeline (Scientific Stream - using Final Tau)
            # Paper mode: current_tau_sci is fixed from initial.
            # Commercial mode: current_tau_sci is rolling.
            flags_sci, state_sci, hist_sci = pipeline_flags(
                scores_day, current_tau_sci, 
                W=config["W"], K=config["K"], low_ratio=config["LOW_RATIO"], 
                init_state=state_sci, prev_raw_states=hist_sci
            )
            
            audit_df.loc[day_mask, "ADIC_GHOST_BUDGET"] = flags_budget
            audit_df.loc[day_mask, "ADIC_GHOST_SCI"]    = flags_sci
            audit_df.loc[day_mask, "TAU_BUDGET"]        = current_tau_budget
            audit_df.loc[day_mask, "TAU_SCI"]           = current_tau_sci
            audit_df.loc[day_mask, "TAU_CAP_HIT"]       = cap_hit
        
        day_cursor += pd.Timedelta(days=1)
    
    return audit_df

def apply_classic_outlier_detection(audit_df: pd.DataFrame, calib_subset: pd.DataFrame, target: str, group_cols: List[str], config: AuditConfig) -> pd.DataFrame:
    """Classic 3σ outlier detection (reference)"""
    maps = build_mean_std_tables(calib_subset, group_cols, target, min_n=config['THR_MIN_N'])
    audit_df['DEMAND_MU'], audit_df['DEMAND_SD'] = map_mean_std(audit_df, group_cols, maps)
    audit_df['Z_DEMAND'] = np.abs(audit_df[target].values - audit_df['DEMAND_MU'].values) / (audit_df['DEMAND_SD'].values + 1e-9)
    
    outlier_flags, _, _ = pipeline_flags(audit_df['Z_DEMAND'].values, 3.0, W=config['W'], K=config['K'], low_ratio=config['LOW_RATIO'])
    audit_df['OUTLIER_3SIGMA'] = outlier_flags
    return audit_df

def extract_events_list(df: pd.DataFrame, target_col: str = "ADIC_GHOST_SCI", cooldown_hours: int = 24) -> pd.DataFrame:
    """List detected events and attach peak score and reason"""
    flag = df[target_col].astype(int).values
    idx = np.where(flag == 1)[0]
    events = []
    if len(idx) == 0: return pd.DataFrame(events)

    df_reset = df.reset_index(drop=True)
    current = {"start_idx": idx[0], "end_idx": idx[0]}

    def _flush(ev):
        seg = df_reset.iloc[ev["start_idx"]:ev["end_idx"] + 1]
        reason_major = seg["REASON"].value_counts().idxmax()
        peak_i = int(seg["SCORE"].values.argmax() + ev["start_idx"])
        cap_hit = bool(df_reset["TAU_CAP_HIT"].iloc[peak_i] == 1) if "TAU_CAP_HIT" in df_reset.columns else False
        return {
            "start": seg["DATETIME"].iloc[0],
            "end": seg["DATETIME"].iloc[-1],
            "hours": int(len(seg)),
            "peak_score": float(seg["SCORE"].max()),
            "reason_major": reason_major,
            "action": format_action(reason_major, cap_hit=cap_hit),
        }

    for j in range(1, len(idx)):
        t_curr = df_reset["DATETIME"].iloc[idx[j]]
        t_prev = df_reset["DATETIME"].iloc[current["end_idx"]]
        time_diff = (t_curr - t_prev).total_seconds() / 3600

        if idx[j] == idx[j - 1] + 1 or time_diff <= cooldown_hours:
            current["end_idx"] = idx[j]
        else:
            events.append(_flush(current))
            current = {"start_idx": idx[j], "end_idx": idx[j]}

    events.append(_flush(current))
    return pd.DataFrame(events).sort_values("peak_score", ascending=False)

def visualize_results(test_eval: pd.DataFrame, target_col: str, baseline_tau: float, config: AuditConfig) -> None:
    """Visualize audit results: time series plot and threshold evolution"""
    print_section("[STEP 7] Visualization")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # 1. Demand & Events
    ax1.plot(test_eval['DATETIME'], test_eval[target_col], label='Actual', color='black', alpha=0.6, linewidth=1)
    ax1.plot(test_eval['DATETIME'], test_eval['PRED'], label='Beacon Model', color='blue', linestyle='--', alpha=0.6, linewidth=1)
    
    # 3σ Reference
    outlier = test_eval[test_eval['OUTLIER_3SIGMA'] == 1]
    if not outlier.empty:
        ax1.scatter(outlier['DATETIME'], outlier[target_col], color='orange', marker='x', label='Classic Outlier (3σ)', s=30, zorder=4)

    # Ghost Event
    ghost = test_eval[test_eval['ADIC_GHOST_SCI'] == 1]
    if not ghost.empty:
        ax1.scatter(ghost['DATETIME'], ghost[target_col], color='red', label='Ghost Drift (Structure Break)', s=40, zorder=5)
    
    ax1.set_title(f"Demand vs Pred + Ghost Events (Profile: {config['PROFILE']})", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. TAU Trend (Separated Policies)
    ax2.plot(test_eval['DATETIME'], test_eval['TAU_BUDGET'], label='Ops Budget Tau (Constraint)', color='green', marker='o', markersize=3, linestyle='-', alpha=0.7)
    ax2.plot(test_eval['DATETIME'], test_eval['TAU_SCI'],    label='Final Decision Tau (Used)', color='black', marker='.', markersize=3, linestyle='-')
    ax2.axhline(y=baseline_tau, color='blue', linestyle='--', label=f'Scientific Fixed Tau ({baseline_tau:.2f})', alpha=0.8)
    ax2.axhline(y=baseline_tau * config['TAU_CAP_RATIO'], color='orange', linestyle=':', label='Sci Cap Limit')
    
    high_tau_mask = test_eval['TAU_CAP_HIT'] == 1
    if high_tau_mask.sum() > 0:
        ax2.fill_between(test_eval['DATETIME'], 0, float(test_eval['TAU_BUDGET'].max())*1.1, where=high_tau_mask, color='orange', alpha=0.2, label='Cap Hit (Audit NG)')
    
    ax2.set_title('TAU Policy Separation (Scientific Fixed vs Ops Budget)', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Formatting
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. Certificate Generation (Fully Implemented)
# ==========================================
def make_adic_certificate(
    dataset_label: str,
    target_label: str,
    time_range: Dict[str, str],
    premise: List[str],
    verdict: str,
    verdict_reason: str,
    key_metrics: Dict[str, Any],
    top_reason: Dict[str, Any],
    environment: Dict[str, str],
    fingerprints: Fingerprints,
    config: AuditConfig,
    split_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate the certificate (JSON) body"""

    # Deterministic certificate ID uses data, config, split, and code fingerprints
    base = (
        fingerprints.get("data_sha256", "")
        + "|" + fingerprints.get("config_sha256", "")
        + "|" + fingerprints.get("split_sha256", "")
        + "|" + fingerprints.get("code_sha256", "")
    ).encode("utf-8")
    cert_id = "GDADIC-" + _sha256_bytes(base)[:16]

    # Verdict Badge Logic
    if verdict == "OK":
        badge = "✅ CERTIFICATE OK"
    elif verdict == "DEMO":
        badge = "🚧 DEMO (NO VERDICT)"
    else:
        badge = "⚠ CERTIFICATE NG"

    # Tau Policy description
    if config["PROFILE"] == "paper":
        tau_policy = "Frozen (Fixed at Initial Calibration). Separate Ops Budget calculated but constant."
    else:
        tau_policy = "Rolling (Updated weekly based on Ops Budget). Scientific Fixed baseline provided for reference."

    scope_lines = [
        f"Profile: {config['PROFILE'].upper()}",
        f"Dataset: {dataset_label}",
        f"Calib  : {time_range.get('calib', '')}",
        f"Test   : {time_range.get('test', '')}",
        f"Policy : {tau_policy}",
        f"Fingerprints: data={fingerprints.get('data_sha256')[:8]}... / config={fingerprints.get('config_sha256')[:8]}...",
    ]

    reason_map = {
        "TAU_CAP_HIT": (
            "Scientific cap hit detected (baseline not stable).",
            "Treat as hard NG. Recalibrate baseline or reduce sensitivity; never hide via budget.",
        ),
        "BEACON_UNDERPERFORM": (
            "Beacon underperforms naive baseline.",
            "Retrain or revise features; freeze decisions until Beacon beats naive on Calib.",
        ),
        "MODEL_DRIFT": (
            "Model drift detected (recent window degrades).",
            "Retrain with recent window; check covariate/data pipeline drift.",
        ),
        "SCORE_SHIFT": (
            "Score distribution shift detected.",
            "Inspect regime shift/data changes; rebuild thresholds and verify integrity.",
        ),
        "BASELINE_STABLE": (
            "Scientific gate passed (no cap hit / no drift / no shift).",
            "Deploy as monitoring; keep budget as operational load-control only.",
        ),
        "DEMO_MODE": (
            "Demo mode active. No verdict.",
            "Switch PROFILE to 'commercial' or 'paper' for valid certification.",
        )
    }

    one_liner, next_action = reason_map.get(
        verdict_reason,
        ("Verdict computed.", "Review metrics and proceed with the listed action."),
    )

    cert = {
        "certificate_id": cert_id,
        "issued_at_utc": _utc_now_iso(),
        "badge": badge,
        "human_summary": {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "one_liner": one_liner,
            "next_action": next_action,
            "scope_of_certificate": scope_lines,
        },
        "metrics": key_metrics,
        "top_reason": top_reason,
        "scope": {
            "dataset_label": dataset_label,
            "target_label": target_label,
            "time_range": time_range,
            "premise": premise,
        },
        "environment": environment,
        "fingerprints": fingerprints,
        "config": config,
        "split_spec": split_spec,
        "tau_policy": tau_policy,
    }

    cert["certificate_sha256"] = _sha256_json(cert)
    return cert

# ==========================================
# 7. Data Generation & Loading
# ==========================================
def generate_synthetic_dataset(start_date: str = '2023-01-01', days: int = 500, seed: int = 42) -> pd.DataFrame:
    """[Demo] Generate synthetic data with full reproducibility guaranteed"""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=days * 24, freq='H')
    n = len(dates)
    
    df = pd.DataFrame({'DATETIME': dates})
    df['HOUR'] = df['DATETIME'].dt.hour
    df['DAYOFYEAR'] = df['DATETIME'].dt.dayofyear
    df['WEEKDAY'] = df['DATETIME'].dt.weekday
    df['IS_WEEKEND'] = (df['WEEKDAY'] >= 5).astype(int)

    base_temp = 15 + 10 * np.sin(2 * np.pi * (df['DAYOFYEAR'] - 100) / 365)
    daily_temp = 3 * np.sin(2 * np.pi * (df['HOUR'] - 9) / 24)
    df['TEMP'] = base_temp + daily_temp + np.random.normal(0, 2.0, n)
    
    df['SUN'] = np.clip(np.sin(2 * np.pi * (df['HOUR'] - 6) / 24) * 10 + np.random.normal(0, 2, n), 0, None)
    df['HUMID'] = 60 + 10 * np.cos(2 * np.pi * df['HOUR'] / 24) + np.random.normal(0, 5, n)

    demand = 3000.0
    demand += 500 * np.cos(2 * np.pi * (df['DAYOFYEAR'] - 20) / 365) ** 2
    demand += 800 * -np.cos(2 * np.pi * (df['HOUR'] - 4) / 24) 
    demand += 10 * (df['TEMP'] - 18) ** 2
    demand -= 400 * df['IS_WEEKEND']
    demand += np.random.normal(0, 50, n)
    
    df['DEMAND'] = demand
    print(f"Generated Synthetic Data: {n} rows (Seed={seed})")
    return df

def _validate_data_contract(df: pd.DataFrame, config: AuditConfig) -> None:
    """Data Contract (minimum): required columns + time monotonic + 1H continuity."""
    # Collect violations
    errors: List[str] = []
    required_cols = ["DATETIME", "DEMAND", "TEMP"]
    for c in required_cols:
        if c not in df.columns:
            errors.append(f"Missing column: {c}")

    # Check DATETIME monotonicity and continuity
    if "DATETIME" in df.columns:
        if not df["DATETIME"].is_monotonic_increasing:
            errors.append("DATETIME not monotonic increasing")
        if df["DATETIME"].notna().sum() > 2:
            diffs = df["DATETIME"].diff().dropna()
            frac_1h = float((diffs == pd.Timedelta(hours=1)).mean())
            if frac_1h < 0.98:
                errors.append(f"1H continuity low: {frac_1h:.3f}")
            dup = int(df["DATETIME"].duplicated().sum())
            if dup > 0:
                errors.append(f"DATETIME duplicated: {dup}")

    # Compute simple metrics
    missing_rate = 0.0
    for c in ("DEMAND", "TEMP"):
        if c in df.columns:
            missing_rate = max(missing_rate, float(df[c].isna().mean()))

    one_hour_ratio: Optional[float] = None
    monotonic: Optional[bool] = None
    dup: Optional[int] = None
    if "DATETIME" in df.columns:
        dt = df["DATETIME"].dropna()
        monotonic = bool(dt.is_monotonic_increasing)
        dup = int(dt.duplicated().sum())
        if len(dt) >= 2:
            diffs = dt.sort_values().diff().dropna()
            one_hour_ratio = float((diffs == pd.Timedelta(hours=1)).mean()) if len(diffs) else 0.0

    status = "PASS" if not errors else "FAIL"
    one_hour_str = f"{one_hour_ratio:.3f}" if one_hour_ratio is not None else "NA"
    mono_str = str(monotonic) if monotonic is not None else "NA"
    dup_str = str(dup) if dup is not None else "NA"
    print(f"  Data Contract : {status} (missing_rate={missing_rate:.3f}, 1H_ratio={one_hour_str}, monotonic={mono_str}, dup={dup_str})")

    # If there are any violations, decide whether to raise based on profile
    if errors:
        msg = " / ".join(errors)
        profile = str(config.get("PROFILE", "")).lower()
        if profile == "demo":
            print(f"    -> DETAIL (demo): {msg}")
        else:
            raise RuntimeError(f"Data Contract VIOLATION ({str(config.get('PROFILE','')).upper()} MODE): {msg}")

def load_and_preprocess_data(config: AuditConfig) -> pd.DataFrame:
    """Load data and engineer features"""
    print_section("[STEP 1] Load & Merge Data")
    
    # PROFILE is the single source of truth
    profile = str(config.get("PROFILE", "commercial")).lower()
    is_demo = (profile == "demo")
    # Determine scientific-grade enforcement: default to True unless explicitly disabled
    scientific_grade = bool(config.get("SCIENTIFIC_GRADE", True)) or (not is_demo)
    # If scientific grade is active, demo mode is not allowed due to reproducibility requirements
    if scientific_grade and is_demo:
        raise DataContractError("SCIENTIFIC_GRADE requires IS_DEMO_MODE=False.")
    # Legacy mirror only (must not drive branching elsewhere)
    config["DEMO_MODE"] = is_demo

    # Strict: synthetic data is allowed only in DEMO (NO VERDICT)
    if config.get("USE_SYNTHETIC_DATA", False) and (not is_demo):
        raise RuntimeError("USE_SYNTHETIC_DATA is allowed only when PROFILE='demo' (NO VERDICT).")

    # Flags for unambiguous labeling
    config["_DEMO_FALLBACK_SYNTH"] = False
    config["_DEMO_SYNTH_DEMAND"] = False

    # Direct synthetic mode
    if config.get("USE_SYNTHETIC_DATA", False):
        print("  [Data] Mode: Synthetic Data (Reproducibility Guaranteed)")
        df = generate_synthetic_dataset(seed=config.get("SYNTHETIC_SEED", 42))
        config["_DEMO_FALLBACK_SYNTH"] = True
        config["_EXTERNAL_CSV_USED"] = None
        return df

    if config['USE_SYNTHETIC_DATA']:
        print("★ Mode: Synthetic Data (Reproducibility Guaranteed)")
        df = generate_synthetic_dataset(seed=config['SYNTHETIC_SEED'])
        config["_EXTERNAL_CSV_USED"] = None
        
    else:
        if is_demo:
            mode_msg = "★ Mode: External Data (Demo Mode: Fallback & Synthesis Active)"
        else:
            mode_msg = f"★ Mode: External Data ({config['PROFILE'].upper()}: Strict Contract Gate)"
        print(mode_msg)
        
        found_file = None
        raw_df = None
        
# 1. Search external files (including /mnt/data), and allow loading weather-only CSV
        # Build search directories for CSV discovery. Include the current working directory,
        # the directory of this script (to catch bundled CSVs), and common data mount points.
        # Filtering ensures all entries are valid directories before use.
        # __file__ may not exist when executed via certain mechanisms; fall back to cwd.
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() and __file__ else os.getcwd()
        search_dirs = [os.getcwd(), script_dir, "/mnt/data", "/content", "/kaggle/input"]
        search_dirs = [d for d in search_dirs if isinstance(d, str) and os.path.isdir(d)]

        base_names = list(config.get("EXTERNAL_CSV_PATHS", []))
        candidates = []

# Explicit paths (as-is + also check directly under the search dirs)
        for name in base_names:
            candidates.append(name)
            for d in search_dirs:
                candidates.append(os.path.join(d, name))

# Existing CSVs (cwd + direct child / one-level under search dirs)
        for d in [os.getcwd()] + search_dirs:
            candidates.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))
            candidates.extend(sorted(glob.glob(os.path.join(d, "*", "*.csv"))))

# Unique (preserve order)
        _uniq = []
        for c in candidates:
            if c and c not in _uniq:
                _uniq.append(c)
        candidates = _uniq

        def _read_jp_weather_csv(p):
            """Read a CP932/Shift_JIS weather CSV with JP-style headers and normalize to DATETIME/TEMP/HUMID/SUN."""
            encodings = ["cp932", "shift_jis", "utf-8", "utf-8-sig"]
            for enc in encodings:
                try:
                    with open(p, "r", encoding=enc, errors="ignore") as f:
                        head = [next(f) for _ in range(80)]
                    skip_rows = None
                    for i, line in enumerate(head):
                        if "\u5e74\u6708\u65e5\u6642" in line:
                            skip_rows = i
                            break
                    if skip_rows is None:
                        continue

                    t = pd.read_csv(p, encoding=enc, skiprows=skip_rows)

# Rename (pick from existing columns)
                    rename_map = {
                        "\u5e74\u6708\u65e5\u6642": "DATETIME",
                        "\u6c17\u6e29(\u2103)": "TEMP",
                        "\u6e7f\u5ea6(\uff05)": "HUMID",
# Support various humidity column names (some weather files may use 'relative humidity')
                        "\u76f8\u5bfe\u6e7f\u5ea6(\uff05)": "HUMID",
                        "\u76f8\u5bfe\u6e7f\u5ea6(%)": "HUMID",
                        "\u65e5\u7167\u6642\u9593(h)": "SUN",
                        "\u65e5\u7167\u6642\u9593(\u6642\u9593)": "SUN",
                        "\u65e5\u7167\u6642\u9593": "SUN",
                        "\u96fb\u529b\u9700\u8981": "DEMAND",
                        "\u9700\u8981": "DEMAND",
                        "\u9700\u8981(\u4e07kW)": "DEMAND",
                        "\u5f53\u65e5\u5b9f\u7e3e(\u4e07kW)": "DEMAND",
                    }
                    cols = []
                    for c in list(t.columns):
                        c2 = str(c).strip()
                        cols.append(c2)
                    t.columns = cols

# Pick by prefix match (handles CSVs with duplicate column names)
                    # new_cols maps original column names to standardized names; ensure each target name
                    # appears only once by checking against the existing mapped values rather than keys.
                    new_cols: Dict[str, str] = {}
                    for c in t.columns:
                        for k, v in rename_map.items():
                            if c.startswith(k) and v not in new_cols.values():
                                new_cols[c] = v
                                break
                    if new_cols:
                        t = t.rename(columns=new_cols)

# Required
                    if "DATETIME" not in t.columns or "TEMP" not in t.columns:
                        continue

                    out = pd.DataFrame()
                    out["DATETIME"] = t["DATETIME"]
                    out["TEMP"] = pd.to_numeric(t["TEMP"], errors="coerce")
                    if "HUMID" in t.columns:
                        out["HUMID"] = pd.to_numeric(t["HUMID"], errors="coerce")
                    if "SUN" in t.columns:
                        out["SUN"] = pd.to_numeric(t["SUN"], errors="coerce")
                    if "DEMAND" in t.columns:
                        out["DEMAND"] = pd.to_numeric(t["DEMAND"], errors="coerce")

                    return out
                except Exception:
                    continue
            return None

        def _download_tepco_demand_zip(start_dt, end_dt):
            """Download TEPCO monthly ZIPs programmatically and return DATETIME/DEMAND (10k kW)."""
            import io, zipfile, urllib.request

# Enumerate months
            start_dt = pd.to_datetime(start_dt)
            end_dt   = pd.to_datetime(end_dt)
            sm = start_dt.year * 12 + (start_dt.month - 1)
            em = end_dt.year   * 12 + (end_dt.month - 1)

            frames = []
            for m in range(sm, em + 1):
                y = m // 12
                mo = (m % 12) + 1
                yyyymm = f"{y:04d}{mo:02d}"
                url = f"https://www.tepco.co.jp/forecast/html/images/{yyyymm}_power_usage.zip"

                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()

                z = zipfile.ZipFile(io.BytesIO(data))
                csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    continue

# Read the first CSV
                with z.open(csv_names[0]) as f:
                    b = f.read()

# Try multiple encodings
                df_m = None
                for enc in ["cp932", "shift_jis", "utf-8", "utf-8-sig"]:
                    try:
                        df_m = pd.read_csv(io.BytesIO(b), encoding=enc)
                        break
                    except Exception:
                        df_m = None
                if df_m is None:
                    continue

# Normalize (DATE+TIME or JP timestamp header)
                cols = [str(c).strip() for c in df_m.columns]
                df_m.columns = cols

                if "DATETIME" not in df_m.columns:
                    if "\u5e74\u6708\u65e5\u6642" in df_m.columns:
                        df_m = df_m.rename(columns={"\u5e74\u6708\u65e5\u6642": "DATETIME"})
                    elif ("DATE" in df_m.columns) and ("TIME" in df_m.columns):
                        df_m["DATETIME"] = df_m["DATE"].astype(str).str.strip() + " " + df_m["TIME"].astype(str).str.strip()
                    elif ("\u65e5\u4ed8" in df_m.columns) and ("\u6642\u523b" in df_m.columns):
                        df_m["DATETIME"] = df_m["\u65e5\u4ed8"].astype(str).str.strip() + " " + df_m["\u6642\u523b"].astype(str).str.strip()

                if "DATETIME" not in df_m.columns:
                    continue

# Infer DEMAND column (assume 10k kW scale)
                demand_col = None
                if "DEMAND" in df_m.columns:
                    demand_col = "DEMAND"
                else:
# Prefer columns that look like actuals
                    prefer = ["\u5f53\u65e5\u5b9f\u7e3e(\u4e07kW)", "\u5f53\u65e5\u5b9f\u7e3e", "\u5b9f\u7e3e(\u4e07kW)", "\u5b9f\u7e3e", "\u9700\u8981(\u4e07kW)", "\u9700\u8981"]
                    for ptn in prefer:
                        for c in df_m.columns:
                            if ptn in c:
                                demand_col = c
                                break
                        if demand_col is not None:
                            break

                if demand_col is None:
                    continue

                tmp = pd.DataFrame()
                tmp["DATETIME"] = pd.to_datetime(df_m["DATETIME"], errors="coerce")
                tmp["DEMAND"] = pd.to_numeric(df_m[demand_col], errors="coerce")
                tmp = tmp.dropna(subset=["DATETIME", "DEMAND"])

                frames.append(tmp)

            if not frames:
                raise RuntimeError("TEPCO demand zip download returned no usable data.")

            ddf = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["DATETIME"]).sort_values("DATETIME")

# Cut by range and align to time
            ddf = ddf[(ddf["DATETIME"] >= start_dt) & (ddf["DATETIME"] <= end_dt)]
            if ddf.empty:
                raise RuntimeError("TEPCO demand is empty in the requested time range.")

            ddf = ddf.set_index("DATETIME").resample("1H").mean(numeric_only=True).reset_index()
            return ddf[["DATETIME", "DEMAND"]]

        found_file = None
        raw_df = None

        for p in candidates:
            if not p or not os.path.exists(p):
                continue

# Try the normal CSV (UTF family) first
            t = None
            try:
                t = pd.read_csv(p)
            except Exception:
                try:
                    t = pd.read_csv(p, encoding="utf-8-sig")
                except Exception:
                    t = None

# If DATETIME/TEMP not visible, try the JP-header CSV (allowed even in commercial mode)
            if t is None or ("DATETIME" not in t.columns and "\u5e74\u6708\u65e5\u6642" in t.columns):
                t2 = _read_jp_weather_csv(p)
                if t2 is not None:
                    t = t2

# If still no DATETIME/TEMP, try the JP parser once more
            if t is not None and ("DATETIME" not in t.columns or "TEMP" not in t.columns):
                t2 = _read_jp_weather_csv(p)
                if t2 is not None:
                    t = t2

            if t is None:
                continue

# Minimum acceptance: DATETIME/TEMP
            if ("DATETIME" in t.columns) and ("TEMP" in t.columns):
                found_file = p
                raw_df = t.copy()
                break

        config["_EXTERNAL_CSV_USED"] = found_file
# 2. Process loaded results
        if found_file is None or raw_df is None:
            if is_demo:
                # In demo, fallback to deterministic synthetic dataset when no external CSV is found
                print("  [Data] No external CSV found. Using synthetic dataset (DEMO).")
                df = generate_synthetic_dataset(seed=config.get('SYNTHETIC_SEED', 42))
                config["_DEMO_FALLBACK_SYNTH"] = True
                config["_DEMO_SYNTH_DEMAND"] = False
                config["_EXTERNAL_CSV_USED"] = None
                return df
            else:
                raise FileNotFoundError(f"External CSV not found or invalid format in {config['PROFILE']} mode.")

        config["_EXTERNAL_CSV_USED"] = found_file
        # When external CSV is used, ensure flags reflect no synthetic fallback or demand synthesis
        config["_DEMO_FALLBACK_SYNTH"] = False
        config["_DEMO_SYNTH_DEMAND"] = False
        
# 3. Preprocessing
        raw_df["DATETIME"] = pd.to_datetime(raw_df["DATETIME"], errors="coerce")
        raw_df = raw_df.dropna(subset=["DATETIME"]).sort_values("DATETIME").reset_index(drop=True)
        for c in ["TEMP", "HUMID", "SUN"]:
            if c in raw_df.columns:
                raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')

        # Strict schema & hourly grid validation (scientific-grade)
        _assert_required_cols(raw_df, ["DATETIME", "TEMP"], "weather_csv")
        # HUMID and SUN are required unless running a demo; raise if missing
        if scientific_grade:
            _assert_required_cols(raw_df, ["HUMID", "SUN"], "weather_csv")
        _assert_datetime_hourly(raw_df["DATETIME"], "weather_csv")
        raw_df = _assert_unique_sorted_datetime(raw_df, "weather_csv")

        # 4. Feature Engineering & Scenario Injection (Demo Only)
        if "DEMAND" not in raw_df.columns:
            if is_demo:
                # Synthesize demand deterministically (demo only)
                n = len(raw_df)
                temps = raw_df["TEMP"].fillna(18.0).values.astype(float)
                hours = raw_df["DATETIME"].dt.hour.values
                is_we = (raw_df["DATETIME"].dt.dayofweek >= 5).values.astype(float)
                np.random.seed(config.get("SYNTHETIC_SEED", 42))
                sim_demand = 3000.0 + 15 * (temps - 18)**2 + 200 * np.sin(2 * np.pi * hours / 24) - 400 * is_we + np.random.normal(0, 50, n)
                raw_df["DEMAND"] = sim_demand
                config["_DEMO_SYNTH_DEMAND"] = True
                print("  [WARN] DEMO: Missing DEMAND -> synthesized target (NO VERDICT).")
            else:
                # COMMERCIAL/PAPER: Use local power_usage.csv as the sole demand source; merge strictly on DATETIME
                import glob as glob_module

                def _read_power_usage_csv_local(p: str) -> pd.DataFrame:
                    """Read a power usage CSV and normalize to DATETIME/DEMAND."""
                    last_err = None
                    for enc in ["utf-8-sig", "cp932", "shift_jis", "utf-8"]:
                        try:
                            t = pd.read_csv(p, encoding=enc)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            t = None
                    if t is None:
                        raise DataContractError(f"power_usage.csv read failed: {last_err}")
                    # Normalize column names
                    t.columns = [str(c).replace("\ufeff", "").strip() for c in t.columns]
                    # Parse DATETIME
                    if "DATETIME" in t.columns:
                        dtv = pd.to_datetime(t["DATETIME"], errors="coerce")
                    elif ("DATE" in t.columns and "TIME" in t.columns):
                        dtv = pd.to_datetime(t["DATE"].astype(str).str.strip() + " " + t["TIME"].astype(str).str.strip(), errors="coerce")
                    elif ("\u65e5\u4ed8" in t.columns and "\u6642\u523b" in t.columns):
                        dtv = pd.to_datetime(t["\u65e5\u4ed8"].astype(str).str.strip() + " " + t["\u6642\u523b"].astype(str).str.strip(), errors="coerce")
                    else:
                        raise DataContractError("power_usage.csv must have DATETIME or (DATE,TIME) or (JP_DATE,JP_TIME)")
                    # Find demand column
                    prefer = ["\u5f53\u65e5\u5b9f\u7e3e(\u4e07kW)", "\u5b9f\u7e3e(\u4e07kW)", "\u9700\u8981(\u4e07kW)", "\u96fb\u529b\u9700\u8981", "\u9700\u8981"]
                    dcol = None
                    if "DEMAND" in t.columns:
                        dcol = "DEMAND"
                    else:
                        for key in prefer:
                            if key in t.columns:
                                dcol = key
                                break
                    if dcol is None:
                        raise DataContractError(f"power_usage.csv missing demand column. have={list(t.columns)}")
                    out = pd.DataFrame({
                        "DATETIME": dtv,
                        "DEMAND": pd.to_numeric(t[dcol], errors="coerce"),
                    }).dropna()
                    # Ensure hourly grid and uniqueness
                    _assert_datetime_hourly(out["DATETIME"], "power_usage_csv")
                    out = out.drop_duplicates("DATETIME").sort_values("DATETIME")
                    # Reject negative values
                    if (out["DEMAND"] < 0).any():
                        raise DataContractError("power_usage_csv: DEMAND contains negative values.")
                    return out

                def _find_power_usage_file() -> Optional[str]:
                    # Search for candidate power usage files in configured paths and search directories
                    patterns = list(config.get("POWER_USAGE_CSV_PATHS", ["power_usage.csv", "*power_usage*.csv"]))
                    cand = []
                    for pat in patterns:
                        cand.append(pat)
                        for d in search_dirs:
                            cand.append(os.path.join(d, pat))
                    # Expand globs
                    expanded = []
                    for ptn in cand:
                        if any(x in ptn for x in ["*", "?", "["]):
                            expanded.extend(glob_module.glob(ptn))
                        else:
                            expanded.append(ptn)
                    # Unique while preserving order
                    seen = set()
                    uniq = []
                    for ptn in expanded:
                        if ptn in seen:
                            continue
                        seen.add(ptn)
                        uniq.append(ptn)
                    for pth in uniq:
                        if os.path.exists(pth) and os.path.isfile(pth):
                            return pth
                    return None

                weather_n = len(raw_df)
                start_dt = raw_df["DATETIME"].min()
                end_dt   = raw_df["DATETIME"].max()

                power_file = _find_power_usage_file()
                demand_df = None
                if power_file is not None:
                    demand_df = _read_power_usage_csv_local(power_file)
                    # restrict to overlapping range
                    demand_df = demand_df[(demand_df["DATETIME"] >= start_dt) & (demand_df["DATETIME"] <= end_dt)]
                    config["_POWER_CSV_USED"] = power_file
                    config["_DEMAND_SOURCE"] = "local_csv"
                    config["_DEMAND_DOWNLOADED"] = False
                else:
                    if bool(config.get("ALLOW_NET_DEMAND_FETCH", False)):
                        demand_df = _download_tepco_demand_zip(start_dt, end_dt)
                        config["_DEMAND_SOURCE"] = "net_zip"
                        config["_DEMAND_DOWNLOADED"] = True
                    else:
                        raise DataContractError("Data Contract VIOLATION: Missing DEMAND. Provide local power_usage.csv.")

                # Align to hourly & strict merge (no fill)
                demand_df = demand_df.dropna(subset=["DATETIME", "DEMAND"])
                demand_df = demand_df.drop_duplicates("DATETIME").sort_values("DATETIME")
                how = str(config.get("DEMAND_MERGE_HOW", "inner")).lower().strip()
                raw_df = raw_df.drop(columns=["DEMAND"], errors="ignore")
                merged = raw_df.merge(demand_df, on="DATETIME", how=how).dropna(subset=["DEMAND"])

                overlap_ratio = len(merged) / max(1, min(weather_n, len(demand_df)))
                config["_MERGE_REPORT"] = {
                    "weather_rows": int(weather_n),
                    "demand_rows": int(len(demand_df)),
                    "merged_rows": int(len(merged)),
                    "overlap_ratio": float(overlap_ratio),
                    "merge_how": how,
                }
                if len(merged) == 0 or overlap_ratio < float(config.get("MIN_MERGE_OVERLAP_RATIO", 0.98)):
                    raise DataContractError(f"merge overlap too small: ratio={overlap_ratio:.6f}")
                raw_df = merged
                print(f"  [Data] DEMAND merged ({config.get('_DEMAND_SOURCE')}) rows={len(raw_df)} overlap_ratio={overlap_ratio:.6f}")

        # Strict schema & time-grid (HUMID/SUN missing in scientific runs should trigger error)
        missing_cols = [c for c in ["SUN", "HUMID"] if c not in raw_df.columns]
        if missing_cols:
            if is_demo:
                if "SUN" not in raw_df.columns: raw_df["SUN"] = 0.0
                if "HUMID" not in raw_df.columns: raw_df["HUMID"] = 50.0
            else:
                raise DataContractError(f"weather CSV missing columns: {missing_cols}")
        
        # 5. Validate
        _validate_data_contract(raw_df, config)
        
        # 6. Feature Engineering
        if "HOUR" not in raw_df.columns: raw_df["HOUR"] = raw_df["DATETIME"].dt.hour
        if "DAYOFYEAR" not in raw_df.columns: raw_df["DAYOFYEAR"] = raw_df["DATETIME"].dt.dayofyear
        if "WEEKDAY" not in raw_df.columns: raw_df["WEEKDAY"] = raw_df["DATETIME"].dt.weekday
        if "IS_WEEKEND" not in raw_df.columns: raw_df["IS_WEEKEND"] = (raw_df["WEEKDAY"] >= 5).astype(int)

        df = raw_df.copy()
        print(f"  Loaded & Preprocessed: {found_file} (rows={len(df)})")

    df['SIN_DOY'] = np.sin(2 * np.pi * df['DAYOFYEAR'] / 366.0)
    df['COS_DOY'] = np.cos(2 * np.pi * df['DAYOFYEAR'] / 366.0)
    df['TEMP2'] = df['TEMP'].values ** 2 
    df['LAG_1H'] = df['DEMAND'].shift(1)
    df['LAG_24H'] = df['DEMAND'].shift(24)
    df['LAG_168H'] = df['DEMAND'].shift(168)
    df = df.dropna().sort_values('DATETIME').reset_index(drop=True)
    # Record provenance for scientific-grade reproducibility
    try:
        power_used = config.get("_POWER_CSV_USED")
        ext_used   = config.get("_EXTERNAL_CSV_USED") or config.get("EXTERNAL_CSV_USED") or config.get("_EXT_CSV_USED")
        prov_files = []
        if ext_used and isinstance(ext_used, str) and os.path.exists(ext_used):
            prov_files.append({
                "role": "weather_csv",
                "path": ext_used,
                "sha256": _sha256_file(ext_used),
                "bytes": int(os.path.getsize(ext_used)),
            })
        if power_used and isinstance(power_used, str) and os.path.exists(power_used):
            prov_files.append({
                "role": "power_usage_csv",
                "path": power_used,
                "sha256": _sha256_file(power_used),
                "bytes": int(os.path.getsize(power_used)),
            })
        prov = {
            "code_sha256": CODE_SHA256,
            "code_version": CODE_VERSION,
            "files": prov_files,
            "merge_report": config.get("_MERGE_REPORT", {}),
            "rules": {
                "demand_merge_how": str(config.get("DEMAND_MERGE_HOW", "inner")),
                "min_merge_overlap_ratio": float(config.get("MIN_MERGE_OVERLAP_RATIO", 0.98)),
                "no_imputation_scientific": True,
                "no_net_fetch_scientific": True,
                "hourly_grid_required": True,
            },
        }
        source_sha = _sha256_json(prov) if prov_files else ""
        config["_PROVENANCE"] = prov
        config["_PROVENANCE_SHA256"] = source_sha
    except Exception:
        # provenance assignment is best effort; ignore errors silently
        pass
    return df

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    print_banner()

    # 1. Load Data
    features = ['TEMP', 'TEMP2', 'SUN', 'HUMID', 'WEEKDAY', 'IS_WEEKEND', 'HOUR', 'SIN_DOY', 'COS_DOY', 'LAG_1H', 'LAG_24H', 'LAG_168H']
    target = 'DEMAND'
    df = load_and_preprocess_data(AUDIT_CONFIG)
    
    # 2. Split Spec & Time Split
    split_spec = make_split_spec(df, AUDIT_CONFIG)
    fit, calib, test = split_time_periods(df, split_spec)
    test_start_date = test["DATETIME"].min()
    
    # 3. Train Model
    model = train_beacon_model(fit, calib, features, target)
    
    # 4. Score & Thresholding
    audit_df, calib_subset = calculate_ghost_scores(model, calib, test, features, target, AUDIT_CONFIG)
    
    # Auto-Tuning of W, K
    if AUDIT_CONFIG.get("AUTO_TUNE_WK", False):
        print("★ Running Auto-Tuning for W, K...")
        W_best, K_best, _, _ = auto_tune_WK(
            calib_subset["SCORE"].values,
            target_events=AUDIT_CONFIG["TARGET_EVENTS_PER_WEEK"],
            W_candidates=AUDIT_CONFIG["W_CANDIDATES"],
            K_candidates=AUDIT_CONFIG["K_CANDIDATES"],
            low_ratio=AUDIT_CONFIG["LOW_RATIO"],
        )
        AUDIT_CONFIG["W"] = int(W_best)
        AUDIT_CONFIG["K"] = int(K_best)
        print(f"  -> Selected: W={W_best}, K={K_best}")

    # Compute Initial Taus (Separation of Concerns)
    init_tau_fixed, init_tau_budget = compute_tau_policies(calib_subset["SCORE"].values, AUDIT_CONFIG)
    print(f"  -> Initial Taus: Fixed(Sci)={init_tau_fixed:.4f}, Budget(Ops)={init_tau_budget:.4f}")
    
    # 5. Baseline Comparison (3σ Reference)
    audit_df = apply_classic_outlier_detection(audit_df, calib_subset, target, ['HOUR', 'IS_WEEKEND'], AUDIT_CONFIG)

    # 6. Run Simulation (Audit)
    audit_df = run_audit_simulation(audit_df, init_tau_fixed, init_tau_budget, AUDIT_CONFIG, test_start_date)
    
    # 7. Generate Metrics & Events
    test_eval = audit_df[audit_df['DATETIME'] >= test_start_date].copy()
    events_df = extract_events_list(test_eval, 'ADIC_GHOST_SCI', AUDIT_CONFIG['COOLDOWN_HOURS'])

    # ------------------------------------------------------------------
# [Certificate Authority] Certificate issuance process
    # ------------------------------------------------------------------
    print_section("[STEP 6] Issuing Audit Certificate (Dual-View)")

    # Health, Drift & Shift Checks
    health = beacon_vs_naive_health(test_eval, target, ratio_alert=AUDIT_CONFIG["BEACON_VS_NAIVE_RMSE_RATIO_ALERT"])
    drift = model_drift_health(calib_subset, test_eval, target, window_days=AUDIT_CONFIG["MODEL_DRIFT_WINDOW_DAYS"], ratio_alert=AUDIT_CONFIG["MODEL_DRIFT_RMSE_RATIO_ALERT"])
    shift = score_shift_health(calib_subset["SCORE"].values, test_eval, window_days=AUDIT_CONFIG["SCORE_SHIFT_WINDOW_DAYS"], q=AUDIT_CONFIG["SCORE_SHIFT_Q"], ratio_alert=AUDIT_CONFIG["SCORE_SHIFT_RATIO_ALERT"])

    # Key Metrics
    cap_hit_mask = (test_eval["TAU_CAP_HIT"] == 1)
    cap_hit_hours = int(cap_hit_mask.sum())
    cap_hit_days = int(test_eval.loc[cap_hit_mask, "DATETIME"].dt.date.nunique()) if cap_hit_hours > 0 else 0

    key_metrics = {
        "cap_hit_days": cap_hit_days,
        "cap_hit_hours": cap_hit_hours,
        "ghost_rate": float(test_eval['ADIC_GHOST_SCI'].mean()),
        "ghost_events": int(count_events(test_eval['ADIC_GHOST_SCI'].astype(int).values)),
        "ghost_events_budget": int(count_events(test_eval['ADIC_GHOST_BUDGET'].astype(int).values)),
        "suppressed_hours_by_budget": int(((test_eval['ADIC_GHOST_SCI']==1) & (test_eval['ADIC_GHOST_BUDGET']==0)).sum()),
        "suppressed_events_by_budget": int(count_events(((test_eval['ADIC_GHOST_SCI']==1) & (test_eval['ADIC_GHOST_BUDGET']==0)).astype(int).values)),
        "outlier_rate": float(test_eval['OUTLIER_3SIGMA'].mean()),
        "outlier_events": int(count_events(test_eval['OUTLIER_3SIGMA'].astype(int).values)),
        "overlap_both": float(((test_eval['ADIC_GHOST_SCI']==1) & (test_eval['OUTLIER_3SIGMA']==1)).mean()),
        "adic_only": float(((test_eval['ADIC_GHOST_SCI']==1) & (test_eval['OUTLIER_3SIGMA']==0)).mean()),
        "outlier_only": float(((test_eval['ADIC_GHOST_SCI']==0) & (test_eval['OUTLIER_3SIGMA']==1)).mean()),
    }
    key_metrics.update(health)
    key_metrics.update(drift)
    key_metrics.update(shift)
    key_metrics.update({
        "baseline_tau": float(init_tau_fixed),
        "W": int(AUDIT_CONFIG["W"]),
        "K": int(AUDIT_CONFIG["K"]),
    })

    # Verdict Logic (Strict)
    reason_flags = {
        "DEMO_MODE": (AUDIT_CONFIG["PROFILE"] == "demo"),
        "TAU_CAP_HIT": (key_metrics["cap_hit_days"] > 0),
        "BEACON_UNDERPERFORM": (health.get("beacon_underperform", 0) == 1),
        "MODEL_DRIFT": (drift.get("model_drift_alert", 0) == 1),
        "SCORE_SHIFT": (shift.get("score_shift_alert", 0) == 1),
    }

    if reason_flags["DEMO_MODE"]:
        verdict = "DEMO"
        verdict_reason = "DEMO_MODE"
    else:
        hard_ng = any([v for k, v in reason_flags.items() if k != "DEMO_MODE"])
        verdict = "NG" if hard_ng else "OK"
        if reason_flags["TAU_CAP_HIT"]: verdict_reason = "TAU_CAP_HIT"
        elif reason_flags["BEACON_UNDERPERFORM"]: verdict_reason = "BEACON_UNDERPERFORM"
        elif reason_flags["MODEL_DRIFT"]: verdict_reason = "MODEL_DRIFT"
        elif reason_flags["SCORE_SHIFT"]: verdict_reason = "SCORE_SHIFT"
        else: verdict_reason = "BASELINE_STABLE"

    # Top Reason
    top_reason = {}
    if not events_df.empty:
        top_row = events_df.iloc[0]
        top_reason = {"reason_major": top_row["reason_major"], "action": top_row["action"], "peak_score": top_row["peak_score"]}
    
    # Fingerprints
    cols_inputs = ["DATETIME", target] + list(features)
    cols_evidence = cols_inputs + ["PRED", "PRED_NAIVE", "RES", "GRAD_RES", "RES_NAIVE", "SCORE", "REASON", "TAU_SCI", "TAU_BUDGET", "TAU_CAP_HIT", "ADIC_GHOST_SCI", "ADIC_GHOST_BUDGET", "OUTLIER_3SIGMA"]
    cols_evidence = [c for c in cols_evidence if c in test_eval.columns]
    data_sha = _fingerprint_dataframe(test_eval, cols=cols_evidence)

    # Compute source hash for multiple inputs (weather + demand)
    source_sha = ""
    ext_used = AUDIT_CONFIG.get("_EXTERNAL_CSV_USED")
    power_used = AUDIT_CONFIG.get("_POWER_CSV_USED")
    src = {}
    if ext_used and os.path.exists(ext_used):
        src["weather_csv_sha256"] = _sha256_file(ext_used)
    if power_used and os.path.exists(power_used):
        src["power_csv_sha256"] = _sha256_file(power_used)
    source_sha = _sha256_json(src) if src else ""

    # Build stable configuration and compute hashes
    stable_config = {k: v for k, v in AUDIT_CONFIG.items() if (not str(k).startswith("_")) and (k != "DEMO_MODE")}
    split_sha = _sha256_json(split_spec)
    config_dump = {
        "AUDIT_CONFIG": stable_config,
        "features": list(features),
        "target": target,
        "split_spec": split_spec,
        "code_version": CODE_VERSION,
    }
    config_sha = _sha256_json(config_dump)
    env_info = _get_environment_info()
    env_sha = _sha256_json(env_info)

    # Dataset label (unambiguous)
    profile = str(AUDIT_CONFIG.get("PROFILE", "commercial")).lower()
    is_demo = (profile == "demo")
    ext_base = os.path.basename(ext_used) if ext_used else "(none)"
    if is_demo:
        if AUDIT_CONFIG.get("USE_SYNTHETIC_DATA", False) or AUDIT_CONFIG.get("_DEMO_FALLBACK_SYNTH", False):
            dataset_lbl = "DEMO (Synthetic / Reproducible) [NO VERDICT]"
        else:
            if AUDIT_CONFIG.get("_DEMO_SYNTH_DEMAND", False):
                dataset_lbl = f"DEMO (External Weather CSV: {ext_base} + Synth DEMAND) [NO VERDICT]"
            else:
                dataset_lbl = f"DEMO (External CSV: {ext_base}) [NO VERDICT]"
    else:
        dataset_lbl = f"External CSV: {ext_base}"
        # If a separate demand CSV is used, append its name to the label for clarity
        power_base = os.path.basename(power_used) if power_used else None
        if power_base:
            dataset_lbl = f"External CSV: {ext_base} + {power_base}"
        if profile == "paper":
            dataset_lbl += " [PAPER MODE: FIXED SPLIT]"

    calib_range = f"{calib['DATETIME'].min().strftime('%Y-%m-%d')}..{calib['DATETIME'].max().strftime('%Y-%m-%d')}"
    test_range  = f"{test_eval['DATETIME'].min().strftime('%Y-%m-%d')}..{test_eval['DATETIME'].max().strftime('%Y-%m-%d')}"

    # Compose fingerprints, including split and code hashes
    fingerprints = {
        "data_sha256": data_sha,
        "config_sha256": config_sha,
        "split_sha256": split_sha,
        "code_sha256": CODE_SHA256,
        "env_sha256": env_sha,
        "source_csv_sha256": source_sha,
    }

    certificate = make_adic_certificate(
        dataset_label=dataset_lbl,
        target_label="DEMAND (MW)",
        time_range={"calib": calib_range, "test": test_range},
        premise=["Seasonality", "WeatherLink", "Inertia"],
        verdict=verdict,
        verdict_reason=verdict_reason,
        key_metrics=key_metrics,
        top_reason=top_reason,
        environment=env_info,
        fingerprints=fingerprints,
        config=AUDIT_CONFIG,
        split_spec=split_spec,
    )
    # Attach provenance bundle to certificate
    certificate["provenance"] = AUDIT_CONFIG.get("_PROVENANCE", {})
    certificate["provenance_sha256"] = AUDIT_CONFIG.get("_PROVENANCE_SHA256", "")

    # Save & Print
    paths = write_certificate_and_ledger(certificate, out_dir="adic_out")
    # Write business artefacts for meetings and operational use
    extra_paths = write_business_artifacts(
        test_eval=test_eval,
        events_df=events_df,
        cert=certificate,
        target_col=target,
        out_dir="adic_out",
    )
    print_business_view(certificate, paths)
    print_scientific_view(certificate)
    visualize_results(test_eval, target, init_tau_fixed, AUDIT_CONFIG)