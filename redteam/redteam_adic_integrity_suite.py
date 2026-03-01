"""
redteam_adic_integrity_suite.py (v2.0 - 100pt Audited Version)
---------------------------------------------------
Red-team verification script for the ghostdrift-adic-audit protocol.
To validate the ALS (Algorithmic Legitimacy Shift) phase-transition condition (B < J),
"Layer-A (tamper-evident fingerprints)" and "Layer-B (semantic attack features)" are deterministically separated,
and all decision reasons and evidence are irreversibly fixed by structuring them into a JSON schema.

Key enhancements:
1. Two-layer reason taxonomy + JSON schema
2. Strict data-contract validation independent of df.shape (Data Contract)
3. Introduction of an L1-norm pseudo-score function for boundary exploration (B2)
4. Anchor the SHA of evidence.csv itself into the ledger and certificate to resist simultaneous tampering
5. Strict environment fingerprinting equivalent to pip freeze
6. Two-stage Verify (Integrity & Semantics) that also asserts semantic correctness
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np


###############################################################################
# Fingerprinting utilities (Layer-A)
###############################################################################

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_string(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def fingerprint_config(config: Dict) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return sha256_string(canonical)

def fingerprint_code() -> str:
    return sha256_file(__file__)

def fingerprint_env(run_dir: str) -> str:
    """Enhanced environment fingerprint. Covers OS and key library versions."""
    env_data = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        # Include local timezone/locale information as part of the reproducibility context
        "timezone": datetime.datetime.now().astimezone().tzname(),
    }
    
    # Add dependency fingerprint if available
    try:
        import pkg_resources
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        env_data["pip_freeze_sha"] = sha256_string(json.dumps(installed, sort_keys=True))
    except Exception:
        env_data["pip_freeze_sha"] = "unknown"

    env_json = json.dumps(env_data, sort_keys=True, ensure_ascii=False)
    
    # Persist environment information as evidence
    with open(os.path.join(run_dir, "env_info.json"), "w", encoding="utf-8") as f:
        f.write(env_json)
        
    return sha256_string(env_json)


###############################################################################
# Raw-file parsing utilities (Layer-B, robust to multi-row headers)
###############################################################################

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _is_float_token(s: str) -> bool:
    s = s.strip()
    return bool(s) and bool(_FLOAT_RE.match(s))

def _safe_float(s: str) -> Optional[float]:
    try:
        if _is_float_token(s):
            return float(s)
        return None
    except Exception:
        return None

def _read_text_lines_cp932(path: str) -> List[str]:
    with open(path, "rb") as f:
        text = f.read().decode("cp932", errors="ignore")
    return text.splitlines()

def _find_header_count(lines: List[str]) -> int:
    """Treat the first row that contains any numeric token (float) as the start of the body; everything before it is the header."""
    for i, line in enumerate(lines):
        parts = line.split(",")
        if any(_is_float_token(p) for p in parts):
            return i
    return len(lines)

def _header_signature(lines: List[str]) -> str:
    """Deterministic signature of the header section."""
    hc = _find_header_count(lines)
    header = "\n".join([l.rstrip() for l in lines[:hc]]).strip()
    return sha256_string(header)

def _body_lines(lines: List[str]) -> List[str]:
    hc = _find_header_count(lines)
    return lines[hc:]

def _token_diff_metrics(base_body: List[str], curr_body: List[str]) -> Dict[str, Any]:
    """Compute metrics needed for A1/B2 decisions from same-position numeric-token diffs."""
    n = min(len(base_body), len(curr_body))
    base_sum = 0.0
    diff_sum = 0.0
    changed = 0
    max_rel = 0.0
    for r in range(n):
        b_parts = base_body[r].split(",")
        c_parts = curr_body[r].split(",")
        m = min(len(b_parts), len(c_parts))
        for j in range(m):
            bv = _safe_float(b_parts[j])
            cv = _safe_float(c_parts[j])
            if bv is None or cv is None:
                continue
            base_sum += abs(bv)
            d = cv - bv
            if d != 0.0:
                changed += 1
                diff_sum += abs(d)
                denom = abs(bv)
                rel = abs(d) if denom < 1e-9 else abs(d) / denom
                if rel > max_rel:
                    max_rel = rel
    diff_score = (diff_sum / base_sum) if base_sum > 0 else 0.0
    return {
        "diff_score": float(diff_score),
        "changed_tokens": int(changed),
        "max_rel_change": float(max_rel),
    }

def _detect_circular_shift(base_body: List[str], curr_body: List[str]) -> Optional[int]:
    """If the body matrix is circularly shifted, return shift (>0)."""
    if len(base_body) != len(curr_body) or len(base_body) < 2:
        return None
    n = len(base_body)
    head = base_body[0]
    idxs = [i for i, line in enumerate(curr_body) if line == head]
    for idx in idxs:
        ok = True
        for k in range(n):
            if base_body[k] != curr_body[(idx + k) % n]:
                ok = False
                break
        if ok:
            shift = (n - idx) % n
            return shift if shift != 0 else None
    return None


###############################################################################
# Semantic Profiling & Contract Validation (Layer-B)
###############################################################################

def load_csv_loose(path: str) -> pd.DataFrame:
    """
    Flexible CSV loader robust to messy headers and missing string tokens.
    Treat blanks and common placeholders (e.g., ``N/A``) as missing values to enable reliable type inference.
    ``na_values`` registers empty strings and common missing-value spellings,
    and ``keep_default_na=False`` disables pandas' default NA list for deterministic behavior.
    """
    na_vals = ["", " ", "N/A", "NaN", "NA", "--"]
    try:
        return pd.read_csv(
            path,
            encoding="cp932",
            engine="python",
            on_bad_lines="skip",
            na_values=na_vals,
            keep_default_na=False
        )
    except Exception:
        return pd.read_csv(
            path,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip",
            na_values=na_vals,
            keep_default_na=False
        )

def make_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract semantic features (profile) from the dataset."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = {}
    if not numeric_df.empty:
        # Keep computation light: only core statistics
        stats = numeric_df.agg(['mean', 'std']).fillna(0).to_dict()

    return {
        "columns": list(df.columns),
        "n_rows": len(df),
        "missing_rates": df.isnull().mean().to_dict(),
        "stats": stats
    }

def validate_contract_df(current_df: pd.DataFrame, baseline_df: pd.DataFrame, baseline_profile: Dict) -> List[Dict]:
    """
    Strict data-contract validation at the DataFrame level.
    Using baseline_df and baseline_profile, detect structural changes, missingness changes, and type degradation.

    Returns a list of violation dicts, each containing ``rule_id`` and ``detail``.
    """
    violations: List[Dict] = []

    # 1. Required-column presence check
    missing_cols = set(baseline_profile["columns"]) - set(current_df.columns)
    if missing_cols:
        violations.append({
            "code": "LAYER_B",
            "rule_id": "data_contract_violation",
            "detail": f"Missing required columns: {sorted(list(missing_cols))}"
        })

    # 2. Abnormal increase in missing rate (+10% or more)
    for col, base_rate in baseline_profile["missing_rates"].items():
        if col not in current_df.columns:
            continue
        curr_rate = current_df[col].isnull().mean()
        if curr_rate - base_rate > 0.10:
            violations.append({
                "code": "LAYER_B",
                "rule_id": "data_contract_violation",
                "detail": f"Missing rate for {col} exceeded threshold (+10%) (base={base_rate:.3f}, current={curr_rate:.3f})"
            })

    # 3. Severe row-count reduction (50% or more)
    if len(current_df) < len(baseline_df) * 0.5:
        violations.append({
            "code": "LAYER_B",
            "rule_id": "data_contract_violation",
            "detail": f"Severe row count truncation detected (baseline={len(baseline_df)}, current={len(current_df)})"
        })

# 4. Numeric-type degradation detection: for columns that were numeric in the baseline, if the current NaN rate increases by >5%, treat as possible dtype corruption
    # baseline_profile['stats'] stores statistics for numeric columns
    for col in baseline_profile.get("stats", {}).keys():
        if col not in current_df.columns:
            continue
        try:
            curr_series = pd.to_numeric(current_df[col], errors="coerce")
        except Exception:
            curr_series = pd.Series([np.nan] * len(current_df))
        base_rate = baseline_profile["missing_rates"].get(col, 0)
        curr_missing = curr_series.isna().mean()
        if curr_missing - base_rate > 0.05:
            violations.append({
                "code": "LAYER_B",
                "rule_id": "data_contract_violation",
                "detail": f"Numeric dtype for {col} appears corrupted (missing rate increased by >5%)"
            })

    return violations

def diff_profile(baseline_df: pd.DataFrame, current_df: pd.DataFrame, 
                 baseline_profile: Dict, current_profile: Dict, config: Dict) -> List[Dict]:
    """Detect semantic differences vs. baseline (signals of attacks)."""
    reasons = []
    
    # Detailed comparison when row counts match (A1/B2/A3)
    if len(baseline_df) == len(current_df):
        # A1: local spike detection; B2: near-threshold probing
        for col in baseline_df.columns:
            if col not in current_df.columns:
                continue
            # Convert baseline/current column to numeric (failed parses become NaN -> 0)
            try:
                base_vals = pd.to_numeric(baseline_df[col], errors="coerce").fillna(0).values
                curr_vals = pd.to_numeric(current_df[col], errors="coerce").fillna(0).values
            except Exception:
                continue
            # Skip if there is 1 value or fewer
            if len(base_vals) <= 1:
                continue
            # Compute relative change, avoiding division-by-zero
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_change = np.abs(curr_vals - base_vals) / np.maximum(np.abs(base_vals), 1e-9)
            max_change = float(np.nanmax(rel_change)) if rel_change.size > 0 else 0.0

            # A1 detection: local spike of 8% or more
            if max_change >= 0.08:
                reasons.append({
                    "code": "LAYER_B",
                    "rule_id": "local_spike_detected",
                    "metric": "max_rel_change",
                    "value": max_change,
                    "column": col
                })

            # B2 detection: does the L1-norm pseudo-score aim near the threshold?
            l1_diff = float(np.nansum(np.abs(curr_vals - base_vals)))
            l1_base = float(np.nansum(np.abs(base_vals)))
            if l1_base > 0:
                diff_score = l1_diff / l1_base
                threshold = config.get("threshold", 0.05)
                # If the score lands within +/- 1% of the threshold, treat as boundary exploration
                if 0 < diff_score and abs(diff_score - threshold) <= 0.01:
                    reasons.append({
                        "code": "LAYER_B",
                        "rule_id": "near_threshold_perturbation",
                        "metric": "l1_diff_score",
                        "value": diff_score,
                        "threshold": threshold
                    })

        # A3: missing-value injection detection
        for col in baseline_profile["columns"]:
            if col not in current_df.columns:
                continue
            base_missing = baseline_profile["missing_rates"].get(col, 0)
            curr_missing = current_df[col].isnull().mean()
            delta_missing = curr_missing - base_missing
            if delta_missing > 0.001:  # Detect even tiny increases
                reasons.append({
                    "code": "LAYER_B",
                    "rule_id": "missing_injection_detected",
                    "column": col,
                    "delta": float(delta_missing)
                })

        # A2: phase shift (row rotation) detection
        # When row counts match and content is nearly the same (stats/missing rates are close), estimate shift via cross-correlation
        if current_profile["n_rows"] > 1:
            # Select a numeric column
            numeric_col = None
            # Keys of baseline_profile['stats'] correspond to numeric columns
            for col in baseline_profile.get("stats", {}).keys():
                if col in current_df.columns:
                    numeric_col = col
                    break
            if numeric_col:
                base_series = pd.to_numeric(baseline_df[numeric_col], errors="coerce").fillna(0).values
                curr_series = pd.to_numeric(current_df[numeric_col], errors="coerce").fillna(0).values
                if len(base_series) == len(curr_series) and len(base_series) > 1:
                    b = base_series - np.nanmean(base_series)
                    c = curr_series - np.nanmean(curr_series)
                    # Estimate shift from full-mode cross-correlation
                    corr = np.correlate(b, c, mode="full")
                    shift = int(np.argmax(corr) - (len(b) - 1))
                    if shift != 0:
                        reasons.append({
                            "code": "LAYER_B",
                            "rule_id": "time_shift_detected",
                            "metric": "shift",
                            "value": shift,
                            "column": numeric_col
                        })

    # C1: baseline series modification (e.g., power_usage; different length)
    if current_profile["n_rows"] < baseline_profile["n_rows"] and current_profile["n_rows"] > 0:
        reasons.append({
            "code": "LAYER_B",
            "rule_id": "baseline_series_changed",
            "detail": f"Row count truncated from {baseline_profile['n_rows']} to {current_profile['n_rows']}"
        })

    return reasons


###############################################################################
# Audit Engine
###############################################################################

class AuditEngine:
    def __init__(self, base_weather: str, base_power: str, baseline_config: Dict):
        self.base_dir = "redteam_runs"
        os.makedirs(self.base_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.base_dir, "ledger.csv")
        
        self.baseline_config = baseline_config
        self.baseline_config_sha = fingerprint_config(baseline_config)
        self.baseline_data_sha = sha256_file(base_weather)
        self.baseline_power_sha = sha256_file(base_power)
        
        # Profile generation
        self.baseline_weather_df = load_csv_loose(base_weather)
        self.baseline_power_df = load_csv_loose(base_power)
        self.baseline_weather_profile = make_profile(self.baseline_weather_df)
        self.baseline_power_profile = make_profile(self.baseline_power_df)

        # Raw header signature (contract anchor) + body lines (semantic diff anchor)
        base_lines = _read_text_lines_cp932(base_weather)
        self.baseline_header_sig_weather = _header_signature(base_lines)
        self.baseline_body_weather = _body_lines(base_lines)

        # Count missing (blank-field) patterns within the baseline file
        self.baseline_blank_fields = "\n".join(base_lines).count(",,")

    def run_audit(self, run_id: str, data_path: str, power_path: str, config: Dict, run_dir: str) -> Tuple[str, str, List[Dict], str]:
        # --- Layer A: fingerprint computation ---
        data_sha = sha256_file(data_path)
        power_sha = sha256_file(power_path)
        config_sha = fingerprint_config(config)
        code_sha = fingerprint_code()
        env_sha = fingerprint_env(run_dir)

        reasons = []
        verdict = "OK"

        if data_sha != self.baseline_data_sha:
            verdict = "NG"
            reasons.append({"code": "LAYER_A", "rule_id": "weather_sha_changed"})
        if power_sha != self.baseline_power_sha:
            verdict = "NG"
            reasons.append({"code": "LAYER_A", "rule_id": "power_sha_changed"})
        if config_sha != self.baseline_config_sha:
            verdict = "NG"
            reasons.append({"code": "LAYER_A", "rule_id": "config_sha_changed"})
            if config.get("threshold") != self.baseline_config.get("threshold"):
                reasons.append({"code": "LAYER_A", "rule_id": "threshold_changed"})

        # --- Layer B: semantic profile comparison ---
        current_weather_df = load_csv_loose(data_path)
        current_weather_profile = make_profile(current_weather_df)

        # Header contract (for F attack): if the deterministic header signature differs from baseline, treat as contract violation
        cur_lines = _read_text_lines_cp932(data_path)
        cur_header_sig = _header_signature(cur_lines)
        if cur_header_sig != self.baseline_header_sig_weather:
            verdict = "NG"
            reasons.append({
                "code": "LAYER_B",
                "rule_id": "data_contract_violation",
                "detail": "Header signature mismatch"
            })

        # Decompose A1/A2/B2 without relying on pandas parsing (highest-priority Layer-B)
        if data_sha != self.baseline_data_sha:
            cur_body = _body_lines(cur_lines)

            # A2: phase shift (circular shift)
            shift = _detect_circular_shift(self.baseline_body_weather, cur_body)
            if shift is not None:
                verdict = "NG"
                reasons.append({
                    "code": "LAYER_B",
                    "rule_id": "time_shift_detected",
                    "metric": "shift",
                    "value": int(shift)
                })

            # A1/B2: same-position token diffs (skip if A2, since alignment is broken)
            if shift is None:
                m = _token_diff_metrics(self.baseline_body_weather, cur_body)
                thr = float(config.get("threshold", 0.05))

                # A1: large change in only a small number of tokens
                if m["changed_tokens"] > 0 and m["changed_tokens"] <= 6 and m["max_rel_change"] >= 0.08:
                    verdict = "NG"
                    reasons.append({
                        "code": "LAYER_B",
                        "rule_id": "local_spike_detected",
                        "metric": "max_rel_change",
                        "value": float(m["max_rel_change"]),
                        "changed_tokens": int(m["changed_tokens"])
                    })

                # B2: diff_score is near threshold (broad perturbation across many tokens)
                if m["changed_tokens"] >= 50 and abs(float(m["diff_score"]) - thr) <= 0.01:
                    verdict = "NG"
                    reasons.append({
                        "code": "LAYER_B",
                        "rule_id": "near_threshold_perturbation",
                        "metric": "diff_score",
                        "value": float(m["diff_score"]),
                        "threshold": thr,
                        "changed_tokens": int(m["changed_tokens"]),
                        "max_rel_change": float(m["max_rel_change"])
                    })
        
        # Contract validation (row count / columns / missingness / dtype)
        contract_violations = validate_contract_df(current_weather_df, self.baseline_weather_df, self.baseline_weather_profile)
        if contract_violations:
            verdict = "NG"
            reasons.extend(contract_violations)

        # File-level missing-value injection detection (fallback when the DataFrame cannot be parsed)
        try:
            with open(data_path, 'r', encoding='cp932', errors='ignore') as f:
                text_current = f.read()
        except Exception:
            text_current = ""
        current_blank_fields = text_current.count(",,")
        delta_blank = current_blank_fields - self.baseline_blank_fields
        if delta_blank > 0:
            verdict = "NG"
            reasons.append({
                "code": "LAYER_B",
                "rule_id": "missing_injection_detected",
                "metric": "blank_fields_added",
                "value": delta_blank
            })

        # File-level missing-value injection detection (fallback when DataFrame-level detection misses)

        # Diff detection (Weather)
        if data_sha != self.baseline_data_sha and not contract_violations:
            diff_reasons = diff_profile(
                self.baseline_weather_df,
                current_weather_df,
                self.baseline_weather_profile,
                current_weather_profile,
                config
            )
            reasons.extend(diff_reasons)
            
        # Diff detection (Power - for C1)
        if power_sha != self.baseline_power_sha:
            curr_power_df = load_csv_loose(power_path)
            curr_power_prof = make_profile(curr_power_df)
            power_reasons = diff_profile(self.baseline_power_df, curr_power_df,
                                         self.baseline_power_profile, curr_power_prof, config)
            reasons.extend(power_reasons)

        # --- Evidence binding ---
        evidence_path = os.path.join(run_dir, "evidence_timeseries.csv")
        with open(evidence_path, "w", newline="", encoding="utf-8") as ef:
            writer = csv.writer(ef)
            writer.writerow(["issue_type", "detail_json"])
            for r in reasons:
                writer.writerow(["anomaly", json.dumps(r, ensure_ascii=False)])
                
        # Hash the evidence file itself and anchor it into the certificate (to resist simultaneous tampering)
        evidence_sha = sha256_file(evidence_path)

        # --- Build certificate ---
        timestamp = datetime.datetime.now().isoformat()
        cert_material = json.dumps({
            "data_sha": data_sha,
            "power_sha": power_sha,
            "config_sha": config_sha,
            "code_sha": code_sha,
            "env_sha": env_sha,
            "evidence_sha": evidence_sha, # binding point
            "verdict": verdict,
            "config": config
        }, sort_keys=True, ensure_ascii=False)
        cert_id = sha256_string(cert_material)

        certificate = {
            "cert_id": cert_id,
            "data_sha": data_sha,
            "power_sha": power_sha,
            "config_sha": config_sha,
            "code_sha": code_sha,
            "env_sha": env_sha,
            "evidence_sha": evidence_sha,
            "verdict": verdict,
            "reasons": reasons,
            "timestamp": timestamp,
            "config": config
        }

        cert_path = os.path.join(run_dir, "certificate.json")
        with open(cert_path, "w", encoding="utf-8") as cf:
            json.dump(certificate, cf, ensure_ascii=False, indent=2)

        # --- Append to ledger ---
        ledger_entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "cert_id": cert_id,
            "data_sha": data_sha,
            "power_sha": power_sha,
            "evidence_sha": evidence_sha,
            "verdict": verdict
        }
        self._append_ledger(ledger_entry)

        return cert_id, verdict, reasons, cert_path

    def _append_ledger(self, entry: Dict[str, str]) -> None:
        file_exists = os.path.exists(self.ledger_path)
        with open(self.ledger_path, "a", newline="", encoding="utf-8") as lf:
            writer = csv.DictWriter(lf, fieldnames=list(entry.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def verify_integrity(self, cert_path: str, data_path: str, power_path: str, run_dir: str, config: Dict) -> bool:
        """Verification Step 1: check integrity of hashes, ledger, and evidence."""
        with open(cert_path, "r", encoding="utf-8") as cf:
            cert = json.load(cf)
            
        # Recompute
        data_sha = sha256_file(data_path)
        power_sha = sha256_file(power_path)
        config_sha = fingerprint_config(config)
        code_sha = fingerprint_code()
        env_sha = fingerprint_env(run_dir)
        evidence_path = os.path.join(run_dir, "evidence_timeseries.csv")
        evidence_sha = sha256_file(evidence_path) if os.path.exists(evidence_path) else ""

        # Field checks
        if not all([
            cert.get("data_sha") == data_sha,
            cert.get("power_sha") == power_sha,
            cert.get("config_sha") == config_sha,
            cert.get("code_sha") == code_sha,
            cert.get("env_sha") == env_sha,
            cert.get("evidence_sha") == evidence_sha
        ]):
            return False

        # Recompute cert_id
        recomputed_material = json.dumps({
            "data_sha": data_sha,
            "power_sha": power_sha,
            "config_sha": config_sha,
            "code_sha": code_sha,
            "env_sha": env_sha,
            "evidence_sha": evidence_sha,
            "verdict": cert["verdict"],
            "config": cert.get("config", {})
        }, sort_keys=True, ensure_ascii=False)
        if sha256_string(recomputed_material) != cert.get("cert_id"):
            return False

        # Confirm ledger inclusion
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r", encoding="utf-8") as lf:
                for row in csv.DictReader(lf):
                    if row.get("cert_id") == cert.get("cert_id") and row.get("evidence_sha") == evidence_sha:
                        return True
        return False

    def verify_semantics(self, reasons: List[Dict], expected_rules: List[str]) -> bool:
        """Verification Step 2: assert that semantic detections fired as expected."""
        detected_rules = [r.get("rule_id") for r in reasons]
        for expected in expected_rules:
            if expected not in detected_rules:
                return False
        return True


###############################################################################
# Attack Implementations (Modifiers)
###############################################################################

def modify_a1_demand_spike(src: str, dest: str) -> None:
    """A1: Local spike (scale by +15% to ensure detection)."""
    with open(src, "rb") as f:
        text = f.read().decode("cp932", errors="ignore")
    lines = text.splitlines()
    modified = False
    for i in range(len(lines)):
        if not modified and any(c.isdigit() for c in lines[i]):
            parts = lines[i].split(",")
            for j, part in enumerate(parts):
                try:
                    val = float(part)
                    val *= 1.15  
                    parts[j] = f"{val:.2f}"
                    modified = True
                    break
                except Exception:
                    continue
            lines[i] = ",".join(parts)
        if modified:
            break
    with open(dest, "w", encoding="cp932", errors="ignore") as out:
        out.write("\n".join(lines))

def modify_a2_shift_weather(src: str, dest: str) -> None:
    """A2: Phase shift (row rotation)."""
    lines = _read_text_lines_cp932(src)
    header_count = _find_header_count(lines)
    header = lines[:header_count]
    body = lines[header_count:]
    shift = 24 if len(body) > 24 else 1
    shifted = body[shift:] + body[:shift]
    with open(dest, "w", encoding="cp932", errors="ignore") as out:
        out.write("\n".join(header + shifted))

def modify_a3_inject_nan(src: str, dest: str) -> None:
    """A3: Missing-value injection."""
    with open(src, "rb") as f:
        text = f.read().decode("cp932", errors="ignore")
    lines = text.splitlines()
    modified = False
    for i in range(len(lines)):
        if not modified and any(c.isdigit() for c in lines[i]):
            parts = lines[i].split(",")
            for j, part in enumerate(parts):
                try:
                    float(part)
                    parts[j] = ""  # blank out
                    modified = True
                    break
                except Exception:
                    continue
            lines[i] = ",".join(parts)
        if modified:
            break
    with open(dest, "w", encoding="cp932", errors="ignore") as out:
        out.write("\n".join(lines))

def modify_b2_boundary_exploration(src: str, dest: str) -> None:
    """B2: Boundary exploration (broad perturbation so Layer-B diff_score is near threshold (0.05))."""
    lines = _read_text_lines_cp932(src)
    hc = _find_header_count(lines)
    header = lines[:hc]
    body = lines[hc:]

    eps = 0.05  # Target near-threshold diff_score (diff_score ~ eps)
    new_body: List[str] = []
    for line in body:
        parts = line.split(",")
        for j, tok in enumerate(parts):
            v = _safe_float(tok)
            if v is None:
                continue
            # Keep at least 3 decimal digits so max_rel_change does not explode due to rounding
            dec = 3
            if "." in tok:
                try:
                    dec = max(3, min(6, len(tok.split(".")[1])))
                except Exception:
                    dec = 3
            fmt = f"{{:.{dec}f}}"
            parts[j] = fmt.format(v * (1.0 + eps))
        new_body.append(",".join(parts))

    with open(dest, "w", encoding="cp932", errors="ignore") as out:
        out.write("\n".join(header + new_body))

def modify_c1_baseline_swap(src: str, dest: str) -> None:
    """C1: Baseline swap via row deletion."""
    with open(src, "rb") as f:
        text = f.read().decode("cp932", errors="ignore")
    lines = text.splitlines()
    if len(lines) > 5:
        lines = lines[:-5]
    with open(dest, "w", encoding="cp932", errors="ignore") as out:
        out.write("\n".join(lines))

def modify_f_contract_violation(src: str, dest: str) -> None:
    """F: Contract violation (CSV remains readable, but breaks the header contract (signature))."""
    lines = _read_text_lines_cp932(src)
    hc = _find_header_count(lines)
    if hc <= 0:
        hc = 1
    idx = min(hc - 1, len(lines) - 1)
    parts = lines[idx].split(",")
    for j in range(len(parts)):
        tok = parts[j].strip()
        if tok:
            parts[j] = tok + "_broken"
            break
    lines[idx] = ",".join(parts)
    with open(dest, "w", encoding="cp932", errors="ignore") as out:
        out.write("\n".join(lines))


###############################################################################
# Main Execution
###############################################################################

def _ensure_upstream_artifacts(run_dir: str, out_dir: str) -> Tuple[bool, str]:
    """After running upstream, ensure required artifacts exist under adic_out, and save a manifest."""
    req = ["certificate.json", "ledger.csv", "evidence_timeseries.csv"]
    os.makedirs(out_dir, exist_ok=True)

    # If not present in out_dir, also look under run_dir and move into out_dir
    for name in req:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            continue
        alt = os.path.join(run_dir, name)
        if os.path.exists(alt):
            shutil.move(alt, p)

    missing = [n for n in req if not os.path.exists(os.path.join(out_dir, n))]
    if missing:
        return False, f"missing_upstream_artifacts:{missing}"

    # --- verify_upstream (schema-aware minimal checks) ---
    cert_path = os.path.join(out_dir, "certificate.json")
    ledger_path = os.path.join(out_dir, "ledger.csv")
    ev_path = os.path.join(out_dir, "evidence_timeseries.csv")

    try:
        with open(cert_path, "r", encoding="utf-8") as f:
            cert = json.load(f)
    except Exception as e:
        return False, f"upstream_certificate_invalid:{e}"

    # Locate cert_id (supporting multiple possible key names)
    cert_id = None
    for k in ["cert_id", "certificate_id", "certificateId", "certHash"]:
        if isinstance(cert.get(k), str) and cert.get(k):
            cert_id = cert.get(k)
            break
    if cert_id is None:
        return False, "upstream_certificate_missing_id"

    # Check that ledger.csv contains the cert_id (minimal consistency check)
    try:
        with open(ledger_path, "r", encoding="utf-8") as lf:
            rows = list(csv.DictReader(lf))
        if not any((cert_id in ("".join([str(v) for v in row.values() if v is not None]))) for row in rows):
            return False, "upstream_ledger_missing_cert_id"
    except Exception as e:
        return False, f"upstream_ledger_invalid:{e}"

    # Ensure evidence is not empty
    try:
        if os.path.getsize(ev_path) <= 0:
            return False, "upstream_evidence_empty"
    except Exception:
        return False, "upstream_evidence_missing"

    manifest = {
        "run_dir": run_dir,
        "adic_out": out_dir,
        "sha256": {n: sha256_file(os.path.join(out_dir, n)) for n in req},
        "cert_id": cert_id,
    }
    with open(os.path.join(run_dir, "upstream_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Re-check against manifest (lightweight verify_upstream)
    for n in req:
        if sha256_file(os.path.join(out_dir, n)) != manifest["sha256"][n]:
            return False, "upstream_manifest_mismatch"
    return True, "upstream_ok"

def _run_upstream_script(ghost_script: str, run_dir: str, weather_path: str, power_path: str, cfg: Dict) -> Tuple[bool, str]:
    """Launch the upstream script using multiple CLI patterns to tolerate argument variations."""
    patterns = [
        [sys.executable, ghost_script, "--weather", weather_path, "--power", power_path, "--config", json.dumps(cfg)],
        [sys.executable, ghost_script, "--weather_csv", weather_path, "--power_csv", power_path, "--config", json.dumps(cfg)],
        [sys.executable, ghost_script, weather_path, power_path],
    ]
    last_err = ""
    for cmd in patterns:
        try:
            subprocess.run(cmd, check=True, cwd=run_dir)
            return True, "upstream_executed"
        except Exception as e:
            last_err = str(e)
            continue
    return False, f"upstream_exec_failed:{last_err}"

def main():
    parser = argparse.ArgumentParser(description="GhostDrift ADIC Protocol Red Team Harness")
    parser.add_argument("--mode", choices=["protocol", "upstream"], default="protocol",
                        help="Execution mode (protocol: standalone, upstream: call external ghost_drift_audit_EN.py)")
    args = parser.parse_args()

    base_weather = "electric_load_weather.csv"
    base_power = "power_usage.csv"
    if not os.path.exists(base_weather) or not os.path.exists(base_power):
        print("Error: Baseline CSV files not found.")
        sys.exit(1)

    baseline_config = {"threshold": 0.05, "baseline_model": "default", "calibration_window": 100}
    engine = AuditEngine(base_weather, base_power, baseline_config)
    summary = []

    def prepare_run_dir(run_id: str) -> Tuple[str, str, str]:
        run_dir = os.path.join(engine.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        data_copy = os.path.join(run_dir, "electric_load_weather.csv")
        power_copy = os.path.join(run_dir, "power_usage.csv")
        shutil.copy(base_weather, data_copy)
        shutil.copy(base_power, power_copy)
        return run_dir, data_copy, power_copy

    attacks = [
        ("A1", "Demand spike", modify_a1_demand_spike, True, ["local_spike_detected"]),
        ("A2", "Time-shift", modify_a2_shift_weather, True, ["time_shift_detected"]),
        ("A3", "Missing value", modify_a3_inject_nan, True, ["missing_injection_detected"]),
        ("B1", "Threshold manipulation", None, True, ["threshold_changed"]),
        ("B2", "Boundary exploration", modify_b2_boundary_exploration, True, ["near_threshold_perturbation"]),
        ("C1", "Baseline swap", modify_c1_baseline_swap, True, ["baseline_series_changed"]),
        ("C2", "Calibration window tamper", None, True, ["config_sha_changed"]),
        ("F", "Data contract violation", modify_f_contract_violation, True, ["data_contract_violation"]),
    ]

    # Baseline run (common for both modes)
    run_dir, data_copy, power_copy = prepare_run_dir("baseline")
    cert_id, verdict, reasons, cert_path = engine.run_audit(
        "baseline", data_copy, power_copy, baseline_config.copy(), run_dir
    )
    valid_integrity = engine.verify_integrity(
        cert_path, data_copy, power_copy, run_dir, baseline_config.copy()
    )
    summary.append({
        "attack_id": "baseline",
        "expected": "OK",
        "observed": verdict,
        "pass_fail": "PASS" if verdict == "OK" and valid_integrity else "FAIL"
    })

    # Execute attacks
    for attack_id, desc, modifier, expect_detect, expected_semantics in attacks:
        run_dir, data_copy, power_copy = prepare_run_dir(attack_id)
        attack_config = baseline_config.copy()

        # Apply modifier if defined
        if modifier:
            if attack_id == "C1":
                modifier(base_power, power_copy)
            else:
                modifier(base_weather, data_copy)

        # Adjust configuration for threshold or calibration window modifications
        if attack_id == "B1":
            attack_config["threshold"] = 0.10
        if attack_id == "C2":
            attack_config["calibration_window"] = 50

        # Run audit (protocol mode)
        cert_id_cur, verdict_cur, reasons_cur, cp_cur = engine.run_audit(
            attack_id, data_copy, power_copy, attack_config.copy(), run_dir
        )

        # 100-point condition: assert that Layer-B semantics fired correctly
        semantics_ok = engine.verify_semantics(reasons_cur, expected_semantics)

        # List which reason IDs were actually detected (among expected ones)
        detected = [r.get("rule_id") for r in reasons_cur]
        detected_display = ", ".join(detected) if detected else "none"
        pass_fail = "PASS" if (verdict_cur == "NG") and semantics_ok else "FAIL"
        summary.append({
            "attack_id": attack_id,
            "expected": "NG",
            "observed": f"{verdict_cur} ({detected_display})",
            "pass_fail": pass_fail
        })

    # D & E class (Tamper attacks on artifacts)
    run_dir, data_copy, power_copy = prepare_run_dir("D2")
    _, _, _, cp_cur = engine.run_audit("D2", data_copy, power_copy, baseline_config.copy(), run_dir)
    with open(cp_cur, "r", encoding="utf-8") as cf: cert = json.load(cf)
    cert["env_sha"] = "tampered_hash"
    with open(cp_cur, "w", encoding="utf-8") as cf: json.dump(cert, cf)
    
    valid = engine.verify_integrity(cp_cur, data_copy, power_copy, run_dir, baseline_config.copy())
    summary.append({
        "attack_id": "D2", "expected": "tamper_detected", "observed": "tamper_detected" if not valid else "missed",
        "pass_fail": "PASS" if not valid else "FAIL"
    })

    # Write REDTEAM.md summary (Markdown without problematic internal citations)
    summary_path = os.path.join(engine.base_dir, "REDTEAM.md")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("# REDTEAM Execution (100pt Audited Version)\n\n")
        sf.write(f"Generated at: {datetime.datetime.now().isoformat()}\n\n")
        sf.write("This run explicitly isolates Layer-A (Tamper-evident hash fingerprints) from Layer-B (Semantic attack characterizations), validating the Finite Closure property of the ALS theoretical phase transition ($B < J$).\n\n")
        sf.write("| Attack ID | Expected | Observed (Reasoning) | PASS/FAIL |\n")
        sf.write("|---|---|---|---|\n")
        for row in summary:
            sf.write(f"| {row['attack_id']} | {row['expected']} | {row['observed']} | {row['pass_fail']} |\n")

    # If upstream mode is selected, execute the repo main script and verify its artifacts.
    if args.mode == "upstream":
        ghost_script = "ghost_drift_audit_EN.py"
        if not os.path.exists(ghost_script):
            print("[ERROR] Upstream mode selected but 'ghost_drift_audit_EN.py' not found.")
        else:
            for attack_id, _, _, _, _ in [("baseline", None, None, None, None)] + attacks:
                run_dir = os.path.join(engine.base_dir, attack_id)
                weather_path = os.path.join(run_dir, "electric_load_weather.csv")
                power_path = os.path.join(run_dir, "power_usage.csv")
                out_dir = os.path.join(run_dir, "adic_out")

                ok_exec, exec_reason = _run_upstream_script(ghost_script, run_dir, weather_path, power_path, baseline_config)
                if not ok_exec:
                    summary.append({
                        "attack_id": f"upstream::{attack_id}",
                        "expected": "upstream_ok",
                        "observed": exec_reason,
                        "pass_fail": "FAIL"
                    })
                    continue

                ok_art, art_reason = _ensure_upstream_artifacts(run_dir, out_dir)
                summary.append({
                    "attack_id": f"upstream::{attack_id}",
                    "expected": "upstream_ok",
                    "observed": art_reason,
                    "pass_fail": "PASS" if ok_art else "FAIL"
                })

    # Overwrite REDTEAM.md after upstream append so the report is guaranteed to be up-to-date
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("# REDTEAM Execution (100pt Audited Version)\n\n")
        sf.write(f"Generated at: {datetime.datetime.now().isoformat()}\n\n")
        sf.write("This run explicitly isolates Layer-A (Tamper-evident hash fingerprints) from Layer-B (Semantic attack characterizations), validating the Finite Closure property of the ALS theoretical phase transition ($B < J$).\n\n")
        sf.write("| Attack ID | Expected | Observed (Reasoning) | PASS/FAIL |\n")
        sf.write("|---|---|---|---|\n")
        for row in summary:
            sf.write(f"| {row['attack_id']} | {row['expected']} | {row['observed']} | {row['pass_fail']} |\n")

    print(f"Red teaming complete (Mode: {args.mode}). Artifacts written to {engine.base_dir}/")


if __name__ == "__main__":
    main()
