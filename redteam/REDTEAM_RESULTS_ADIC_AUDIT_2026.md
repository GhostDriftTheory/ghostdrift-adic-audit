# ADIC Demo Red Team Report (ghostdrift-adic-audit)

Generated at: **2026-03-01 (Asia/Tokyo)**

## 1. Purpose
This report summarizes the results of red teaming (adversarial scenario testing) conducted on the **ADIC audit demo** in the GitHub repository **ghostdrift-adic-audit**, to verify:

- **Recomputability**: a third party can rerun the protocol and reach the same conclusion.
- **Responsibility Fixation**: tampering/attacks can be detected and a “NG” (non-compliant) result can be deterministically fixed.

- Target repository: https://github.com/GhostDriftTheory/ghostdrift-adic-audit  
- Note: CSV URLs in the repository are **dummy data** and are **not used**.  
- This test uses **only** the attached **real input CSVs** (`electric_load_weather.csv` / `power_usage.csv`).

## 2. Scope (What This Report Guarantees / Does Not Guarantee)

### What it guarantees
- **Layer-A (Fingerprinting)**: if input data / config / code / environment / evidence changes, the run becomes **NG**.
- **Layer-B (Semantics)**: for each attack type, an **attack-specific detection reason** is produced.
- **Evidence Binding**: the SHA of the evidence is anchored to the certificate and ledger, and can be rechecked via `verify`.

### What it does NOT guarantee
- Model quality, predictive performance, or correctness of training.
- Final irreversible fixation via external anchoring (e.g., third-party timestamps / public ledgers).
  - This demo targets **local artifact integrity** (strengthened resistance against “simultaneous tampering” within local outputs).

## 3. Artifacts (Red Team Harness)
- Script: `run_redteam_script_final.py`
- Inputs: `electric_load_weather.csv`, `power_usage.csv`
- Output directory: `redteam_runs/`

Outputs (inside each `run_id` directory)
- `certificate.json` (certificate)
- `ledger.csv` (ledger; aggregated under `redteam_runs/`)
- `evidence_timeseries.csv` (reasons saved as JSON within CSV)
- `env_info.json` (environment fingerprint details)

## 4. Attack Scenarios
This test executes the following attack classes.

### A. Data attacks (Weather side)
- **A1 Demand spike**
  - Goal: local spike (sharp change in a small number of tokens)
  - Expected: `local_spike_detected`
- **A2 Time-shift**
  - Goal: circular shift of the time-series body
  - Expected: `time_shift_detected`
- **A3 Missing injection**
  - Goal: inject missing values (blank cells)
  - Expected: `missing_injection_detected`

### B. Configuration attacks
- **B1 Threshold manipulation**
  - Goal: unauthorized threshold modification
  - Expected: `threshold_changed`
- **B2 Boundary exploration**
  - Goal: broad perturbation targeting near-boundary states (diff_score ≈ threshold)
  - Expected: `near_threshold_perturbation`

### C. Data attacks (Power side)
- **C1 Baseline swap**
  - Goal: truncate series length (baseline substitution)
  - Expected: `baseline_series_changed`
- **C2 Calibration window tamper**
  - Goal: unauthorized modification of `calibration_window`
  - Expected: `config_sha_changed`

### F. Data contract violation (Readable but contract-breaching)
- **F Data contract violation**
  - Goal: CSV remains parseable, but header contract (header signature) is broken
  - Expected: `data_contract_violation`

### D. Output tampering
- **D2 Certificate tamper**
  - Goal: tamper with `env_sha` in the certificate and ensure verify fails
  - Expected: `tamper_detected`

## 5. Detection Design (Key Requirement Mapping)

### 5.1 Layer-A (Fingerprint / tamper-evident)
- `weather_sha_changed`, `power_sha_changed`, `config_sha_changed`, `threshold_changed`
- Include `code_sha`, `env_sha` in the certificate

### 5.2 Layer-B (Semantics / attack-specific)
- Introduce **raw CSV parsing** that does **not** depend on pandas parsing
  - allow multi-row headers
  - detect circular time shifts (A2)
  - detect A1/B2 via token-level diffs (diff_score / max_rel_change / changed_tokens)

### 5.3 Evidence Binding (Strengthening resistance to simultaneous tampering)
- Compute `evidence_sha`, then:
  - store it in `certificate.json`
  - store it in `ledger.csv`
  - recheck via `verify_integrity()`

## 6. How to Run (Reproduction Steps)
Place the following three files in the same directory and run:

- `run_redteam_script_final.py`
- `electric_load_weather.csv`
- `power_usage.csv`

Command:
```bash
python run_redteam_script_final.py
```

Outputs:
- Artifacts are generated under `redteam_runs/`.
- The summary table is written to `redteam_runs/REDTEAM.md`.

(Optional) Upstream run:
- If `ghost_drift_audit_EN.py` exists at the repository root, you can start upstream with:
```bash
python run_redteam_script_final.py --mode upstream
```
This collects artifacts under `adic_out/` and performs minimal verification (`verify_upstream`).

## 7. Results
Below are the results produced in this environment with `--mode protocol`.

| Attack ID | Expected | Observed (Reasoning) | PASS/FAIL |
|---|---|---|---|
| baseline | OK | OK | PASS |
| A1 | NG | NG (weather_sha_changed, local_spike_detected) | PASS |
| A2 | NG | NG (weather_sha_changed, time_shift_detected) | PASS |
| A3 | NG | NG (weather_sha_changed, missing_injection_detected) | PASS |
| B1 | NG | NG (config_sha_changed, threshold_changed) | PASS |
| B2 | NG | NG (weather_sha_changed, near_threshold_perturbation, near_threshold_perturbation) | PASS |
| C1 | NG | NG (power_sha_changed, baseline_series_changed) | PASS |
| C2 | NG | NG (config_sha_changed) | PASS |
| F | NG | NG (weather_sha_changed, data_contract_violation) | PASS |
| D2 | tamper_detected | tamper_detected | PASS |

## 8. Conclusion (Report Statement)
In this red teaming, we executed attack scenarios A1–F and D2 against the ADIC audit demo protocol (certificate, ledger, and evidence binding). We confirmed that:

- **NG decisions are deterministically fixed** with a clear separation between **Layer-A (tamper fingerprints)** and **Layer-B (attack-specific semantic reasons)**, and
- **evidence is bound** to the certificate and ledger and can be **rechecked via verify**.
