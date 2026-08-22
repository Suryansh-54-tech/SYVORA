# SentinelRisk

SentinelRisk is a decision-support system for post-payment chargeback triage: an operator enters a dispute manually, and the system sanitizes any customer complaint text, engineers a deterministic feature set, produces a calibrated win probability with a Random Forest classifier, explains that score with exact TreeSHAP attributions, weighs the economics of contesting against a non-refundable arbitration fee, applies deterministic policy gates, and returns one of three verdicts — **CONTEST**, **REVIEW**, or **SURRENDER** — together with a provenance-aware evidence dossier and an append-only cryptographic audit trail. Everything runs locally on synthetic data through a Streamlit operations console.

> [!IMPORTANT]
> **This is a synthetic simulation / technical demonstration — not production software.**
>
> - No live bank, issuer, Visa, Mastercard, Razorpay, carrier, or payment-gateway APIs are queried. There are zero external API calls in the codebase.
> - All datasets and telemetry are synthetic.
> - The system must not be used to make real financial or dispute decisions.
> - The generated evidence provenance IDs are simulated.

## What It Does

The complete workflow, executed deterministically for every dispute:

```
Manual dispute input (Streamlit intake form)
   │
   ▼
Customer complaint sanitization ──── prompt-injection firewall;
   │                                 original preserved as labeled untrusted audit data
   ▼
Deterministic feature engineering ── 41 fixed-schema features, leakage-guarded
   │
   ▼
Calibrated ML probability ────────── Random Forest + probability calibration
   │
   ▼
TreeSHAP explanation ─────────────── exact per-feature attributions (probability units)
   │
   ▼
Bayesian Expected Value ──────────── E[EV] vs break-even threshold
   │
   ▼
Deterministic policy gates ───────── amount / confidence / economics /
   │                                 deadline / evidence-readiness
   ▼
CONTEST  |  REVIEW  |  SURRENDER
   │
   ▼
Evidence dossier ─────────────────── observed-vs-derived separation, provenance citations
   │
   ▼
Cryptographic audit ledger ───────── append-only SHA-256 hash chain, optional HMAC signing
```

## Key Capabilities

- 41 engineered features with strict target/metadata leakage guards
- Random Forest classifier with probability calibration (5-fold out-of-fold)
- TreeSHAP explainability with correct unit labeling
- Expected Value decisioning with break-even analysis
- Evidence-readiness scoring (0–100 composite index)
- Deterministic policy gates that force human review regardless of EV or confidence
- Prompt-injection containment for untrusted customer text
- Provenance-aware evidence dossier (Markdown + JSON export)
- SHA-256 hash-chained append-only audit ledger
- Optional HMAC-SHA256 entry authentication (`SENTINEL_AUDIT_SECRET`)
- Streamlit operations console with live triage, manual intake, benchmark, ledger, and sanitizer views
- Manual dispute intake for ad-hoc cases

## ML Evaluation

Evaluated on the project's **synthetic chronological test split** (N=180, held-out by dispute date). These are simulation metrics, not real-world financial performance.

| Metric | Score |
|---|---|
| Test PR-AUC | 0.8347 |
| Test ROC-AUC | 0.8597 |
| Test Brier Score (calibrated) | 0.1506 |
| Test Accuracy | 80.56% |
| Test Precision | 82.72% |
| Test Recall | 76.14% |
| Test F1 | 0.7929 |

Training protocol: chronological train/validation/test split (70/15/15), model selection by validation PR-AUC, out-of-fold calibration fitted only on training data, final metrics from the untouched test split.

## Decision Engine

Contesting a dispute costs a non-refundable arbitration fee when lost. The engine computes:

```text
E[EV] = P(win) × Amount − (1 − P(win)) × Arbitration Fee

Break-even probability:  τ* = Fee / (Amount + Fee)
```

Three outcomes:

| Verdict | Meaning |
|---|---|
| **CONTEST** | Positive expected value, all policy gates pass — automated defense recommended |
| **REVIEW** | Positive EV but one or more safety gates triggered — escalated to a human operator |
| **SURRENDER** | Negative expected value — accepting the loss avoids the non-refundable fee |

Policy gates can force REVIEW even when Expected Value is positive. Current gates: high-value disputes (≥ ₹25,000), win-probability confidence floor (70%), evidence-readiness minimum (60/100), urgent filing deadline (≤ 3 days), and address-mismatch combined with repeat dispute history. Thresholds are configuration constants in `config.py`, not universal real-world rules.

## Security

- Untrusted customer complaint text passes through a deterministic sanitizer (prompt-injection signatures, delimiter break-outs, control/ANSI/BiDi characters) before anything downstream can interpret it.
- Original complaint text is preserved **only** inside an explicitly labeled `UNTRUSTED / SANITIZED` audit field; display surfaces use the neutralized copy.
- Customer text cannot influence ML features, win probability, expected value, verdicts, or evidence provenance — enforced structurally and covered by regression tests.
- Audit ledger: canonical-JSON SHA-256 payload hashing plus chained entry hashing (`prev_hash ‖ payload_hash ‖ timestamp ‖ id ‖ type`).
- Optional HMAC-SHA256 entry authentication: set `SENTINEL_AUDIT_SECRET` to sign every new entry; verification fails closed on unsigned entries when the key is configured.
- Demo ledger (`data/demo_audit_ledger.jsonl`) is separated from the runtime ledger path (`data/audit_ledger.jsonl`); both are environment-local and never committed.

This is a demonstration of security-engineering patterns, not a production-bank secure system.

## Architecture

```mermaid
flowchart TD
    A[Manual Case Intake<br/>Streamlit form] --> B[Input Sanitizer<br/>prompt-injection firewall]
    B --> C[Feature Pipeline<br/>41 engineered features]
    C --> D[Random Forest + Calibration<br/>P win]
    D --> E[TreeSHAP Explainer]
    D --> F[Expected Value Engine]
    F --> G[Policy Gates<br/>amount/confidence/deadline/readiness/EV]
    G --> H{Verdict}
    H -->|CONTEST| I[Evidence Assembler]
    H -->|REVIEW| I
    H -->|SURRENDER| J[Close case - save fee]
    I --> K[Dossier Formatter<br/>Markdown + JSON]
    K --> L[Audit Ledger<br/>SHA-256 chain, optional HMAC-SHA256]

    subgraph Observed Evidence
        M1[Authentication / 3DS]
        M2[Fulfillment / Carrier POD]
        M3[Session Telemetry]
        M4[Customer History / CE3.0]
    end
    Observed Evidence --> I
```

## Project Structure

```
sentinel_risk/
├── config.py                  # thresholds, paths, seeds (all simulation constants)
├── requirements.txt
├── dashboard/
│   └── app.py                 # Streamlit operations console (triage, manual intake,
│                              #   benchmark, audit ledger, sanitizer views)
├── src/
│   ├── engine.py              # DecisionEngine: EV math + deterministic policy gating
│   ├── ml/
│   │   ├── features.py        # FeaturePipeline (41 features, leakage guards)
│   │   ├── train.py           # training, OOF calibration, evaluation, artifacts
│   │   └── explain.py         # DisputeExplainer (exact TreeSHAP)
│   ├── agent/
│   │   ├── schemas.py         # pydantic schemas, boolean parsing, evidence checklist
│   │   ├── assembler.py       # EvidenceAssembler (provenance, sanitization hook)
│   │   └── dossier.py         # rebuttal dossier formatter
│   └── security/
│       ├── audit.py           # AuditLedger (hash chain, optional HMAC)
│       └── sanitizer.py       # InputSanitizer (injection firewall)
├── data/
│   ├── generate_dataset.py    # seeded synthetic generator
│   └── disputes/train/val/test.csv
├── models/
│   ├── sentinel_model.joblib  # trained calibrated model (tracked, ~4.8 MB)
│   └── test_metrics.json
├── benchmark/
│   └── evaluate.py            # reproducible benchmark harness (+ results JSON)
└── tests/                     # authoritative test suite (50 tests)
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The trained model artifact (`models/sentinel_model.joblib`) is included in the repository, so the console works immediately after cloning. To regenerate data, model, or benchmarks from scratch:

```bash
python data/generate_dataset.py
python src/ml/train.py
python benchmark/evaluate.py
```

To enable HMAC-signed audit entries, set the `SENTINEL_AUDIT_SECRET` environment variable before launching the app.

## Testing

```bash
pytest -q
```

Current status: **50 passed**.

The suite covers unit mathematics (EV/break-even boundaries), feature-pipeline integrity and leakage prevention, model inference determinism, TreeSHAP invariants, decision-policy gating, sanitizer integration and injection containment, hash-chain tamper detection (modification, deletion, reordering, forged signatures), and end-to-end flows from raw dispute to signed ledger entry.

## Important Disclaimer

**SentinelRisk v1.0 is a synthetic simulation and technical demonstration.**

- It does not connect to any bank, card issuer, card network, carrier, or payment gateway.
- The datasets, telemetry, and provenance record IDs are synthetic.
- It must not be used to make real financial or dispute decisions.
- Metrics describe behavior on synthetic data only and do not predict real-world performance.

## Future Work

Planned directions (not yet implemented):

- AI-assisted extraction of structured fields from free-form dispute complaints
- Evidence document ingestion (invoices, receipts, carrier PDFs)
- Real enterprise data connectors
- Production database-backed audit storage with external anchoring
- Asymmetric digital signatures for audit entries
- Human-review workflow analytics
