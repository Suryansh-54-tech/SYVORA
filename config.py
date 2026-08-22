"""
SentinelRisk Configuration
==========================
All financial thresholds, model parameters, and policy constants.
Values are simulation parameters — not universal real-world constants.
"""

# ---------------------------------------------------------------------------
# Financial Simulation Parameters (Configurable)
# ---------------------------------------------------------------------------

# Non-refundable bank arbitration fee incurred when a contested dispute is lost.
# Real-world range: ₹300–₹1,500 / $15–$50 internationally.
# This is a SIMULATION DEFAULT — not a universal constant.
ARBITRATION_FEE_INR: float = 500.0

# ---------------------------------------------------------------------------
# Human-in-the-Loop (HITL) Policy Thresholds
# ---------------------------------------------------------------------------

# Disputes above this amount ALWAYS route to human review,
# regardless of ML confidence.
HITL_AMOUNT_THRESHOLD_INR: float = 25_000.0

# Minimum calibrated win probability required for automated defense.
# Below this → REVIEW_HITL.
HITL_CONFIDENCE_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Evidence Readiness Requirements
# ---------------------------------------------------------------------------

# Minimum Evidence Readiness Score (0–100) required for AUTO_DEFEND.
# Below this → REVIEW_HITL due to insufficient proof.
MIN_EVIDENCE_READINESS_SCORE: int = 60

# ---------------------------------------------------------------------------
# Serial / Friendly-Fraud Forensics Thresholds
# ---------------------------------------------------------------------------

# A customer with more than this many past disputes in 6 months is flagged
# as a potential serial abuser (friendly fraud suspect).
SERIAL_DISPUTE_FLAG_THRESHOLD: int = 2

# ---------------------------------------------------------------------------
# Dataset Generation Parameters
# ---------------------------------------------------------------------------

# Total number of synthetic dispute records to generate.
DATASET_SIZE: int = 1200

# Approximate class balance: fraction of disputes the merchant wins (y=1).
# Real-world merchant win rates vary from 20% to 45%.
WIN_RATE_APPROX: float = 0.35

# Random seed for full reproducibility.
RANDOM_SEED: int = 42

# Temporal split ratios (train / validation+calibration / held-out test).
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

import os
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DATASET_PATH = os.path.join(DATA_DIR, "disputes.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
AUDIT_LEDGER_PATH = os.path.join(DATA_DIR, "audit_ledger.jsonl")
DEMO_LEDGER_PATH = os.path.join(DATA_DIR, "demo_audit_ledger.jsonl")

# ---------------------------------------------------------------------------
# Security & Cryptographic Audit Configuration
# ---------------------------------------------------------------------------

# Application secret key for HMAC-SHA256 audit ledger entry signing.
# Defaults to None (UNSIGNED_DEMO mode) unless provided via environment variable.
AUDIT_SECRET_KEY: Optional[str] = os.getenv("SENTINEL_AUDIT_SECRET", None)

