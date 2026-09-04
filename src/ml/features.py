"""
NYAYANTRA — Feature Engineering Pipeline
=====================================
Deterministic feature transformation for post-payment dispute triage.
Enforces strict point-in-time semantics and guards against target leakage.

Features engineered:
- Log-transformed monetary scale & high-value indicators
- 3D Secure authentication & courier proof indicators
- Visa CE3.0 qualifying indicators (2+ prior undisputed orders + IP match)
- Serial disputer / Friendly-fraud behavioral flags
- Evidence Readiness composite score (0-100)
- Categorical one-hot encodings for reason codes, issuing banks, networks
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agent.schemas import parse_bool


def normalize_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    """
    Authoritative vectorized boolean normalizer for feature extraction.
    Applies parse_bool across every element to guarantee 'False' -> False and 'True' -> True.
    """
    return series.map(lambda x: parse_bool(x, default=default)).astype(bool)


# ---------------------------------------------------------------------------
# Strict Boundary & Leakage Guardrails
# ---------------------------------------------------------------------------

FORBIDDEN_COLUMNS = {
    "dispute_outcome",           # Target label
    "outcome",
    "is_won",
    "bank_resolution",
    "arbitration_fee_incurred",  # Post-event outcomes
    "resolution_date",
    "actual_loss_inr",
}

METADATA_COLUMNS = {
    "dispute_id",
    "transaction_id",
    "dispute_date",
}

# Fixed categorical vocabularies to guarantee deterministic one-hot shapes
REASON_CODE_VOCAB = [
    "VISA_10_4_FRAUD",
    "VISA_13_1_NOT_RECEIVED",
    "VISA_13_3_DEFECTIVE",
    "MC_4837_FRAUD",
    "MC_4853_GOODS_SERVICES",
]

ISSUING_BANK_VOCAB = [
    "HDFC",
    "ICICI",
    "SBI",
    "AXIS",
    "KOTAK",
    "CITI_INTL",
    "AMEX_INTL",
]

CARD_NETWORK_VOCAB = ["VISA", "MASTERCARD"]

MERCHANT_CATEGORY_VOCAB = [
    "ECOMM_RETAIL",
    "ELECTRONICS",
    "DIGITAL_SAAS",
    "FASHION_APPAREL",
    "TRAVEL_HOTEL",
    "FOOD_DELIVERY",
]

COURIER_STATUS_VOCAB = [
    "DELIVERED",
    "IN_TRANSIT",
    "RETURNED",
    "NOT_APPLICABLE",
    "UNKNOWN",
]


class FeaturePipeline:
    """
    Production-grade, leak-free feature extraction pipeline.
    Deterministic across train, validation, test splits, and live inference.
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self._is_fitted: bool = False

    def validate_input(self, df: pd.DataFrame, require_target: bool = False) -> None:
        """
        Guarantees no forbidden post-event columns enter the pipeline.
        """
        columns = set(df.columns)
        
        # If target is explicitly required, verify presence
        if require_target and "dispute_outcome" not in columns:
            raise ValueError("Target column 'dispute_outcome' is missing from training input.")

        # Check for any forbidden post-event columns (excluding target label itself from raw input check)
        post_event_leaks = columns.intersection(FORBIDDEN_COLUMNS - {"dispute_outcome"})
        if post_event_leaks:
            raise ValueError(f"Post-event data leakage detected! Forbidden columns present: {post_event_leaks}")

    def compute_evidence_readiness(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates a deterministic Evidence Readiness Score (0–100).
        Evaluates completeness of digital evidence available at dispute arrival.
        """
        score = pd.Series(0, index=df.index, dtype=int)
        
        signed_pod = normalize_bool_series(df["signed_pod"], default=False)
        ip_geo = normalize_bool_series(df["ip_geo_match"], default=False)
        dev_match = normalize_bool_series(df["device_fingerprint_match"], default=False)
        bill_ship = normalize_bool_series(df["billing_shipping_match"], default=True)

        # Delivery evidence (up to 40 pts)
        score += (df["courier_status"] == "DELIVERED").astype(int) * 20
        score += signed_pod.astype(int) * 20
        
        # Authentication evidence (up to 30 pts)
        score += (df["three_ds_status"] == "Y_AUTHENTICATED").astype(int) * 30
        score += (df["three_ds_status"] == "A_ATTEMPTED").astype(int) * 15
        
        # Telemetry & Forensics (up to 30 pts)
        score += ip_geo.astype(int) * 10
        score += dev_match.astype(int) * 10
        score += bill_ship.astype(int) * 10

        return score.clip(0, 100)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw dispute records into engineered feature matrix X.
        Guarantees point-in-time evaluation.
        """
        self.validate_input(df, require_target=False)
        
        features = pd.DataFrame(index=df.index)

        # -------------------------------------------------------------------
        # 1. Financial & Numerical Features
        # -------------------------------------------------------------------
        features["txn_amount_inr"] = df["txn_amount_inr"].astype(float)
        features["log_txn_amount"] = np.log1p(df["txn_amount_inr"].astype(float))
        features["is_high_value_dispute"] = (
            df["txn_amount_inr"].astype(float) >= config.HITL_AMOUNT_THRESHOLD_INR
        ).astype(int)

        # -------------------------------------------------------------------
        # 2. Temporal Features (Relative to dispute receipt)
        # -------------------------------------------------------------------
        features["txn_age_days"] = df["txn_age_days"].astype(int)
        features["days_to_deadline"] = df["days_to_deadline"].astype(int)
        features["is_urgent_deadline"] = (df["days_to_deadline"].astype(int) <= 3).astype(int)

        # -------------------------------------------------------------------
        # 3. Direct Digital Evidence Indicators (Safe Boolean Normalization)
        # -------------------------------------------------------------------
        signed_pod = normalize_bool_series(df["signed_pod"], default=False)
        ip_geo = normalize_bool_series(df["ip_geo_match"], default=False)
        dev_match = normalize_bool_series(df["device_fingerprint_match"], default=False)
        bill_ship = normalize_bool_series(df["billing_shipping_match"], default=True)

        features["has_signed_pod"] = signed_pod.astype(int)
        features["has_ip_geo_match"] = ip_geo.astype(int)
        features["has_device_fingerprint_match"] = dev_match.astype(int)
        features["has_billing_shipping_match"] = bill_ship.astype(int)
        features["is_3ds_authenticated"] = (
            df["three_ds_status"] == "Y_AUTHENTICATED"
        ).astype(int)

        # -------------------------------------------------------------------
        # 4. Domain & Behavioral Forensics (Cyber Security / Friendly-Fraud)
        # -------------------------------------------------------------------
        # Visa Compelling Evidence 3.0 (CE3.0) Simulation Indicator:
        # 2+ prior undisputed orders + IP/Device match shifts liability to issuer
        features["prior_undisputed_txns"] = df["prior_undisputed_txns"].astype(int)
        features["is_visa_ce3_eligible"] = (
            (df["prior_undisputed_txns"].astype(int) >= 2)
            & ip_geo
        ).astype(int)

        # Serial abuser / friendly fraud flag
        features["customer_past_dispute_count"] = df["customer_past_dispute_count"].astype(int)
        features["is_serial_disputer"] = (
            df["customer_past_dispute_count"].astype(int) >= config.SERIAL_DISPUTE_FLAG_THRESHOLD
        ).astype(int)

        # Evidence Readiness Score (0 to 100)
        features["evidence_readiness_score"] = self.compute_evidence_readiness(df)

        # -------------------------------------------------------------------
        # 5. One-Hot Categorical Encodings (Deterministic Vocabularies)
        # -------------------------------------------------------------------
        for code in REASON_CODE_VOCAB:
            features[f"reason_{code}"] = (df["reason_code"] == code).astype(int)

        for bank in ISSUING_BANK_VOCAB:
            features[f"bank_{bank}"] = (df["issuing_bank"] == bank).astype(int)

        for network in CARD_NETWORK_VOCAB:
            features[f"network_{network}"] = (df["card_network"] == network).astype(int)

        for category in MERCHANT_CATEGORY_VOCAB:
            features[f"cat_{category}"] = (df["merchant_category"] == category).astype(int)

        for status in COURIER_STATUS_VOCAB:
            features[f"courier_{status}"] = (df["courier_status"] == status).astype(int)

        # Save feature schema during first pass
        if not self._is_fitted:
            self.feature_names = list(features.columns)
            self._is_fitted = True
        else:
            # Ensure exact column alignment
            features = features.reindex(columns=self.feature_names, fill_value=0)

        return features

    def process_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Extracts feature matrix X and target y (if present).
        Guarantees y is never included in X.
        """
        y = None
        if "dispute_outcome" in df.columns:
            y = df["dispute_outcome"].astype(int).copy()
        
        # Transform features
        X = self.transform(df)
        return X, y


# ---------------------------------------------------------------------------
# CLI / Verification Runner
# ---------------------------------------------------------------------------

def run_pipeline_check():
    print("=" * 65)
    print("  NYAYANTRA -- Feature Pipeline Verification")
    print("=" * 65)

    pipeline = FeaturePipeline()

    # Load splits generated in Phase 1
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    print(f"\n[1/4] Processing Splits...")
    X_train, y_train = pipeline.process_split(train_df)
    X_val, y_val = pipeline.process_split(val_df)
    X_test, y_test = pipeline.process_split(test_df)

    print(f"      Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"      Val shape:   X={X_val.shape}, y={y_val.shape}")
    print(f"      Test shape:  X={X_test.shape}, y={y_test.shape}")

    # Verify identical columns
    print(f"\n[2/4] Verifying Feature Schema Consistency...")
    train_cols = list(X_train.columns)
    val_cols = list(X_val.columns)
    test_cols = list(X_test.columns)

    assert train_cols == val_cols == test_cols, "Feature columns mismatch across splits!"
    print(f"      [OK] Identical feature schema ({len(train_cols)} features) across all splits.")

    # Verify no target / metadata leakage
    print(f"\n[3/4] Verifying Target & Metadata Leakage Prevention...")
    leaked_forbidden = set(train_cols).intersection(FORBIDDEN_COLUMNS)
    leaked_metadata = set(train_cols).intersection(METADATA_COLUMNS)

    assert len(leaked_forbidden) == 0, f"Target leakage detected in features: {leaked_forbidden}"
    assert len(leaked_metadata) == 0, f"Raw metadata/IDs detected in features: {leaked_metadata}"
    print(f"      [OK] Zero target or forbidden post-event columns in feature matrix.")
    print(f"      [OK] Zero raw ID or timestamp columns in feature matrix.")

    # Check for NaN / Inf values
    print(f"\n[4/4] Checking Numerical Integrity (NaN / Inf)...")
    for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        nans = X.isna().sum().sum()
        infs = np.isinf(X.values).sum()
        assert nans == 0, f"{name} contains {nans} NaN values!"
        assert infs == 0, f"{name} contains {infs} Inf values!"
        print(f"      [OK] {name}: 0 NaNs, 0 Infs across all {X.shape[0] * X.shape[1]:,} cells.")

    print(f"\n{'=' * 65}")
    print(f"  Feature Pipeline Summary: {len(train_cols)} engineered features ready.")
    print(f"  Sample Features: {train_cols[:8]} ...")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run_pipeline_check()
