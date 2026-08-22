"""
SentinelRisk — Synthetic Dispute Dataset Generator
===================================================
Generates realistic post-payment chargeback/dispute records with:
- Real Visa/Mastercard reason codes
- Device & IP telemetry for friendly-fraud forensics
- Noisy latent utility model to prevent trivial ML solutions
- Strict temporal train/validation/test splits

All data is synthetic. No real customer or payment data is used.

Usage:
    python data/generate_dataset.py
"""

import os
import sys
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Domain Constants (Simulation)
# ---------------------------------------------------------------------------

REASON_CODES = [
    # (code_string, network, description, base_merchant_win_bias)
    ("VISA_10_4_FRAUD", "VISA", "Alleged Fraud - Card Absent", -0.15),
    ("VISA_13_1_NOT_RECEIVED", "VISA", "Merchandise Not Received", 0.10),
    ("VISA_13_3_DEFECTIVE", "VISA", "Not as Described / Defective", -0.05),
    ("MC_4837_FRAUD", "MASTERCARD", "No Cardholder Authorization", -0.20),
    ("MC_4853_GOODS_SERVICES", "MASTERCARD", "Goods/Services Dispute", 0.0),
]

ISSUING_BANKS = [
    ("HDFC", 0.05),     # Slightly more merchant-friendly
    ("ICICI", 0.0),
    ("SBI", -0.05),
    ("AXIS", 0.02),
    ("KOTAK", 0.0),
    ("CITI_INTL", -0.10),   # International issuer - harder to win
    ("AMEX_INTL", -0.12),   # International issuer - harder to win
]

MERCHANT_CATEGORIES = [
    "ECOMM_RETAIL",
    "ELECTRONICS",
    "DIGITAL_SAAS",
    "FASHION_APPAREL",
    "TRAVEL_HOTEL",
    "FOOD_DELIVERY",
]

COURIER_CARRIERS = ["DELHIVERY", "BLUEDART", "INDIAPOST", "DTDC", "FEDEX_INTL"]

THREE_DS_STATUSES = [
    ("Y_AUTHENTICATED", 0.25),    # 3DS fully verified - strong evidence
    ("A_ATTEMPTED", 0.10),        # Attempted but not completed
    ("N_NOT_ENROLLED", -0.05),    # Card not enrolled
    ("U_UNAVAILABLE", -0.10),     # System unavailable
]


def generate_disputes(n: int, seed: int) -> pd.DataFrame:
    """
    Generate n synthetic dispute records using a noisy latent utility model.

    The outcome label y ∈ {0, 1} is NOT a simple deterministic rule.
    It is derived from a latent utility with Gaussian noise, bank discretion
    bias, and random edge-case flips to prevent artificial correlation.
    """
    rng = np.random.default_rng(seed)

    records = []
    base_timestamp = datetime(2025, 1, 15)

    for i in range(n):
        # --- Temporal ordering (for valid temporal splits) ---
        days_offset = int((i / n) * 365)  # Spread across ~1 year
        dispute_date = base_timestamp + timedelta(
            days=days_offset, hours=int(rng.integers(0, 24))
        )

        # --- Transaction amount (log-normal for realistic heavy tail) ---
        txn_amount = float(np.clip(rng.lognormal(mean=7.5, sigma=1.2), 150, 200_000))
        txn_amount = round(txn_amount, 2)

        # --- Reason code ---
        rc_idx = rng.integers(0, len(REASON_CODES))
        reason_code, card_network, reason_desc, rc_bias = REASON_CODES[rc_idx]

        # --- Issuing bank ---
        bank_idx = rng.integers(0, len(ISSUING_BANKS))
        issuing_bank, bank_bias = ISSUING_BANKS[bank_idx]

        # --- Merchant category ---
        merchant_category = rng.choice(MERCHANT_CATEGORIES)

        # --- 3D Secure status ---
        tds_idx = rng.integers(0, len(THREE_DS_STATUSES))
        three_ds_status, tds_bias = THREE_DS_STATUSES[tds_idx]

        # --- Courier / Delivery evidence ---
        # Physical goods have courier; digital/SaaS do not
        is_physical = merchant_category not in ("DIGITAL_SAAS",)

        if is_physical:
            courier_status = rng.choice(
                ["DELIVERED", "DELIVERED", "DELIVERED", "IN_TRANSIT", "RETURNED", "UNKNOWN"],
            )
            carrier = rng.choice(COURIER_CARRIERS)
            # Signed POD is only possible if delivered
            signed_pod = bool(
                courier_status == "DELIVERED" and rng.random() < 0.70
            )
        else:
            courier_status = "NOT_APPLICABLE"
            carrier = "NONE"
            signed_pod = False

        # --- Device & IP forensics ---
        ip_geo_match = bool(rng.random() < 0.75)  # 75% of the time IP matches
        device_fingerprint_match = bool(rng.random() < 0.70)

        # --- Customer behavioral history ---
        # Most customers have 0 prior disputes; a small % are serial abusers
        customer_past_dispute_count = int(
            rng.choice([0, 0, 0, 0, 0, 1, 1, 2, 3, 5], size=1)[0]
        )
        prior_undisputed_txns = int(rng.choice([0, 0, 1, 1, 2, 3, 4, 6], size=1)[0])

        # --- Days to deadline ---
        days_to_deadline = int(rng.integers(2, 15))

        # --- Transaction age at dispute (days since purchase) ---
        txn_age_days = int(rng.integers(3, 90))

        # --- Address match ---
        billing_shipping_match = bool(rng.random() < 0.82)

        # ===================================================================
        # NOISY LATENT UTILITY MODEL FOR OUTCOME GENERATION
        # ===================================================================
        # The outcome is NOT a trivial rule. It combines weighted signals
        # with substantial Gaussian noise and bank discretion to ensure
        # the ML model cannot achieve perfect scores.
        #
        # Positive utility → merchant wins (y=1)
        # Negative utility → merchant loses (y=0)
        # ===================================================================

        utility = -0.60  # Base bias toward merchant loss (realistic)

        # Evidence strength signals
        if courier_status == "DELIVERED":
            utility += 0.45
        elif courier_status == "IN_TRANSIT":
            utility += 0.05
        elif courier_status == "RETURNED":
            utility -= 0.20

        if signed_pod:
            utility += 0.35

        if ip_geo_match:
            utility += 0.15

        if device_fingerprint_match:
            utility += 0.10

        if billing_shipping_match:
            utility += 0.10

        # 3DS & authentication
        utility += tds_bias

        # Reason code & bank biases
        utility += rc_bias
        utility += bank_bias

        # Prior undisputed transactions (Visa CE3.0 signal)
        if prior_undisputed_txns >= 2 and ip_geo_match:
            utility += 0.25  # CE3.0 eligibility boost

        # Serial abuser penalty (banks may still side with cardholder)
        if customer_past_dispute_count >= 3:
            utility += 0.15  # Actually helps merchant: bank sees abuser pattern
        elif customer_past_dispute_count == 0:
            utility -= 0.05  # First-time disputer — harder to prove friendly fraud

        # Digital goods are harder to prove delivery
        if not is_physical:
            utility -= 0.15

        # ===================================================================
        # NOISE INJECTION — prevents trivially learnable labels
        # ===================================================================

        # Gaussian noise (bank adjudicator human discretion)
        noise = float(rng.normal(0, 0.35))
        utility += noise

        # 8% random flip: models real-world unpredictability where banks
        # sometimes rule against strong evidence or for weak cases
        random_flip = rng.random() < 0.08

        # Final binary outcome
        raw_win = utility > 0
        outcome = int(not raw_win if random_flip else raw_win)

        # --- Unique dispute ID ---
        dispute_id = f"dsp_{i+1:05d}"

        # --- Simulated Razorpay transaction/payment IDs ---
        txn_id = f"pay_{hashlib.md5(f'{seed}_{i}'.encode()).hexdigest()[:12]}"

        records.append({
            "dispute_id": dispute_id,
            "transaction_id": txn_id,
            "dispute_date": dispute_date.strftime("%Y-%m-%d %H:%M:%S"),
            "txn_amount_inr": txn_amount,
            "reason_code": reason_code,
            "card_network": card_network,
            "issuing_bank": issuing_bank,
            "merchant_category": merchant_category,
            "three_ds_status": three_ds_status,
            "courier_status": courier_status,
            "carrier": carrier,
            "signed_pod": signed_pod,
            "ip_geo_match": ip_geo_match,
            "device_fingerprint_match": device_fingerprint_match,
            "billing_shipping_match": billing_shipping_match,
            "customer_past_dispute_count": customer_past_dispute_count,
            "prior_undisputed_txns": prior_undisputed_txns,
            "txn_age_days": txn_age_days,
            "days_to_deadline": days_to_deadline,
            "dispute_outcome": outcome,  # 1 = merchant won, 0 = merchant lost
        })

    df = pd.DataFrame(records)
    return df


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by chronological order (temporal split).
    This prevents data leakage from future disputes informing past predictions.
    """
    df = df.sort_values("dispute_date").reset_index(drop=True)
    n = len(df)

    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run basic data quality checks and return a validation report."""
    report = {}

    # 1. Shape
    report["total_records"] = len(df)
    report["total_columns"] = len(df.columns)

    # 2. Missing values
    missing = df.isnull().sum()
    report["columns_with_missing"] = {
        col: int(count) for col, count in missing.items() if count > 0
    }

    # 3. Duplicate IDs
    dup_ids = df["dispute_id"].duplicated().sum()
    report["duplicate_dispute_ids"] = int(dup_ids)

    dup_txn_ids = df["transaction_id"].duplicated().sum()
    report["duplicate_transaction_ids"] = int(dup_txn_ids)

    # 4. Class distribution
    class_counts = df["dispute_outcome"].value_counts().to_dict()
    report["class_distribution"] = {
        f"outcome_{k}": int(v) for k, v in class_counts.items()
    }
    win_rate = df["dispute_outcome"].mean()
    report["merchant_win_rate"] = round(win_rate, 4)

    # 5. Amount statistics
    report["amount_stats"] = {
        "min": round(df["txn_amount_inr"].min(), 2),
        "max": round(df["txn_amount_inr"].max(), 2),
        "mean": round(df["txn_amount_inr"].mean(), 2),
        "median": round(df["txn_amount_inr"].median(), 2),
    }

    # 6. Reason code distribution
    report["reason_code_distribution"] = (
        df["reason_code"].value_counts().to_dict()
    )

    # 7. Temporal range
    report["date_range"] = {
        "earliest": df["dispute_date"].min(),
        "latest": df["dispute_date"].max(),
    }

    # 8. Target leakage check: no column should have >0.95 correlation with outcome
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "dispute_outcome" in numeric_cols:
        correlations = df[numeric_cols].corr()["dispute_outcome"].drop("dispute_outcome")
        high_corr = correlations[correlations.abs() > 0.60]
        report["high_correlation_with_target"] = {
            col: round(float(val), 4) for col, val in high_corr.items()
        }

    return report


def main():
    print("=" * 65)
    print("  SentinelRisk -- Synthetic Dispute Dataset Generator")
    print("=" * 65)

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Generate
    print(f"\n[1/4] Generating {config.DATASET_SIZE} synthetic dispute records...")
    df = generate_disputes(config.DATASET_SIZE, config.RANDOM_SEED)

    # Save full dataset
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"      Saved full dataset to: {config.DATASET_PATH}")

    # Temporal split
    print(f"\n[2/4] Splitting temporally ({config.TRAIN_RATIO:.0%} / "
          f"{config.VAL_RATIO:.0%} / {config.TEST_RATIO:.0%})...")
    train, val, test = split_temporal(df)
    train.to_csv(config.TRAIN_PATH, index=False)
    val.to_csv(config.VAL_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    print(f"      Train: {len(train)} records  ->  {config.TRAIN_PATH}")
    print(f"      Val:   {len(val)} records  ->  {config.VAL_PATH}")
    print(f"      Test:  {len(test)} records  ->  {config.TEST_PATH}")

    # Validate
    print("\n[3/4] Running data quality validation...")
    report = validate_dataset(df)

    print(f"\n{'-' * 50}")
    print(f"  VALIDATION REPORT")
    print(f"{'-' * 50}")
    print(f"  Total Records:         {report['total_records']}")
    print(f"  Total Columns:         {report['total_columns']}")
    print(f"  Duplicate Dispute IDs: {report['duplicate_dispute_ids']}")
    print(f"  Duplicate Txn IDs:     {report['duplicate_transaction_ids']}")
    print(f"  Missing Values:        {report['columns_with_missing'] or 'None'}")
    print(f"  Merchant Win Rate:     {report['merchant_win_rate']:.2%}")
    print(f"  Class Distribution:    {report['class_distribution']}")
    print(f"  Amount Range:          INR {report['amount_stats']['min']:,.2f} -- "
          f"INR {report['amount_stats']['max']:,.2f}")
    print(f"  Amount Mean:           INR {report['amount_stats']['mean']:,.2f}")
    print(f"  Date Range:            {report['date_range']['earliest']} -> "
          f"{report['date_range']['latest']}")
    print(f"  Reason Codes:          {report['reason_code_distribution']}")

    if report.get("high_correlation_with_target"):
        print(f"\n  [!] HIGH CORRELATION WITH TARGET (|r| > 0.60):")
        for col, val in report["high_correlation_with_target"].items():
            print(f"    - {col}: r = {val}")
    else:
        print(f"\n  [OK] No feature has |correlation| > 0.60 with target")
        print(f"    (Confirms noise injection is preventing trivial learnability)")

    # Split validation
    print(f"\n{'-' * 50}")
    print(f"  SPLIT VALIDATION")
    print(f"{'-' * 50}")
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        wr = split_df["dispute_outcome"].mean()
        print(f"  {name:5s}: {len(split_df):4d} records | "
              f"Win rate: {wr:.2%} | "
              f"Date range: {split_df['dispute_date'].min()[:10]} -> "
              f"{split_df['dispute_date'].max()[:10]}")

    print(f"\n[4/4] [OK] Phase 1 dataset generation complete.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
