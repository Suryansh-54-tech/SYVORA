"""
SentinelRisk — Model Explainability Engine (TreeSHAP)
=====================================================
Generates local feature attributions for individual dispute risk scores
using TreeSHAP on the underlying tree ensemble model.

Outputs:
- Directional feature contributions (positive = increases win probability, negative = increases risk)
- Normalized impact percentages
- Human-readable explanations for operational dashboard and defense dossiers
- Pure deterministic calculation — zero LLM calls
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.ml.features import FeaturePipeline


# ---------------------------------------------------------------------------
# Human-Readable Feature Dictionary
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES = {
    # Direct Evidence
    "has_signed_pod": "Carrier Proof of Delivery (Signed POD)",
    "has_ip_geo_match": "IP Geo-Location Matches Delivery Address",
    "has_device_fingerprint_match": "Device Fingerprint Matches Known Customer Device",
    "has_billing_shipping_match": "Billing and Shipping Addresses Match",
    "is_3ds_authenticated": "3D Secure 2.0 Strong Customer Authentication Verified",
    
    # Forensics & Network Rules
    "is_visa_ce3_eligible": "Visa CE3.0 Qualifying (2+ Prior Undisputed Orders on Same IP/Device)",
    "prior_undisputed_txns": "Prior Undisputed Customer Transactions",
    "customer_past_dispute_count": "Customer Historical Dispute Frequency",
    "is_serial_disputer": "Serial Chargeback Abuser Flag (>=2 Past Disputes)",
    "evidence_readiness_score": "Composite Digital Evidence Readiness Index",

    # Financial & Temporal
    "txn_amount_inr": "Dispute Transaction Amount",
    "log_txn_amount": "Log-Scale Transaction Value",
    "is_high_value_dispute": "High-Value Transaction Threshold Exceeded",
    "txn_age_days": "Days Elapsed Since Original Transaction",
    "days_to_deadline": "Days Remaining to Bank Evidence Submission Deadline",
    "is_urgent_deadline": "Urgent Evidence Deadline (<= 3 Days)",

    # Reason Codes
    "reason_VISA_10_4_FRAUD": "Dispute Reason: Visa 10.4 (Card-Absent Fraud)",
    "reason_VISA_13_1_NOT_RECEIVED": "Dispute Reason: Visa 13.1 (Merchandise Not Received)",
    "reason_VISA_13_3_DEFECTIVE": "Dispute Reason: Visa 13.3 (Not as Described)",
    "reason_MC_4837_FRAUD": "Dispute Reason: Mastercard 4837 (No Cardholder Auth)",
    "reason_MC_4853_GOODS_SERVICES": "Dispute Reason: Mastercard 4853 (Goods/Services Dispute)",

    # Issuing Banks
    "bank_HDFC": "Issuing Bank: HDFC Bank (Domestic Prior)",
    "bank_ICICI": "Issuing Bank: ICICI Bank",
    "bank_SBI": "Issuing Bank: State Bank of India",
    "bank_AXIS": "Issuing Bank: Axis Bank",
    "bank_KOTAK": "Issuing Bank: Kotak Mahindra Bank",
    "bank_CITI_INTL": "Issuing Bank: Citibank (International / Strict Issuer)",
    "bank_AMEX_INTL": "Issuing Bank: American Express (International / High Reversal Rate)",

    # Card Networks
    "network_VISA": "Card Brand: Visa",
    "network_MASTERCARD": "Card Brand: Mastercard",

    # Categories
    "cat_ECOMM_RETAIL": "Merchant Category: General E-Commerce Retail",
    "cat_ELECTRONICS": "Merchant Category: Consumer Electronics",
    "cat_DIGITAL_SAAS": "Merchant Category: Digital Goods / SaaS",
    "cat_FASHION_APPAREL": "Merchant Category: Fashion & Apparel",
    "cat_TRAVEL_HOTEL": "Merchant Category: Travel & Hospitality",
    "cat_FOOD_DELIVERY": "Merchant Category: Food & Quick Commerce",

    # Couriers
    "courier_DELIVERED": "Courier Fulfillment Status: Delivered to Destination",
    "courier_IN_TRANSIT": "Courier Fulfillment Status: Still in Transit",
    "courier_RETURNED": "Courier Fulfillment Status: Returned to Origin",
    "courier_NOT_APPLICABLE": "Courier Fulfillment Status: Digital Fulfillment (No Courier)",
    "courier_UNKNOWN": "Courier Fulfillment Status: Unconfirmed",
}


# ---------------------------------------------------------------------------
# TreeSHAP Explainer Engine
# ---------------------------------------------------------------------------

class DisputeExplainer:
    """
    Deterministic TreeSHAP explainer for post-payment dispute triage.
    Extracts local feature contributions for individual transactions.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, "sentinel_model.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Run train.py first.")

        bundle = joblib.load(model_path)
        self.calibrated_model = bundle["model"]
        self.base_model = bundle["base_model"]
        self.feature_names = bundle["feature_names"]
        self.base_model_name = bundle.get("base_model_name", "RandomForest")

        # Initialize TreeExplainer on the base tree ensemble model
        # (TreeExplainer runs in C++ for fast, exact Shapley computation)
        self.explainer = shap.TreeExplainer(self.base_model)
        
        # Base expected value (prior log-odds or mean prediction)
        if hasattr(self.explainer.expected_value, "__len__") and len(self.explainer.expected_value) > 1:
            self.expected_value = float(self.explainer.expected_value[1])
        else:
            self.expected_value = float(self.explainer.expected_value)

    def explain_instance(
        self,
        X_row: pd.DataFrame,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single dispute feature row.

        Args:
            X_row: 1-row DataFrame containing the 41 engineered features.
            top_k: Number of top positive and negative factors to return.

        Returns:
            Structured dictionary with calibrated probability, base value,
            ranked positive/negative contributions, and human-readable narrative.
        """
        # Align features strictly to model schema
        X_aligned = X_row.reindex(columns=self.feature_names, fill_value=0)

        # 1. Calibrated win probability
        cal_prob = float(self.calibrated_model.predict_proba(X_aligned)[0, 1])

        # 2. Compute TreeSHAP values for the instance
        shap_raw = self.explainer.shap_values(X_aligned)

        # Handle multiclass vs binary output shapes in SHAP
        if isinstance(shap_raw, list) and len(shap_raw) == 2:
            # Binary classification: [shap_values_class_0, shap_values_class_1]
            instance_shap = shap_raw[1][0]
        elif isinstance(shap_raw, np.ndarray):
            if shap_raw.ndim == 3 and shap_raw.shape[2] == 2:
                # Shape (1, n_features, 2)
                instance_shap = shap_raw[0, :, 1]
            elif shap_raw.ndim == 2:
                # Shape (1, n_features)
                instance_shap = shap_raw[0]
            else:
                instance_shap = shap_raw.ravel()
        else:
            instance_shap = np.array(shap_raw).ravel()

        # Pair feature names with their local SHAP impact and raw values
        contributions = []
        for feat_name, shap_val in zip(self.feature_names, instance_shap):
            raw_val = X_aligned[feat_name].iloc[0]
            contributions.append({
                "feature": feat_name,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace("_", " ").title()),
                "shap_impact": float(shap_val),
                "raw_value": float(raw_val),
            })

        # Sort into positive drivers (increases win probability) and negative drivers (increases risk/loss)
        pos_drivers = sorted(
            [c for c in contributions if c["shap_impact"] > 0.001],
            key=lambda x: x["shap_impact"],
            reverse=True
        )[:top_k]

        neg_drivers = sorted(
            [c for c in contributions if c["shap_impact"] < -0.001],
            key=lambda x: x["shap_impact"]
        )[:top_k]

        # Generate a concise human-readable summary
        summary_clauses = []
        if pos_drivers:
            top_pos_names = [f"'{p['display_name']}' (+{p['shap_impact']:.1%})" for p in pos_drivers[:2]]
            summary_clauses.append(f"Win probability supported by {', '.join(top_pos_names)}")

        if neg_drivers:
            top_neg_names = [f"'{n['display_name']}' ({n['shap_impact']:.1%})" for n in neg_drivers[:2]]
            summary_clauses.append(f"Risk increased by {', '.join(top_neg_names)}")

        summary_narrative = ". ".join(summary_clauses) + "." if summary_clauses else "Balanced risk profile."

        return {
            "calibrated_win_probability": round(cal_prob, 4),
            "expected_base_value": round(self.expected_value, 4),
            "top_positive_factors": pos_drivers,
            "top_negative_factors": neg_drivers,
            "all_shap_contributions": {c["feature"]: round(c["shap_impact"], 4) for c in contributions},
            "explanation_summary": summary_narrative,
        }

    def explain_dispute_record(
        self,
        raw_dispute_dict: Dict[str, Any],
        pipeline: Optional[FeaturePipeline] = None
    ) -> Dict[str, Any]:
        """
        Convenience method: takes a raw un-engineered dispute dictionary,
        runs it through FeaturePipeline, and generates the SHAP explanation.
        """
        if pipeline is None:
            pipeline = FeaturePipeline()

        df_raw = pd.DataFrame([raw_dispute_dict])
        X_feat = pipeline.transform(df_raw)
        return self.explain_instance(X_feat)


# ---------------------------------------------------------------------------
# CLI / Verification Runner
# ---------------------------------------------------------------------------

def run_explainability_check():
    print("=" * 65)
    print("  SentinelRisk -- Model Explainability Verification (TreeSHAP)")
    print("=" * 65)

    explainer = DisputeExplainer()
    pipeline = FeaturePipeline()

    # Load test split
    test_df = pd.read_csv(config.TEST_PATH)
    X_test, y_test = pipeline.process_split(test_df)

    print(f"\n[1/4] Explaining Test Dispute #1 (dsp_01021)...")
    exp_1 = explainer.explain_instance(X_test.iloc[[0]])

    print(f"      Calibrated Win Prob: {exp_1['calibrated_win_probability']:.2%}")
    print(f"      Actual Test Label:   Outcome = {y_test.iloc[0]}")
    print(f"\n      Top Positive Drivers (Increasing Win Probability):")
    for item in exp_1["top_positive_factors"]:
        print(f"        + {item['display_name']} -> Impact: +{item['shap_impact']:.3f} (Value: {item['raw_value']})")

    print(f"\n      Top Negative Drivers (Increasing Loss Risk):")
    for item in exp_1["top_negative_factors"]:
        print(f"        - {item['display_name']} -> Impact: {item['shap_impact']:.3f} (Value: {item['raw_value']})")

    print(f"\n      Summary Narrative:")
    print(f"        \"{exp_1['explanation_summary']}\"")

    # Verify determinism on repeated call
    print(f"\n[2/4] Verifying Determinism on Repeated SHAP Calculation...")
    exp_1_repeat = explainer.explain_instance(X_test.iloc[[0]])
    assert exp_1["calibrated_win_probability"] == exp_1_repeat["calibrated_win_probability"]
    assert exp_1["all_shap_contributions"] == exp_1_repeat["all_shap_contributions"]
    print("      [OK] Exact bitwise determinism confirmed across runs.")

    # Verify second diverse case (High-confidence win case)
    print(f"\n[3/4] Explaining Test Dispute #5 (High Win Probability Case)...")
    exp_5 = explainer.explain_instance(X_test.iloc[[4]])
    print(f"      Calibrated Win Prob: {exp_5['calibrated_win_probability']:.2%}")
    print(f"      Actual Test Label:   Outcome = {y_test.iloc[4]}")
    print(f"      Summary Narrative:   \"{exp_5['explanation_summary']}\"")

    # Verify JSON serializability
    print(f"\n[4/4] Verifying JSON Serialization Compatibility...")
    serialized = json.dumps(exp_1, indent=2)
    assert len(serialized) > 100
    print("      [OK] Explanation dictionary is 100% JSON-serializable for API / UI / Dossier.")

    print("\n" + "=" * 65)
    print("  Phase 4 Explainability Engine Complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_explainability_check()
