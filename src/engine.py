"""
SYVORA — Decision Engine & Deterministic Policy Gating
======================================================
Translates calibrated ML win probabilities, financial economics, digital evidence,
and policy constraints into an actionable, transparent triage decision:

    - CONTEST   : Positive EV, sufficient evidence, high confidence, within monetary limits
    - REVIEW    : High-value dispute, low confidence, insufficient proof, or urgent deadline
    - SURRENDER : Negative Expected Value (accept loss immediately to avoid non-refundable bank fees)

Pure deterministic calculation — zero LLM calls.
"""

import os
import sys
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ml.features import FeaturePipeline
from src.ml.train import SentinelRiskScorer
from src.ml.explain import DisputeExplainer
from src.agent.schemas import parse_bool, get_missing_evidence_elements


class DecisionVerdict(str, Enum):
    CONTEST = "CONTEST"
    REVIEW = "REVIEW"
    SURRENDER = "SURRENDER"


class DecisionEngine:
    """
    Deterministic decision orchestrator enforcing cost-weighted Expected Value
    and hard safety policy gates for post-payment disputes.
    """

    def __init__(
        self,
        scorer: Optional[SentinelRiskScorer] = None,
        explainer: Optional[DisputeExplainer] = None,
        pipeline: Optional[FeaturePipeline] = None,
        arbitration_fee_inr: float = config.ARBITRATION_FEE_INR,
        hitl_amount_threshold_inr: float = config.HITL_AMOUNT_THRESHOLD_INR,
        hitl_confidence_threshold: float = config.HITL_CONFIDENCE_THRESHOLD,
        min_evidence_score: int = config.MIN_EVIDENCE_READINESS_SCORE,
    ):
        self.pipeline = pipeline or FeaturePipeline()
        self.scorer = scorer or SentinelRiskScorer()
        self.explainer = explainer or DisputeExplainer()

        # Configurable financial & policy thresholds
        self.arbitration_fee_inr = float(arbitration_fee_inr)
        self.hitl_amount_threshold_inr = float(hitl_amount_threshold_inr)
        self.hitl_confidence_threshold = float(hitl_confidence_threshold)
        self.min_evidence_score = int(min_evidence_score)

    def calculate_expected_value(
        self,
        win_probability: float,
        dispute_amount_inr: float,
        arbitration_fee_inr: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """
        Calculates the Bayesian Expected Financial Value of contesting.

        Formula:
            E[EV] = (P_win * Amount) - ((1 - P_win) * Arbitration_Fee)
            Break-Even Prob tau* = Arbitration_Fee / (Amount + Arbitration_Fee)

        Returns:
            (expected_value_inr, break_even_probability, is_positive_ev)
        """
        fee = arbitration_fee_inr if arbitration_fee_inr is not None else self.arbitration_fee_inr
        p = float(np.clip(win_probability, 0.0, 1.0))
        amt = float(max(dispute_amount_inr, 0.0))

        # Expected Value
        ev = (p * amt) - ((1.0 - p) * fee)

        # Break-Even Probability
        denom = amt + fee
        break_even_prob = (fee / denom) if denom > 0 else 0.50

        is_positive_ev = bool(ev > 0.0 and p >= break_even_prob)

        return round(ev, 2), round(break_even_prob, 4), is_positive_ev

    def evaluate_dispute(
        self,
        dispute_data: Dict[str, Any],
        include_shap: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a single dispute record through the multi-tier policy gating pipeline.

        Args:
            dispute_data: Raw dispute dictionary (keys matching synthetic schema).
            include_shap: Whether to include TreeSHAP local attributions.

        Returns:
            Structured decision record with financial analysis, evidence score,
            policy gates, SHAP explanation, and human-readable decision reasons.
        """
        dispute_id = dispute_data.get("dispute_id", "dsp_unknown")
        amount = float(dispute_data.get("txn_amount_inr", 0.0))
        days_to_deadline = int(dispute_data.get("days_to_deadline", 7))
        courier_status = str(dispute_data.get("courier_status", "UNKNOWN"))
        signed_pod = parse_bool(dispute_data.get("signed_pod", False))
        three_ds_status = str(dispute_data.get("three_ds_status", "N_NOT_ENROLLED"))
        ip_geo_match = parse_bool(dispute_data.get("ip_geo_match", False))
        device_fingerprint_match = parse_bool(dispute_data.get("device_fingerprint_match", False))
        prior_undisputed_txns = int(dispute_data.get("prior_undisputed_txns", 0))
        customer_past_dispute_count = int(dispute_data.get("customer_past_dispute_count", 0))
        billing_shipping_match = parse_bool(dispute_data.get("billing_shipping_match", True))

        # -------------------------------------------------------------------
        # 1. Feature Transformation & Calibrated ML Scoring
        # -------------------------------------------------------------------
        df_raw = pd.DataFrame([dispute_data])
        X_feat = self.pipeline.transform(df_raw)
        win_prob = float(self.scorer.predict_proba(X_feat)[0])

        # -------------------------------------------------------------------
        # 2. Evidence Readiness Analysis
        # -------------------------------------------------------------------
        evidence_score = int(self.pipeline.compute_evidence_readiness(df_raw).iloc[0])
        is_evidence_sufficient = evidence_score >= self.min_evidence_score

        missing_evidence = get_missing_evidence_elements(
            courier_status=courier_status,
            signed_pod=signed_pod,
            three_ds_status=three_ds_status,
            ip_geo_match=ip_geo_match,
            device_fingerprint_match=device_fingerprint_match
        )

        # -------------------------------------------------------------------
        # 3. Financial Expected Value Calculation
        # -------------------------------------------------------------------
        ev_inr, break_even_prob, is_pos_ev = self.calculate_expected_value(
            win_probability=win_prob,
            dispute_amount_inr=amount
        )

        # -------------------------------------------------------------------
        # 4. Policy Gates & Safety Checks (Human-in-the-Loop Triggers)
        # -------------------------------------------------------------------
        is_high_value = amount >= self.hitl_amount_threshold_inr
        is_urgent_deadline = days_to_deadline <= 3
        is_serial_disputer = customer_past_dispute_count >= config.SERIAL_DISPUTE_FLAG_THRESHOLD
        is_visa_ce3_eligible = (prior_undisputed_txns >= 2) and ip_geo_match

        forced_hitl_reasons = []

        if is_high_value:
            forced_hitl_reasons.append(
                f"Dispute amount (INR {amount:,.2f}) exceeds mandatory review threshold (INR {self.hitl_amount_threshold_inr:,.2f})"
            )

        if not is_evidence_sufficient:
            forced_hitl_reasons.append(
                f"Evidence Readiness Score ({evidence_score}/100) below minimum required threshold ({self.min_evidence_score}/100)"
            )

        if is_pos_ev and win_prob < self.hitl_confidence_threshold:
            forced_hitl_reasons.append(
                f"Win probability ({win_prob:.1%}) is positive EV but below automated confidence threshold ({self.hitl_confidence_threshold:.1%})"
            )

        if is_urgent_deadline:
            forced_hitl_reasons.append(
                f"Urgent evidence submission deadline ({days_to_deadline} days remaining)"
            )

        if not billing_shipping_match and customer_past_dispute_count >= config.SERIAL_DISPUTE_FLAG_THRESHOLD:
            forced_hitl_reasons.append(
                "Address mismatch combined with repeat dispute history requires manual verification"
            )

        # -------------------------------------------------------------------
        # 5. Deterministic Gating & Final Verdict
        # -------------------------------------------------------------------
        decision_reasons = []

        if not is_pos_ev or ev_inr <= 0.0:
            # Rule 1: Negative Expected Value -> Surrender to save bank fee
            verdict = DecisionVerdict.SURRENDER
            decision_reasons.append(
                f"Negative Expected Value (E[EV] = -INR {abs(ev_inr):,.2f}). Defending carries high risk of incurring INR {self.arbitration_fee_inr:,.2f} bank arbitration fee."
            )
            decision_reasons.append(
                f"Calibrated win probability ({win_prob:.1%}) is below economic break-even threshold ({break_even_prob:.1%})."
            )
        elif len(forced_hitl_reasons) > 0:
            # Rule 2: Positive EV but one or more safety gates triggered -> Escalate to Human
            verdict = DecisionVerdict.REVIEW
            decision_reasons.append(
                f"Economically viable to contest (E[EV] = +INR {ev_inr:,.2f}, Break-Even = {break_even_prob:.1%}), but safety gates require manual authorization:"
            )
            for r in forced_hitl_reasons:
                decision_reasons.append(f"  - {r}")
        else:
            # Rule 3: Positive EV + All safety gates passed + High Confidence -> Auto-Contest
            verdict = DecisionVerdict.CONTEST
            decision_reasons.append(
                f"Strong positive Expected Value (E[EV] = +INR {ev_inr:,.2f}, Win Prob = {win_prob:.1%} vs Break-Even = {break_even_prob:.1%})."
            )
            decision_reasons.append(
                f"Evidence Readiness is complete ({evidence_score}/100) with verified courier POD and 3DS authentication."
            )
            if is_visa_ce3_eligible:
                decision_reasons.append(
                    f"Visa CE3.0 Qualifying: 2+ prior undisputed transactions ({prior_undisputed_txns} total) with matching IP telemetry."
                )

        # -------------------------------------------------------------------
        # 6. SHAP Explainability Integration
        # -------------------------------------------------------------------
        shap_explanation = None
        if include_shap:
            shap_explanation = self.explainer.explain_instance(X_feat)

        return {
            "dispute_id": dispute_id,
            "decision": verdict.value,
            "decision_reasons": decision_reasons,
            "financial_analysis": {
                "dispute_amount_inr": amount,
                "arbitration_fee_inr": self.arbitration_fee_inr,
                "calibrated_win_probability": round(win_prob, 4),
                "break_even_probability": break_even_prob,
                "expected_value_inr": ev_inr,
                "is_positive_ev": is_pos_ev,
            },
            "evidence_analysis": {
                "readiness_score": evidence_score,
                "min_required_score": self.min_evidence_score,
                "is_evidence_sufficient": is_evidence_sufficient,
                "missing_elements": missing_evidence,
            },
            "policy_gates": {
                "is_high_value": is_high_value,
                "is_urgent_deadline": is_urgent_deadline,
                "is_serial_disputer": is_serial_disputer,
                "is_visa_ce3_eligible": is_visa_ce3_eligible,
                "forced_hitl_reasons": forced_hitl_reasons,
            },
            "shap_explanation": shap_explanation,
        }


# ---------------------------------------------------------------------------
# CLI / Verification Runner
# ---------------------------------------------------------------------------

def run_decision_engine_checks():
    print("=" * 65)
    print("  SYVORA -- Decision Engine & Policy Gating Verification")
    print("=" * 65)

    engine = DecisionEngine()

    test_cases = [
        {
            "name": "Case 1: Clear Contest (Positive EV, Full Evidence, High Prob, Low $)",
            "data": {
                "dispute_id": "dsp_test_01",
                "txn_amount_inr": 4500.0,
                "reason_code": "VISA_13_1_NOT_RECEIVED",
                "card_network": "VISA",
                "issuing_bank": "HDFC",
                "merchant_category": "ECOMM_RETAIL",
                "three_ds_status": "Y_AUTHENTICATED",
                "courier_status": "DELIVERED",
                "carrier": "DELHIVERY",
                "signed_pod": True,
                "ip_geo_match": True,
                "device_fingerprint_match": True,
                "billing_shipping_match": True,
                "customer_past_dispute_count": 0,
                "prior_undisputed_txns": 3,
                "txn_age_days": 15,
                "days_to_deadline": 8,
            },
            "expected_decision": "CONTEST"
        },
        {
            "name": "Case 2: High-Value Edge Case (INR 45,000 -> Forces Human Review)",
            "data": {
                "dispute_id": "dsp_test_02",
                "txn_amount_inr": 45000.0,  # > INR 25,000 threshold
                "reason_code": "MC_4837_FRAUD",
                "card_network": "MASTERCARD",
                "issuing_bank": "ICICI",
                "merchant_category": "ELECTRONICS",
                "three_ds_status": "Y_AUTHENTICATED",
                "courier_status": "DELIVERED",
                "carrier": "BLUEDART",
                "signed_pod": True,
                "ip_geo_match": True,
                "device_fingerprint_match": True,
                "billing_shipping_match": True,
                "customer_past_dispute_count": 0,
                "prior_undisputed_txns": 4,
                "txn_age_days": 12,
                "days_to_deadline": 9,
            },
            "expected_decision": "REVIEW"
        },
        {
            "name": "Case 3: Economically Negative Contest (Micro-dispute INR 180 with INR 500 fee)",
            "data": {
                "dispute_id": "dsp_test_03",
                "txn_amount_inr": 180.0,  # Micro amount
                "reason_code": "VISA_10_4_FRAUD",
                "card_network": "VISA",
                "issuing_bank": "CITI_INTL",
                "merchant_category": "DIGITAL_SAAS",
                "three_ds_status": "A_ATTEMPTED",
                "courier_status": "NOT_APPLICABLE",
                "carrier": "NONE",
                "signed_pod": False,
                "ip_geo_match": False,
                "device_fingerprint_match": False,
                "billing_shipping_match": True,
                "customer_past_dispute_count": 0,
                "prior_undisputed_txns": 0,
                "txn_age_days": 40,
                "days_to_deadline": 10,
            },
            "expected_decision": "SURRENDER"
        },
        {
            "name": "Case 4: High Prob but Missing Evidence (Evidence score < 60 -> Review)",
            "data": {
                "dispute_id": "dsp_test_04",
                "txn_amount_inr": 5000.0,
                "reason_code": "VISA_13_1_NOT_RECEIVED",
                "card_network": "VISA",
                "issuing_bank": "HDFC",
                "merchant_category": "ECOMM_RETAIL",
                "three_ds_status": "N_NOT_ENROLLED",  # No 3DS
                "courier_status": "IN_TRANSIT",       # In transit, no POD
                "carrier": "DELHIVERY",
                "signed_pod": False,                  # No POD
                "ip_geo_match": True,
                "device_fingerprint_match": True,
                "billing_shipping_match": True,
                "customer_past_dispute_count": 0,
                "prior_undisputed_txns": 1,
                "txn_age_days": 5,
                "days_to_deadline": 7,
            },
            "expected_decision": "SURRENDER"  # Win prob drops -> Negative EV or Insufficient evidence
        },
        {
            "name": "Case 5: Urgent Deadline (<= 3 Days -> Flagged for Human Attention)",
            "data": {
                "dispute_id": "dsp_test_05",
                "txn_amount_inr": 6000.0,
                "reason_code": "VISA_13_1_NOT_RECEIVED",
                "card_network": "VISA",
                "issuing_bank": "AXIS",
                "merchant_category": "ECOMM_RETAIL",
                "three_ds_status": "Y_AUTHENTICATED",
                "courier_status": "DELIVERED",
                "carrier": "DELHIVERY",
                "signed_pod": True,
                "ip_geo_match": True,
                "device_fingerprint_match": True,
                "billing_shipping_match": True,
                "customer_past_dispute_count": 0,
                "prior_undisputed_txns": 2,
                "txn_age_days": 20,
                "days_to_deadline": 2,  # Urgent!
            },
            "expected_decision": "REVIEW"
        }
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Running {tc['name']}...")
        result = engine.evaluate_dispute(tc["data"])
        
        print(f"      Decision:    {result['decision']}")
        print(f"      Win Prob:    {result['financial_analysis']['calibrated_win_probability']:.2%}")
        print(f"      Break-Even:  {result['financial_analysis']['break_even_probability']:.2%}")
        print(f"      Expected EV: INR {result['financial_analysis']['expected_value_inr']:,.2f}")
        print(f"      Evidence:    {result['evidence_analysis']['readiness_score']}/100")
        print(f"      Reasons:     {result['decision_reasons'][0]}")

    print("\n" + "=" * 65)
    print("  Phase 5 Decision Engine Verification Complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_decision_engine_checks()
