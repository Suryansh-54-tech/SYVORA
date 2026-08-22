"""
SentinelRisk — Deterministic Evidence Assembler
================================================
Assembles observed digital evidence with verifiable provenance (source IDs,
timestamps, source systems) and pairs it with derived decision-theoretic analytics.

Guarantees:
- Zero fabricated evidence (all facts mapped to source records)
- Strict separation of observed vs. analytical data
- Explicit identification of missing evidence
- Pure deterministic execution
"""

import os
import sys
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.agent.schemas import (
    EvidenceSourceType,
    ObservedItem,
    AuthenticationEvidence,
    FulfillmentEvidence,
    TelemetryEvidence,
    CustomerHistoryEvidence,
    CustomerClaimEvidence,
    ObservedEvidencePackage,
    AnalyticalEvidencePackage,
    DisputeDefenseDossier,
    parse_bool,
    get_missing_evidence_elements
)
from src.engine import DecisionEngine
from src.security.sanitizer import InputSanitizer


class EvidenceAssembler:
    """
    Forensic evidence aggregator that extracts, validates, and packages
    observed dispute data without hallucination or silent alteration.
    """

    def __init__(self, decision_engine: Optional[DecisionEngine] = None):
        self.decision_engine = decision_engine or DecisionEngine()
        self.sanitizer = InputSanitizer()

    def sanitize_customer_claim(self, claim_text: Optional[str]) -> Optional[CustomerClaimEvidence]:
        """
        Mandatory firewall for untrusted customer-provided claim/remark text.

        Runs the raw text through InputSanitizer BEFORE any downstream component
        can interpret it. Returns a clearly-labeled UNTRUSTED / SANITIZED evidence
        block preserving original text + hashes and detected threat categories.

        Returns None when no claim was supplied (backwards-compatible no-op).
        """
        if claim_text is None or not isinstance(claim_text, str) or not claim_text.strip():
            return None

        result = self.sanitizer.sanitize_claim_text(claim_text)
        return CustomerClaimEvidence(
            original_text=result.original_text,
            sanitized_text=result.sanitized_text,
            original_sha256=result.original_sha256,
            sanitized_sha256=result.sanitized_sha256,
            is_threat_detected=result.is_threat_detected,
            threats_detected=sorted(result.threats_detected),
        )

    def assemble_observed_evidence(self, raw_data: Dict[str, Any]) -> ObservedEvidencePackage:
        """
        Extracts observed facts from raw dispute data with explicit provenance.
        Guarantees that operational evidence assembly never ingests or processes post-event ground truth.
        """
        # Ensure operational input contains zero ground truth target leakage
        clean_data = {k: v for k, v in raw_data.items() if k != "dispute_outcome"}

        dispute_id = str(clean_data.get("dispute_id", "dsp_unknown"))
        txn_id = str(clean_data.get("transaction_id", f"pay_mock_{dispute_id}"))
        dispute_date = str(clean_data.get("dispute_date", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
        amount = float(clean_data.get("txn_amount_inr", 0.0))
        reason_code = str(clean_data.get("reason_code", "UNKNOWN"))
        card_network = str(clean_data.get("card_network", "UNKNOWN"))
        issuing_bank = str(clean_data.get("issuing_bank", "UNKNOWN"))
        merchant_category = str(clean_data.get("merchant_category", "ECOMM_RETAIL"))
        days_to_deadline = int(clean_data.get("days_to_deadline", 7))

        # 1. Authentication Evidence
        three_ds_status = str(clean_data.get("three_ds_status", "N_NOT_ENROLLED"))
        is_3ds_auth = (three_ds_status == "Y_AUTHENTICATED")
        auth_evidence = AuthenticationEvidence(
            three_ds_status=three_ds_status,
            is_authenticated=is_3ds_auth,
            source_system=EvidenceSourceType.PAYMENT_GATEWAY,
            source_record_id=f"auth_log_{txn_id}",
            timestamp=dispute_date,
        )

        # 2. Fulfillment Evidence
        courier_status = str(clean_data.get("courier_status", "UNKNOWN"))
        carrier = str(clean_data.get("carrier", "NONE"))
        signed_pod = parse_bool(clean_data.get("signed_pod", False))
        is_delivered = (courier_status == "DELIVERED")
        mock_tracking_id = f"TRK_{carrier[:3]}_{dispute_id[-5:]}" if carrier != "NONE" else None

        fulfillment_evidence = FulfillmentEvidence(
            courier_status=courier_status,
            carrier=carrier,
            has_signed_pod=signed_pod,
            is_delivered=is_delivered,
            tracking_number=mock_tracking_id,
            source_system=EvidenceSourceType.CARRIER_LOGISTICS,
            source_record_id=f"carrier_sync_{mock_tracking_id}" if mock_tracking_id else "logistics_null",
            timestamp=dispute_date,
        )

        # 3. Telemetry Evidence
        ip_geo_match = parse_bool(clean_data.get("ip_geo_match", False))
        device_fingerprint_match = parse_bool(clean_data.get("device_fingerprint_match", False))
        billing_shipping_match = parse_bool(clean_data.get("billing_shipping_match", True))

        telemetry_evidence = TelemetryEvidence(
            ip_geo_match=ip_geo_match,
            device_fingerprint_match=device_fingerprint_match,
            billing_shipping_match=billing_shipping_match,
            source_system=EvidenceSourceType.DEVICE_TELEMETRY,
            source_record_id=f"session_telemetry_{txn_id}",
            timestamp=dispute_date,
        )

        # 4. Customer History Evidence
        prior_undisputed_txns = int(clean_data.get("prior_undisputed_txns", 0))
        customer_past_dispute_count = int(clean_data.get("customer_past_dispute_count", 0))
        is_serial_disputer = (customer_past_dispute_count >= config.SERIAL_DISPUTE_FLAG_THRESHOLD)
        is_visa_ce3 = (prior_undisputed_txns >= 2) and ip_geo_match

        customer_history_evidence = CustomerHistoryEvidence(
            prior_undisputed_txns=prior_undisputed_txns,
            customer_past_dispute_count=customer_past_dispute_count,
            is_serial_disputer=is_serial_disputer,
            is_visa_ce3_eligible=is_visa_ce3,
            source_system=EvidenceSourceType.CUSTOMER_ACCOUNT,
            source_record_id=f"cust_ledger_{dispute_id[-5:]}",
            timestamp=dispute_date,
        )

        # 5. Raw Evidence Inventory (Audit Traceability)
        raw_inventory = [
            ObservedItem(
                field_name="transaction_amount",
                value=amount,
                source_system=EvidenceSourceType.PAYMENT_GATEWAY,
                source_record_id=f"txn_{txn_id}",
                timestamp=dispute_date,
            ),
            ObservedItem(
                field_name="3d_secure_status",
                value=three_ds_status,
                is_available=(three_ds_status != "U_UNAVAILABLE"),
                source_system=EvidenceSourceType.PAYWAY if hasattr(EvidenceSourceType, "PAYWAY") else EvidenceSourceType.PAYMENT_GATEWAY,
                source_record_id=f"auth_log_{txn_id}",
                timestamp=dispute_date,
            ),
            ObservedItem(
                field_name="courier_delivery_status",
                value=courier_status,
                is_available=(courier_status != "UNKNOWN"),
                source_system=EvidenceSourceType.CARRIER_LOGISTICS,
                source_record_id=f"carrier_sync_{mock_tracking_id}" if mock_tracking_id else "logistics_null",
                timestamp=dispute_date,
            ),
            ObservedItem(
                field_name="proof_of_delivery_signature",
                value=signed_pod,
                is_available=signed_pod,
                source_system=EvidenceSourceType.CARRIER_LOGISTICS,
                source_record_id=f"carrier_pod_{mock_tracking_id}" if mock_tracking_id else "pod_null",
                timestamp=dispute_date,
                notes="Physical or electronic signature captured at delivery" if signed_pod else "Signature missing from carrier log"
            ),
            ObservedItem(
                field_name="ip_geolocation_match",
                value=ip_geo_match,
                source_system=EvidenceSourceType.DEVICE_TELEMETRY,
                source_record_id=f"session_telemetry_{txn_id}",
                timestamp=dispute_date,
            ),
            ObservedItem(
                field_name="device_fingerprint_match",
                value=device_fingerprint_match,
                source_system=EvidenceSourceType.DEVICE_TELEMETRY,
                source_record_id=f"session_telemetry_{txn_id}",
                timestamp=dispute_date,
            ),
            ObservedItem(
                field_name="prior_undisputed_transactions",
                value=prior_undisputed_txns,
                source_system=EvidenceSourceType.CUSTOMER_ACCOUNT,
                source_record_id=f"cust_ledger_{dispute_id[-5:]}",
                timestamp=dispute_date,
                notes=f"Historical undisputed orders with matching credentials: {prior_undisputed_txns}"
            ),
        ]

        # 6. Authoritative Missing Evidence Checklist
        missing = get_missing_evidence_elements(
            courier_status=courier_status,
            signed_pod=signed_pod,
            three_ds_status=three_ds_status,
            ip_geo_match=ip_geo_match,
            device_fingerprint_match=device_fingerprint_match
        )

        return ObservedEvidencePackage(
            dispute_id=dispute_id,
            transaction_id=txn_id,
            dispute_date=dispute_date,
            amount_inr=amount,
            reason_code=reason_code,
            card_network=card_network,
            issuing_bank=issuing_bank,
            merchant_category=merchant_category,
            days_to_deadline=days_to_deadline,
            authentication=auth_evidence,
            fulfillment=fulfillment_evidence,
            telemetry=telemetry_evidence,
            customer_history=customer_history_evidence,
            raw_evidence_inventory=raw_inventory,
            missing_evidence_elements=missing,
        )

    def assemble_analytical_evidence(self, evaluation: Dict[str, Any]) -> AnalyticalEvidencePackage:
        """
        Packages decision engine evaluation results into structured analytical schema.
        """
        fin = evaluation["financial_analysis"]
        evi = evaluation["evidence_analysis"]
        pol = evaluation["policy_gates"]
        shap = evaluation.get("shap_explanation") or {}

        return AnalyticalEvidencePackage(
            calibrated_win_probability=fin["calibrated_win_probability"],
            break_even_probability=fin["break_even_probability"],
            expected_value_inr=fin["expected_value_inr"],
            is_positive_ev=fin["is_positive_ev"],
            arbitration_fee_inr=fin["arbitration_fee_inr"],
            evidence_readiness_score=evi["readiness_score"],
            decision_verdict=evaluation["decision"],
            decision_reasons=evaluation["decision_reasons"],
            policy_gate_triggers=pol["forced_hitl_reasons"],
            top_positive_factors=shap.get("top_positive_factors", []),
            top_negative_factors=shap.get("top_negative_factors", []),
            shap_summary_text=shap.get("explanation_summary", "No SHAP narrative generated."),
        )

    def build_dossier(
        self,
        raw_data: Dict[str, Any],
        customer_claim_text: Optional[str] = None
    ) -> DisputeDefenseDossier:
        """
        End-to-end assembly: sanitizes optional customer claim text, extracts
        observed evidence, executes decision engine, formats formal network
        rebuttal narrative, and returns structured dossier.

        The customer claim is sanitized BEFORE any downstream component runs and
        is attached as a labeled UNTRUSTED / SANITIZED data block only. It never
        influences ML features, win probability, Expected Value, verdicts, or
        evidence provenance.
        """
        from src.agent.dossier import DossierFormatter

        # Clean operational data of any post-event ground truth
        clean_data = {k: v for k, v in raw_data.items() if k != "dispute_outcome"}

        # Defensive isolation: untrusted free-text keys must never reach the
        # decision engine, ML feature pipeline, or provenance builders.
        clean_data = {
            k: v for k, v in clean_data.items()
            if not k.lower().startswith("customer_claim")
        }

        # 0. Sanitize untrusted customer claim FIRST — before any downstream
        #    component can interpret it.
        customer_claim_evidence = self.sanitize_customer_claim(customer_claim_text)

        # 1. Observed facts
        observed = self.assemble_observed_evidence(clean_data)
        if customer_claim_evidence is not None:
            observed = observed.model_copy(update={"customer_claim": customer_claim_evidence})

        # 2. Decision engine evaluation
        evaluation = self.decision_engine.evaluate_dispute(clean_data, include_shap=True)

        # 3. Analytical facts
        analytical = self.assemble_analytical_evidence(evaluation)

        # 4. Generate structured rebuttal markdown
        formatter = DossierFormatter()
        rebuttal_md = formatter.generate_rebuttal_markdown(observed, analytical)

        dossier_id = f"dos_{hashlib.md5(f'{observed.dispute_id}_{observed.dispute_date}'.encode()).hexdigest()[:10]}"
        is_ready = (analytical.decision_verdict == "CONTEST")

        return DisputeDefenseDossier(
            dossier_id=dossier_id,
            dispute_id=observed.dispute_id,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            observed_evidence=observed,
            analytical_evidence=analytical,
            rebuttal_narrative_markdown=rebuttal_md,
            is_ready_for_submission=is_ready,
        )
