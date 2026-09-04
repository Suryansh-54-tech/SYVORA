"""
NYAYANTRA — Deterministic Offline Claim–Evidence Consistency Advisor
=================================================================
Compares extracted customer claim intents against verified digital evidence records.

Guarantees & Constraints:
- Pure Python deterministic rule heuristics; zero network calls; zero external dependencies.
- Zero access to ML feature pipelines, Random Forest models, Platt calibration,
  TreeSHAP explainers, Expected Value calculations, policy gates, readiness scores,
  or defense verdicts.
- Produces ADVISORY findings with `advisory_only: Literal[True] = True` strictly enforced.
- Strictly observational status nomenclature:
    * CONSISTENT_WITH_EVIDENCE
    * CONTRADICTED_BY_EVIDENCE
    * MIXED_EVIDENCE
    * INSUFFICIENT_EVIDENCE
    * NO_ASSESSMENT
- Never declares definitive truth, customer lying, or confirmed fraud.
- References only authentic observed evidence attributes — zero fabricated IDs.
"""

from typing import Optional, List
from src.agent.schemas import (
    ClaimIntent,
    ClaimSignal,
    ClaimSignalPackage,
    ConsistencyStatus,
    EvidenceSignalConsidered,
    ConsistencyFinding,
    ConsistencyEvaluation,
    ObservedEvidencePackage,
    EvidenceSourceType,
)


class DeterministicConsistencyAdvisor:
    """
    Deterministic rule engine that cross-references customer claim intents
    with verified evidence facts.
    """

    @classmethod
    def evaluate_consistency(
        cls,
        claim_pkg: Optional[ClaimSignalPackage],
        observed: ObservedEvidencePackage,
    ) -> ConsistencyEvaluation:
        """
        Cross-references structured claim signals with verified evidence facts.

        Returns an immutable ConsistencyEvaluation package.
        """
        if (
            claim_pkg is None
            or not claim_pkg.has_structured_claim
            or not claim_pkg.signals
            or claim_pkg.primary_intent == ClaimIntent.OTHER
        ):
            return ConsistencyEvaluation(
                primary_finding=None,
                secondary_findings=[],
                overall_status=ConsistencyStatus.NO_ASSESSMENT,
                source_sanitized_sha256=claim_pkg.source_sanitized_sha256 if claim_pkg else "",
                summary_text="No structured claim intent recognized for consistency evaluation.",
                advisory_only=True,
            )

        # 1. Primary finding
        primary_signal = claim_pkg.signals[0]
        primary_finding = cls._evaluate_single_intent(primary_signal, observed)

        # 2. Secondary findings
        secondary_findings: List[ConsistencyFinding] = []
        for sig in claim_pkg.signals[1:]:
            secondary_findings.append(cls._evaluate_single_intent(sig, observed))

        # 3. Aggregate overall status
        all_findings = [primary_finding] + secondary_findings
        overall_status = cls._determine_overall_status(all_findings)

        # 4. Generate human-readable summary
        summary_text = cls._build_summary(primary_finding, secondary_findings, overall_status)

        return ConsistencyEvaluation(
            primary_finding=primary_finding,
            secondary_findings=secondary_findings,
            overall_status=overall_status,
            source_sanitized_sha256=claim_pkg.source_sanitized_sha256,
            summary_text=summary_text,
            advisory_only=True,
        )

    @classmethod
    def _evaluate_single_intent(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        intent = signal.intent

        if intent == ClaimIntent.NON_DELIVERY:
            return cls._evaluate_non_delivery(signal, observed)
        elif intent == ClaimIntent.UNAUTHORIZED_TRANSACTION:
            return cls._evaluate_unauthorized(signal, observed)
        elif intent == ClaimIntent.DUPLICATE_CHARGE:
            return cls._evaluate_duplicate(signal, observed)
        elif intent == ClaimIntent.WRONG_AMOUNT:
            return cls._evaluate_wrong_amount(signal, observed)
        elif intent == ClaimIntent.REFUND_NOT_RECEIVED:
            return cls._evaluate_refund(signal, observed)
        elif intent == ClaimIntent.CANCELLATION:
            return cls._evaluate_cancellation(signal, observed)
        else:
            return ConsistencyFinding(
                intent=ClaimIntent.OTHER,
                status=ConsistencyStatus.NO_ASSESSMENT,
                rule_matching_confidence=signal.confidence_score,
                evidence_signals=[],
                explanation="No structured intent recognized for consistency evaluation.",
                advisory_only=True,
            )

    @classmethod
    def _evaluate_non_delivery(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        ful = observed.fulfillment
        evidence_signals = [
            EvidenceSignalConsidered(
                field_name="courier_status",
                value=ful.courier_status,
                source_system=ful.source_system.value,
                source_record_id=ful.source_record_id,
            ),
            EvidenceSignalConsidered(
                field_name="has_signed_pod",
                value=ful.has_signed_pod,
                source_system=ful.source_system.value,
                source_record_id=ful.source_record_id,
            ),
        ]

        # Delivery & POD verified
        if ful.courier_status == "DELIVERED" and ful.has_signed_pod:
            status = ConsistencyStatus.CONTRADICTED_BY_EVIDENCE
            explanation = "Carrier logistics records confirm package delivery with physical signed Proof of Delivery (POD)."
        # Returned with no POD
        elif ful.courier_status in ["RETURNED", "RETURN_TO_ORIGIN", "RTO"] and not ful.has_signed_pod:
            status = ConsistencyStatus.CONSISTENT_WITH_EVIDENCE
            explanation = "Carrier logistics records confirm the shipment was returned to origin; no delivery POD on file."
        # Returned but POD present (anomalous)
        elif ful.courier_status in ["RETURNED", "RETURN_TO_ORIGIN", "RTO"] and ful.has_signed_pod:
            status = ConsistencyStatus.MIXED_EVIDENCE
            explanation = "Carrier logistics records indicate returned shipment, but a signed POD record is present."
        # Delivered without POD
        elif ful.courier_status == "DELIVERED" and not ful.has_signed_pod:
            status = ConsistencyStatus.MIXED_EVIDENCE
            explanation = "Carrier logs show delivery status, but signed Proof of Delivery (POD) signature is missing."
        # In transit / unknown / not applicable
        else:
            status = ConsistencyStatus.INSUFFICIENT_EVIDENCE
            explanation = f"Carrier fulfillment status is '{ful.courier_status}'; delivery cannot be conclusively established from existing evidence."

        return ConsistencyFinding(
            intent=signal.intent,
            status=status,
            rule_matching_confidence=signal.confidence_score,
            evidence_signals=evidence_signals,
            explanation=explanation,
            advisory_only=True,
        )

    @classmethod
    def _evaluate_unauthorized(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        auth = observed.authentication
        telem = observed.telemetry
        evidence_signals = [
            EvidenceSignalConsidered(
                field_name="three_ds_status",
                value=auth.three_ds_status,
                source_system=auth.source_system.value,
                source_record_id=auth.source_record_id,
            ),
            EvidenceSignalConsidered(
                field_name="ip_geo_match",
                value=telem.ip_geo_match,
                source_system=telem.source_system.value,
                source_record_id=telem.source_record_id,
            ),
            EvidenceSignalConsidered(
                field_name="device_fingerprint_match",
                value=telem.device_fingerprint_match,
                source_system=telem.source_system.value,
                source_record_id=telem.source_record_id,
            ),
            EvidenceSignalConsidered(
                field_name="billing_shipping_match",
                value=telem.billing_shipping_match,
                source_system=telem.source_system.value,
                source_record_id=telem.source_record_id,
            ),
        ]

        has_telemetry_match = (
            telem.ip_geo_match
            or telem.device_fingerprint_match
            or telem.billing_shipping_match
        )

        if auth.is_authenticated and has_telemetry_match:
            status = ConsistencyStatus.CONTRADICTED_BY_EVIDENCE
            explanation = "Transaction completed via 3D Secure authentication with matching customer session IP/device/address telemetry."
        elif auth.is_authenticated and not has_telemetry_match:
            status = ConsistencyStatus.MIXED_EVIDENCE
            explanation = "3D Secure protocol was verified, but session IP, device fingerprint, and address telemetry do not match customer profile."
        else:
            status = ConsistencyStatus.INSUFFICIENT_EVIDENCE
            explanation = "3D Secure authentication was incomplete or unverified; session telemetry alone is insufficient to prove authorization."

        return ConsistencyFinding(
            intent=signal.intent,
            status=status,
            rule_matching_confidence=signal.confidence_score,
            evidence_signals=evidence_signals,
            explanation=explanation,
            advisory_only=True,
        )

    @classmethod
    def _evaluate_duplicate(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        evidence_signals = [
            EvidenceSignalConsidered(
                field_name="transaction_amount_inr",
                value=observed.amount_inr,
                source_system=EvidenceSourceType.PAYMENT_GATEWAY.value,
                source_record_id=observed.transaction_id,
            )
        ]
        return ConsistencyFinding(
            intent=signal.intent,
            status=ConsistencyStatus.INSUFFICIENT_EVIDENCE,
            rule_matching_confidence=signal.confidence_score,
            evidence_signals=evidence_signals,
            explanation="No multi-transaction gateway ledger records available in current simulated evidence to confirm duplicate charges.",
            advisory_only=True,
        )

    @classmethod
    def _evaluate_wrong_amount(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        evidence_signals = [
            EvidenceSignalConsidered(
                field_name="transaction_amount_inr",
                value=observed.amount_inr,
                source_system=EvidenceSourceType.PAYMENT_GATEWAY.value,
                source_record_id=observed.transaction_id,
            )
        ]
        return ConsistencyFinding(
            intent=signal.intent,
            status=ConsistencyStatus.INSUFFICIENT_EVIDENCE,
            rule_matching_confidence=signal.confidence_score,
            evidence_signals=evidence_signals,
            explanation=f"Verified order transaction amount is INR {observed.amount_inr:,.2f}; no secondary discrepancy or settlement ledger record exists in current evidence package.",
            advisory_only=True,
        )

    @classmethod
    def _evaluate_refund(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        return ConsistencyFinding(
            intent=signal.intent,
            status=ConsistencyStatus.INSUFFICIENT_EVIDENCE,
            rule_matching_confidence=signal.confidence_score,
            evidence_signals=[],
            explanation="Merchant refund gateway transaction logs are unavailable in current evidence package.",
            advisory_only=True,
        )

    @classmethod
    def _evaluate_cancellation(
        cls,
        signal: ClaimSignal,
        observed: ObservedEvidencePackage,
    ) -> ConsistencyFinding:
        return ConsistencyFinding(
            intent=signal.intent,
            status=ConsistencyStatus.INSUFFICIENT_EVIDENCE,
            rule_matching_confidence=signal.confidence_score,
            evidence_signals=[],
            explanation="Order cancellation timestamps and customer communication logs are unavailable in current evidence package.",
            advisory_only=True,
        )

    @staticmethod
    def _determine_overall_status(findings: List[ConsistencyFinding]) -> ConsistencyStatus:
        statuses = [f.status for f in findings]
        if not statuses:
            return ConsistencyStatus.NO_ASSESSMENT
        if ConsistencyStatus.CONTRADICTED_BY_EVIDENCE in statuses:
            return ConsistencyStatus.CONTRADICTED_BY_EVIDENCE
        if ConsistencyStatus.MIXED_EVIDENCE in statuses:
            return ConsistencyStatus.MIXED_EVIDENCE
        if ConsistencyStatus.CONSISTENT_WITH_EVIDENCE in statuses:
            return ConsistencyStatus.CONSISTENT_WITH_EVIDENCE
        if ConsistencyStatus.INSUFFICIENT_EVIDENCE in statuses:
            return ConsistencyStatus.INSUFFICIENT_EVIDENCE
        return ConsistencyStatus.NO_ASSESSMENT

    @staticmethod
    def _build_summary(
        primary: ConsistencyFinding,
        secondary: List[ConsistencyFinding],
        overall: ConsistencyStatus,
    ) -> str:
        parts = [f"Primary claim intent [{primary.intent.value}] is {overall.value}."]
        if secondary:
            sec_intents = ", ".join([f"[{s.intent.value}: {s.status.value}]" for s in secondary])
            parts.append(f"Secondary findings: {sec_intents}.")
        return " ".join(parts)
