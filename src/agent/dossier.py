"""
SYVORA — Structured Dossier Generator & Rebuttal Formatter
===========================================================
Formats structured dispute defense dossiers into standardized, network-compliant
rebuttal documentation suitable for bank adjudication, PDF export, and dashboard display.

Strictly deterministic — formats only observed and analytical facts with full source citations.
"""

import os
import sys
import json
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.agent.schemas import (
    ObservedEvidencePackage,
    AnalyticalEvidencePackage,
    DisputeDefenseDossier,
    ClaimSignalPackage,
    ConsistencyEvaluation,
)


class DossierFormatter:
    """
    Standardized rebuttal documentation generator adhering to global card network
    evidence formats (Visa / Mastercard dispute standards).
    """

    @staticmethod
    def generate_rebuttal_markdown(
        observed: ObservedEvidencePackage,
        analytical: AnalyticalEvidencePackage,
        claim_understanding: Optional[ClaimSignalPackage] = None,
        consistency_eval: Optional[ConsistencyEvaluation] = None,
    ) -> str:
        """
        Generates a formal, structured Markdown defense document with explicit
        source provenance citations for every factual claim.
        """
        lines = []

        # Document Header
        lines.append(f"# Dispute Defense Dossier -- Reference #{observed.dispute_id}")
        lines.append(f"**Target Issuer:** {observed.issuing_bank} (Simulated Demonstration)")
        lines.append(f"**Card Brand:** {observed.card_network} | **Reason Code:** `{observed.reason_code}`")
        lines.append(f"**Dispute Amount:** INR {observed.amount_inr:,.2f} | **Evidence Deadline:** {observed.days_to_deadline} days remaining")
        lines.append("")
        lines.append("> **SIMULATION & DEMONSTRATION ARTIFACT ONLY**")
        lines.append("> *This document is a technical architecture demonstration compiled from synthetic dispute data, simulated 3DS telemetry, and simulated 3PL carrier logs. No live Razorpay, card brand (Visa/Mastercard), or banking network was queried. Not a live bank submission.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 1: Executive Summary & Recommendation
        lines.append("## 1. Executive Summary & Merchant Contesting Position")
        lines.append(f"- **Merchant Position:** `{analytical.decision_verdict}`")
        lines.append(f"- **Calibrated Win Probability:** `{analytical.calibrated_win_probability:.1%}` (Break-Even Threshold: `{analytical.break_even_probability:.1%}`)")
        lines.append(f"- **Expected Financial Recovery:** `+INR {analytical.expected_value_inr:,.2f}`")
        lines.append(f"- **Evidence Readiness Index:** `{analytical.evidence_readiness_score}/100`")
        lines.append("")
        lines.append("**Decision Rationales:**")
        for reason in analytical.decision_reasons:
            lines.append(f"- {reason}")
        lines.append("")

        # Section 2: Strong Customer Authentication & Session Telemetry
        lines.append("## 2. Strong Customer Authentication (3DS) & Telemetry Records (Simulated)")
        auth = observed.authentication
        telem = observed.telemetry
        lines.append(f"| Telemetry Parameter | Observed Status | Simulated Source Record ID |")
        lines.append(f"| :--- | :--- | :--- |")
        lines.append(f"| **3D Secure Protocol** | `{auth.three_ds_status}` ({'AUTHENTICATED' if auth.is_authenticated else 'UNAUTHENTICATED'}) | `{auth.source_record_id}` |")
        lines.append(f"| **IP Geo-Location Match** | `{'MATCHED' if telem.ip_geo_match else 'MISMATCH'}` | `{telem.source_record_id}` |")
        lines.append(f"| **Device Fingerprint** | `{'MATCHED' if telem.device_fingerprint_match else 'UNCONFIRMED'}` | `{telem.source_record_id}` |")
        lines.append(f"| **Billing / Shipping Address** | `{'MATCHED' if telem.billing_shipping_match else 'DIFFERENT'}` | `{telem.source_record_id}` |")
        lines.append("")

        # Section 3: Physical Fulfillment & Courier Proof of Delivery
        lines.append("## 3. Order Fulfillment & Carrier Delivery Records (Simulated)")
        ful = observed.fulfillment
        if ful.courier_status == "NOT_APPLICABLE":
            lines.append("- **Fulfillment Type:** Digital Goods / SaaS Subscription (Fulfillment modeled via account activation log).")
        else:
            lines.append(f"| Logistics Parameter | Observed Status | Simulated Tracking Reference |")
            lines.append(f"| :--- | :--- | :--- |")
            lines.append(f"| **Carrier Name** | `{ful.carrier}` | `{ful.source_record_id}` |")
            lines.append(f"| **Tracking Number** | `{ful.tracking_number or 'N/A'}` | `{ful.source_record_id}` |")
            lines.append(f"| **Delivery Status** | `{ful.courier_status}` | `{ful.source_record_id}` |")
            lines.append(f"| **Signed POD Captured** | `{'YES (Simulated Signature Attached)' if ful.has_signed_pod else 'NO (Unsigned Delivery)'}` | `{ful.source_record_id}` |")
        lines.append("")

        # Section 4: Customer Relationship & Visa CE3.0 Certification
        lines.append("## 4. Prior Relationship & Card Network Compliance Modeling")
        cust = observed.customer_history
        lines.append(f"- **Prior Undisputed Customer Transactions:** `{cust.prior_undisputed_txns}` orders")
        lines.append(f"- **Customer Historical Dispute Count:** `{cust.customer_past_dispute_count}` past chargebacks")
        
        if cust.is_visa_ce3_eligible:
            lines.append("> **[!] Visa Compelling Evidence 3.0 (CE3.0) Liability Modeling:**")
            lines.append("> Simulated customer profile contains two (2) or more prior undisputed transactions matching checkout IP/Device attributes. Under Visa CE3.0 rules, such criteria model liability shifting to the cardholder issuing bank.")
        elif cust.is_serial_disputer:
            lines.append(f"> **[!] Serial Disputer Alert:** Customer account shows `{cust.customer_past_dispute_count}` past disputes across merchant records, indicating potential friendly-fraud pattern.")
        lines.append("")

        # Section 5: Machine Learning Explainability & Feature Attributions
        lines.append("## 5. Machine Learning Explainability (TreeSHAP Attributions)")
        lines.append(f"**Analytical Summary:** {analytical.shap_summary_text}")
        lines.append("")
        if analytical.top_positive_factors:
            lines.append("**Top Positive Evidentiary Drivers:**")
            for f in analytical.top_positive_factors:
                lines.append(f"- `+{f.get('shap_impact', 0):.1%}` — {f.get('display_name', f.get('feature'))}")
        if analytical.top_negative_factors:
            lines.append("")
            lines.append("**Top Evidentiary Risk Drivers:**")
            for f in analytical.top_negative_factors:
                lines.append(f"- `{f.get('shap_impact', 0):.1%}` — {f.get('display_name', f.get('feature'))}")
        lines.append("")

        # Section 6: Missing Evidence Audit Checklist
        lines.append("## 6. Evidentiary Completeness Audit")
        if observed.missing_evidence_elements:
            lines.append("The following evidentiary elements were **not available** at the time of dossier compilation:")
            for m in observed.missing_evidence_elements:
                lines.append(f"- `[MISSING]` {m}")
        else:
            lines.append("- `[COMPLETE]` All primary evidentiary items (3DS, Carrier POD, IP Telemetry) present in simulated record.")
        lines.append("")

        # Section 7: Untrusted Customer Claim Attachment (optional; sanitized only)
        claim = observed.customer_claim
        if claim is not None:
            lines.append("## 7. Customer-Provided Claim Attachment (UNTRUSTED / SANITIZED)")
            lines.append("")
            lines.append("> **Trust Classification:** `UNTRUSTED` | **Processing Status:** `SANITIZED` | **Decision Influence:** `NONE`")
            lines.append("")
            lines.append("- **Original Text SHA-256:** `" + claim.original_sha256 + "`")
            lines.append("- **Sanitized Text SHA-256:** `" + claim.sanitized_sha256 + "`")
            threat_list = ", ".join(claim.threats_detected) if claim.threats_detected else "None"
            lines.append(f"- **Threat Categories Detected:** `{threat_list}`")
            lines.append("")
            lines.append("**Sanitized claim content (data only — excluded from ML scoring, Expected Value, verdict policy, and evidence provenance):**")
            lines.append("")
            lines.append("```text")
            lines.append(claim.sanitized_text)
            lines.append("```")
            lines.append("")
            lines.append("*This attachment is retained for human reviewer context only. The original raw text is preserved exclusively in the structured JSON record for audit purposes.*")
            lines.append("")

        # Section 8: Customer Claim Understanding (Advisory Only)
        if claim_understanding is not None and claim_understanding.has_structured_claim:
            lines.append("## 8. Customer Claim Understanding — Advisory Only")
            lines.append("")
            lines.append("> **Advisory Only:** `TRUE` | **Decision Influence:** `NONE` | **Provenance Source:** `Sanitized Customer Text`")
            lines.append("")
            lines.append(f"- **Primary Claim Classification:** `{claim_understanding.primary_intent.value}`")
            secondary_str = ", ".join([s.value for s in claim_understanding.secondary_intents]) if claim_understanding.secondary_intents else "None"
            lines.append(f"- **Secondary Claim Classifications:** `{secondary_str}`")
            lines.append(f"- **Source Sanitized SHA-256:** `{claim_understanding.source_sanitized_sha256}`")
            lines.append("- **Extracted Advisory Signals:**")
            for sig in claim_understanding.signals:
                kw_str = ", ".join(sig.matched_keywords) if sig.matched_keywords else "N/A"
                lines.append(f"  - `[{sig.intent.value}]` Rule-Matching Confidence: `{sig.confidence_score:.0%}` | Matched Patterns: `{kw_str}`")
            lines.append("")
            lines.append("*Note: Customer claim understanding signals are deterministic heuristic extractions provided for operator advisory context only. They do not constitute observed evidentiary records and have zero mathematical weight in P(Win), Expected Value, or autonomous defense verdicts.*")
            lines.append("")

        # Section 9: Customer Claim–Evidence Consistency (Advisory Only)
        if consistency_eval is not None and consistency_eval.overall_status.value != "NO_ASSESSMENT":
            lines.append("## 9. Customer Claim–Evidence Consistency — Advisory Only")
            lines.append("")
            lines.append("> **Advisory Only:** `TRUE` | **Decision Influence:** `NONE` | **Consistency Status:** `" + consistency_eval.overall_status.value + "`")
            lines.append("")
            lines.append(f"**Consistency Analysis Summary:** {consistency_eval.summary_text}")
            lines.append("")
            if consistency_eval.primary_finding is not None:
                pf = consistency_eval.primary_finding
                lines.append(f"**Primary Claim Finding (`{pf.intent.value}`):**")
                lines.append(f"- **Evaluated Status:** `{pf.status.value}`")
                lines.append(f"- **Rationale:** {pf.explanation}")
                if pf.evidence_signals:
                    lines.append("- **Verified Evidence Fields Considered:**")
                    for es in pf.evidence_signals:
                        lines.append(f"  - `{es.field_name} = {es.value}` ({es.source_system} · `{es.source_record_id}`)")
                lines.append("")
            if consistency_eval.secondary_findings:
                lines.append("**Secondary Claim Findings:**")
                for sf in consistency_eval.secondary_findings:
                    lines.append(f"- `[{sf.intent.value}]` Status: `{sf.status.value}` — {sf.explanation}")
                lines.append("")
            lines.append("*Note: Consistency evaluation provides deterministic cross-referencing between customer assertions and verified system records for human reviewer convenience. It does not constitute legal adjudication, proof of fraud, or autonomous evidence rejection.*")
            lines.append("")

        # Footer Certification
        lines.append("---")
        lines.append("*Compiled deterministically by SYVORA Evidence Engine. Simulated audit trace with deterministic provenance IDs modeled on enterprise payment gateway, logistics, and checkout telemetry standards. Zero live network calls executed.*")

        return "\n".join(lines)

    @staticmethod
    def to_dict(dossier: DisputeDefenseDossier) -> Dict[str, Any]:
        """Serializes dossier to Python dictionary."""
        return dossier.model_dump()

    @staticmethod
    def to_json(dossier: DisputeDefenseDossier, indent: int = 2) -> str:
        """Serializes dossier to formatted JSON string."""
        return json.dumps(dossier.model_dump(), indent=indent)
