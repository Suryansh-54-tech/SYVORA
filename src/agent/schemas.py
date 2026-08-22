"""
SentinelRisk — Evidence & Dossier Pydantic Schemas
==================================================
Strictly typed data schemas for observed digital evidence, derived analytical
metrics, and structured dispute defense dossiers.

Enforces:
- Strict separation of OBSERVED vs. DERIVED evidence
- Explicit provenance (source_system, source_record_id, timestamp)
- Explicit representation of missing / incomplete evidence
- 100% JSON serializability
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


def parse_bool(val: Any, default: bool = False) -> bool:
    """
    Safely parses a boolean from arbitrary input types (bool, str, int, float, None).
    Avoids Python's bool("False") == True pitfall on non-empty strings.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val != 0)
    if isinstance(val, str):
        clean = val.strip().lower()
        if clean in ("true", "1", "yes", "y", "t", "enable", "enabled"):
            return True
        if clean in ("false", "0", "no", "n", "f", "disable", "disabled"):
            return False
    return default


def get_missing_evidence_elements(
    courier_status: str,
    signed_pod: bool,
    three_ds_status: str,
    ip_geo_match: bool,
    device_fingerprint_match: bool
) -> List[str]:
    """
    Authoritative centralized evaluator for missing evidence proofs.
    Ensures DecisionEngine and EvidenceAssembler output 100% identical checklists.
    """
    missing = []
    if courier_status != "NOT_APPLICABLE":
        if courier_status != "DELIVERED":
            missing.append(f"Courier delivery incomplete (Status: {courier_status})")
        if not signed_pod:
            missing.append("Proof of Delivery (POD) signature missing")
    if three_ds_status != "Y_AUTHENTICATED":
        missing.append(f"3D Secure incomplete (Status: {three_ds_status})")
    if not ip_geo_match:
        missing.append("Checkout IP does not match delivery destination")
    if not device_fingerprint_match:
        missing.append("Known device fingerprint match unconfirmed")
    return missing


class EvidenceSourceType(str, Enum):
    PAYMENT_GATEWAY = "PAYMENT_GATEWAY_LOGS"
    ORDER_DATABASE = "MERCHANT_ORDER_DATABASE"
    CARRIER_LOGISTICS = "CARRIER_3PL_API"
    DEVICE_TELEMETRY = "CHECKOUT_SESSION_TELEMETRY"
    CUSTOMER_ACCOUNT = "CUSTOMER_ACCOUNT_HISTORY"
    MERCHANT_POLICY = "MERCHANT_TERMS_DATABASE"


class ObservedItem(BaseModel):
    field_name: str
    value: Any
    is_available: bool = True
    source_system: EvidenceSourceType
    source_record_id: str
    timestamp: str
    notes: Optional[str] = None


class AuthenticationEvidence(BaseModel):
    three_ds_status: str
    is_authenticated: bool
    source_system: EvidenceSourceType = EvidenceSourceType.PAYMENT_GATEWAY
    source_record_id: str
    timestamp: str


class FulfillmentEvidence(BaseModel):
    courier_status: str
    carrier: str
    has_signed_pod: bool
    is_delivered: bool
    tracking_number: Optional[str] = None
    source_system: EvidenceSourceType = EvidenceSourceType.CARRIER_LOGISTICS
    source_record_id: str
    timestamp: str


class TelemetryEvidence(BaseModel):
    ip_geo_match: bool
    device_fingerprint_match: bool
    billing_shipping_match: bool
    source_system: EvidenceSourceType = EvidenceSourceType.DEVICE_TELEMETRY
    source_record_id: str
    timestamp: str


class CustomerHistoryEvidence(BaseModel):
    prior_undisputed_txns: int
    customer_past_dispute_count: int
    is_serial_disputer: bool
    is_visa_ce3_eligible: bool
    source_system: EvidenceSourceType = EvidenceSourceType.CUSTOMER_ACCOUNT
    source_record_id: str
    timestamp: str


class CustomerClaimEvidence(BaseModel):
    """
    Customer-provided free text (claim / remark) after mandatory sanitization.

    This block is DATA ONLY:
    - Explicitly marked UNTRUSTED / SANITIZED
    - Never influences ML features, win probability, Expected Value,
      CONTEST / REVIEW / SURRENDER verdicts, or evidence provenance
    - Original text and SHA-256 preserved for audit; sanitized copy is the
      only form permitted for downstream display
    """
    field_name: str = "customer_claim_text"
    trust_level: str = "UNTRUSTED"
    processing_status: str = "SANITIZED"
    decision_influence: bool = False
    original_text: str
    sanitized_text: str
    original_sha256: str
    sanitized_sha256: str
    is_threat_detected: bool
    threats_detected: List[str]


class ObservedEvidencePackage(BaseModel):
    dispute_id: str
    transaction_id: str
    dispute_date: str
    amount_inr: float
    reason_code: str
    card_network: str
    issuing_bank: str
    merchant_category: str
    days_to_deadline: int

    # Sub-packages
    authentication: AuthenticationEvidence
    fulfillment: FulfillmentEvidence
    telemetry: TelemetryEvidence
    customer_history: CustomerHistoryEvidence

    # Inventory of all observed facts with provenance
    raw_evidence_inventory: List[ObservedItem]
    missing_evidence_elements: List[str]

    # Untrusted customer-provided text (optional; sanitized before storage).
    # Kept OUTSIDE raw_evidence_inventory by design: it carries no system provenance.
    customer_claim: Optional[CustomerClaimEvidence] = None


class AnalyticalEvidencePackage(BaseModel):
    calibrated_win_probability: float
    break_even_probability: float
    expected_value_inr: float
    is_positive_ev: bool
    arbitration_fee_inr: float
    evidence_readiness_score: int
    decision_verdict: str
    decision_reasons: List[str]
    policy_gate_triggers: List[str]
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]
    shap_summary_text: str


class DisputeDefenseDossier(BaseModel):
    dossier_id: str
    dispute_id: str
    generated_at: str
    observed_evidence: ObservedEvidencePackage
    analytical_evidence: AnalyticalEvidencePackage
    rebuttal_narrative_markdown: str
    is_ready_for_submission: bool
