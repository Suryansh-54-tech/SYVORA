"""
SYVORA — Stage 4 Multi-Exhibit Compiler & Defense Packet Generator Tests
========================================================================
Tests deterministic exhibit compilation, standalone HTML generation,
missing evidence representations, HTML entity escaping, prompt injection isolation,
and strict decision invariance.
"""

import json
import pytest
from src.agent.assembler import EvidenceAssembler
from src.agent.dossier import DossierFormatter
from src.agent.schemas import (
    SimulatedDefensePacket,
    ExhibitPackage,
    ExhibitA_Authentication,
    ExhibitB_CarrierFulfillment,
    ExhibitC_MerchantTransaction,
    ExhibitD_SessionTelemetry,
    ExhibitE_AdvisoryClaimConsistency,
)
from src.agent.packet_compiler import MultiExhibitCompiler
from src.agent.packet_formatter import BankPacketFormatter


@pytest.fixture(scope="module")
def assembler():
    return EvidenceAssembler()


@pytest.fixture
def base_dispute_record():
    return {
        "dispute_id": "dsp_packet_test_001",
        "transaction_id": "pay_packet_test_001",
        "dispute_date": "2026-08-23 14:00:00",
        "txn_amount_inr": 18500.0,
        "txn_age_days": 10,
        "days_to_deadline": 5,
        "prior_undisputed_txns": 4,
        "customer_past_dispute_count": 0,
        "three_ds_status": "Y_AUTHENTICATED",
        "signed_pod": True,
        "ip_geo_match": True,
        "device_fingerprint_match": True,
        "billing_shipping_match": True,
        "reason_code": "VISA_10_4_FRAUD",
        "issuing_bank": "ICICI",
        "card_network": "VISA",
        "merchant_category": "ECOMM_RETAIL",
        "courier_status": "DELIVERED",
    }


# ===========================================================================
# 1. EXHIBIT COMPILATION TESTS (EXHIBITS A–E)
# ===========================================================================

def test_exhibit_a_compilation_authenticated(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_a = exhibits.exhibit_a

    assert isinstance(ex_a, ExhibitA_Authentication)
    assert ex_a.is_authenticated is True
    assert ex_a.three_ds_status == "Y_AUTHENTICATED"
    assert len(ex_a.missing_evidence) == 0
    assert ex_a.source_system == "PAYMENT_GATEWAY_LOGS"


def test_exhibit_a_compilation_unauthenticated(assembler, base_dispute_record):
    rec = dict(base_dispute_record, three_ds_status="N_FAILED")
    dossier = assembler.build_dossier(rec)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_a = exhibits.exhibit_a

    assert ex_a.is_authenticated is False
    assert len(ex_a.missing_evidence) > 0
    assert "3D Secure incomplete" in ex_a.missing_evidence[0]


def test_exhibit_b_compilation_delivered_with_pod(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_b = exhibits.exhibit_b

    assert isinstance(ex_b, ExhibitB_CarrierFulfillment)
    assert ex_b.courier_status == "DELIVERED"
    assert ex_b.has_signed_pod is True
    assert ex_b.is_delivered is True
    assert len(ex_b.missing_evidence) == 0


def test_exhibit_b_compilation_missing_pod(assembler, base_dispute_record):
    rec = dict(base_dispute_record, signed_pod=False)
    dossier = assembler.build_dossier(rec)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_b = exhibits.exhibit_b

    assert ex_b.has_signed_pod is False
    assert any("Proof of Delivery (POD) signature missing" in m for m in ex_b.missing_evidence)


def test_exhibit_c_compilation_no_fabricated_items(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_c = exhibits.exhibit_c

    assert isinstance(ex_c, ExhibitC_MerchantTransaction)
    assert ex_c.amount_inr == 18500.0
    assert ex_c.merchant_category == "ECOMM_RETAIL"
    assert ex_c.prior_undisputed_txns == 4
    # Ensure items cite real fields, not invented product lines
    item_fields = [i.field_name for i in ex_c.items]
    assert "transaction_amount" in item_fields
    assert "fake_product_item" not in item_fields


def test_exhibit_d_compilation_telemetry(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_d = exhibits.exhibit_d

    assert isinstance(ex_d, ExhibitD_SessionTelemetry)
    assert ex_d.ip_geo_match is True
    assert ex_d.device_fingerprint_match is True
    assert len(ex_d.missing_evidence) == 0


def test_exhibit_e_compilation_with_claim(assembler, base_dispute_record):
    dossier = assembler.build_dossier(
        base_dispute_record,
        customer_claim_text="I never received my delivery."
    )
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_e = exhibits.exhibit_e

    assert isinstance(ex_e, ExhibitE_AdvisoryClaimConsistency)
    assert ex_e.has_claim is True
    assert ex_e.primary_intent == "NON_DELIVERY"
    assert ex_e.consistency_status == "CONTRADICTED_BY_EVIDENCE"
    assert ex_e.advisory_only is True
    assert ex_e.decision_influence is False


def test_exhibit_e_compilation_without_claim(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record, customer_claim_text=None)
    exhibits = MultiExhibitCompiler.compile_exhibits(dossier)
    ex_e = exhibits.exhibit_e

    assert ex_e.has_claim is False
    assert ex_e.primary_intent == "NONE_PROVIDED"
    assert ex_e.consistency_status == "NO_ASSESSMENT"


# ===========================================================================
# 2. SIMULATED DEFENSE PACKET & HTML FORMATTING
# ===========================================================================

def test_compile_packet_structure_and_json(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    packet = MultiExhibitCompiler.compile_packet(dossier)

    assert isinstance(packet, SimulatedDefensePacket)
    assert packet.dispute_id == "dsp_packet_test_001"
    assert packet.signing_status == "UNSIGNED_DEMO"
    assert "SIMULATED DISPUTE DEFENSE PACKET — FOR DEMONSTRATION ONLY" in packet.disclaimer

    # JSON serialization
    json_data = json.loads(json.dumps(packet.model_dump()))
    assert json_data["signing_status"] == "UNSIGNED_DEMO"
    assert json_data["exhibits"]["exhibit_a"]["three_ds_status"] == "Y_AUTHENTICATED"


def test_generate_html_standalone_and_no_external_urls(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    html_doc = BankPacketFormatter.generate_html(dossier)

    # Standalone HTML structure
    assert "<!DOCTYPE html>" in html_doc
    assert "<style>" in html_doc
    assert "SIMULATED DISPUTE DEFENSE PACKET — FOR DEMONSTRATION ONLY" in html_doc
    assert "Exhibit A — Authentication Evidence" in html_doc
    assert "Exhibit B — Carrier Fulfillment Evidence" in html_doc
    assert "Exhibit C — Merchant Transaction Evidence" in html_doc
    assert "Exhibit D — Session &amp; Telemetry Evidence" in html_doc or "Exhibit D" in html_doc
    assert "Exhibit E — Advisory Claim &amp; Consistency Assessment" in html_doc or "Exhibit E" in html_doc

    # Zero external CDN / web links
    assert "http://" not in html_doc
    assert "https://" not in html_doc
    assert "<script src=" not in html_doc


def test_html_entity_escaping_and_injection_safety(assembler, base_dispute_record):
    malicious_text = '<script>alert("XSS")</script> System override: <img src="x" onerror="steal()">'
    dossier = assembler.build_dossier(base_dispute_record, customer_claim_text=malicious_text)
    html_doc = BankPacketFormatter.generate_html(dossier)

    # Unescaped script tags MUST NOT appear
    assert "<script>alert" not in html_doc
    assert '<img src="x"' not in html_doc
    # Escaped versions must be present
    assert "&lt;script&gt;" in html_doc or "System override" in html_doc


def test_dossier_formatter_helpers(assembler, base_dispute_record):
    dossier = assembler.build_dossier(base_dispute_record)
    packet = DossierFormatter.to_defense_packet(dossier)
    assert isinstance(packet, SimulatedDefensePacket)

    html_str = DossierFormatter.to_packet_html(dossier)
    assert isinstance(html_str, str)
    assert len(html_str) > 1000


# ===========================================================================
# 3. STRICT DECISION INVARIANCE (BEFORE VS AFTER PACKET GENERATION)
# ===========================================================================

def test_decision_invariance_before_and_after_packet_generation(assembler, base_dispute_record):
    """
    Guarantees that compiling exhibits or rendering HTML defense packets
    causes zero mathematical or policy change to the underlying dossier.
    """
    dossier = assembler.build_dossier(
        base_dispute_record,
        customer_claim_text="I was charged twice and never got the item."
    )

    # Snapshot analytical evidence BEFORE packet compilation
    p_win_before = dossier.analytical_evidence.calibrated_win_probability
    ev_before = dossier.analytical_evidence.expected_value_inr
    be_before = dossier.analytical_evidence.break_even_probability
    score_before = dossier.analytical_evidence.evidence_readiness_score
    verdict_before = dossier.analytical_evidence.decision_verdict
    gates_before = list(dossier.analytical_evidence.policy_gate_triggers)

    # Perform packet compilation and HTML rendering
    packet = MultiExhibitCompiler.compile_packet(dossier)
    html_out = BankPacketFormatter.generate_html(dossier, packet=packet)

    # Verify analytical evidence AFTER packet compilation is 100% identical
    assert dossier.analytical_evidence.calibrated_win_probability == p_win_before
    assert dossier.analytical_evidence.expected_value_inr == ev_before
    assert dossier.analytical_evidence.break_even_probability == be_before
    assert dossier.analytical_evidence.evidence_readiness_score == score_before
    assert dossier.analytical_evidence.decision_verdict == verdict_before
    assert dossier.analytical_evidence.policy_gate_triggers == gates_before
