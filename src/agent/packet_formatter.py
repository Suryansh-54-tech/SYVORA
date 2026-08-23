"""
SYVORA — Standalone Defense Packet HTML Formatter (Stage 4)
============================================================
Generates self-contained, publication-grade, print-optimized HTML defense packets.

Guarantees & Constraints:
- Pure standard library (html.escape, json, etc.).
- Embedded CSS only — zero external CDNs, fonts, or tracking scripts.
- Every dynamic field is strictly escaped via html.escape() to guarantee XSS safety.
- Embeds prominent "SIMULATED DISPUTE DEFENSE PACKET — FOR DEMONSTRATION ONLY" disclaimers.
- Downstream presentational output only; zero access or influence over ML/decision math.
"""

import html
from typing import Optional
from src.agent.schemas import DisputeDefenseDossier, SimulatedDefensePacket
from src.agent.packet_compiler import MultiExhibitCompiler


class BankPacketFormatter:
    """
    Renders standalone, print-ready HTML defense packets from compiled dispute dossiers.
    """

    @classmethod
    def generate_html(
        cls,
        dossier: DisputeDefenseDossier,
        packet: Optional[SimulatedDefensePacket] = None,
        audit_hash: Optional[str] = None,
        signing_status: str = "UNSIGNED_DEMO",
    ) -> str:
        """
        Renders a complete, standalone, single-file HTML document for the defense packet.
        """
        if packet is None:
            packet = MultiExhibitCompiler.compile_packet(
                dossier, audit_hash=audit_hash, signing_status=signing_status
            )

        obs = dossier.observed_evidence
        ana = dossier.analytical_evidence
        ex = packet.exhibits

        # Helpers for escaping
        e = html.escape

        # Status badge styles
        verdict_color = "#10B981" if ana.decision_verdict == "CONTEST" else ("#F59E0B" if ana.decision_verdict == "REVIEW" else "#EF4444")

        css = """
        <style>
            @page {
                size: A4 portrait;
                margin: 15mm 15mm 15mm 15mm;
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            body {
                background-color: #0F172A;
                color: #F8FAFC;
                line-height: 1.5;
                font-size: 13px;
                padding: 24px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 32px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            }
            .watermark-banner {
                background: linear-gradient(90deg, #B91C1C, #991B1B);
                color: #FFFFFF;
                text-align: center;
                padding: 10px 16px;
                font-weight: 800;
                font-size: 11px;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                border-radius: 6px;
                margin-bottom: 24px;
                border: 1px solid #DC2626;
            }
            .header-table {
                width: 100%;
                border-bottom: 2px solid #334155;
                padding-bottom: 16px;
                margin-bottom: 20px;
            }
            .brand-title {
                font-size: 22px;
                font-weight: 800;
                color: #38BDF8;
                letter-spacing: -0.02em;
            }
            .brand-sub {
                font-size: 12px;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .packet-meta {
                text-align: right;
                font-family: "SF Mono", Consolas, monospace;
                font-size: 11px;
                color: #CBD5E1;
            }
            .section {
                margin-bottom: 24px;
            }
            .section-title {
                font-size: 14px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #38BDF8;
                border-bottom: 1px solid #334155;
                padding-bottom: 6px;
                margin-bottom: 12px;
            }
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
            }
            .grid-3 {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 12px;
            }
            .card {
                background-color: #0F172A;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 12px 14px;
            }
            .card-label {
                font-size: 10px;
                text-transform: uppercase;
                color: #94A3B8;
                letter-spacing: 0.04em;
                margin-bottom: 4px;
            }
            .card-val {
                font-size: 13px;
                font-weight: 600;
                color: #F8FAFC;
                font-family: "SF Mono", Consolas, monospace;
            }
            .card-val-lg {
                font-size: 16px;
                font-weight: 700;
            }
            .exhibit-box {
                background-color: #0F172A;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-bottom: 16px;
                overflow: hidden;
            }
            .exhibit-header {
                background-color: #1E293B;
                border-bottom: 1px solid #334155;
                padding: 10px 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .exhibit-title {
                font-size: 12px;
                font-weight: 700;
                color: #F8FAFC;
                text-transform: uppercase;
            }
            .exhibit-badge {
                font-size: 10px;
                font-weight: 700;
                padding: 2px 8px;
                border-radius: 4px;
                font-family: monospace;
            }
            .badge-verified {
                background-color: rgba(16, 185, 129, 0.2);
                color: #34D399;
                border: 1px solid rgba(16, 185, 129, 0.4);
            }
            .badge-missing {
                background-color: rgba(239, 68, 68, 0.2);
                color: #F87171;
                border: 1px solid rgba(239, 68, 68, 0.4);
            }
            .badge-advisory {
                background-color: rgba(56, 189, 248, 0.2);
                color: #38BDF8;
                border: 1px solid rgba(56, 189, 248, 0.4);
            }
            .exhibit-body {
                padding: 14px;
            }
            .table-exhibit {
                width: 100%;
                border-collapse: collapse;
                font-size: 11px;
                margin-top: 8px;
            }
            .table-exhibit th {
                text-align: left;
                padding: 6px 8px;
                background-color: #1E293B;
                color: #94A3B8;
                font-weight: 600;
                text-transform: uppercase;
                border-bottom: 1px solid #334155;
            }
            .table-exhibit td {
                padding: 6px 8px;
                border-bottom: 1px solid #1E293B;
                color: #CBD5E1;
            }
            .table-exhibit tr:last-child td {
                border-bottom: none;
            }
            .missing-banner {
                background-color: rgba(239, 68, 68, 0.1);
                border-left: 3px solid #EF4444;
                padding: 6px 10px;
                font-size: 11px;
                color: #FCA5A5;
                margin-top: 8px;
            }
            .rebuttal-narrative {
                background-color: #0F172A;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 14px;
                font-family: monospace;
                font-size: 11px;
                color: #CBD5E1;
                white-space: pre-wrap;
                max-height: 280px;
                overflow-y: auto;
            }
            .footer-box {
                margin-top: 24px;
                border-top: 1px solid #334155;
                padding-top: 14px;
                font-size: 10px;
                color: #64748B;
                text-align: center;
            }
            @media print {
                body {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                    padding: 0;
                }
                .container {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                    border: none !important;
                    box-shadow: none !important;
                    padding: 0 !important;
                    max-width: 100% !important;
                }
                .card, .exhibit-box, .rebuttal-narrative {
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                    border: 1px solid #CCCCCC !important;
                }
                .exhibit-header {
                    background-color: #F1F5F9 !important;
                    border-bottom: 1px solid #CCCCCC !important;
                }
                .brand-title, .section-title {
                    color: #0F172A !important;
                }
                .card-label, .table-exhibit th, .brand-sub, .packet-meta {
                    color: #475569 !important;
                }
                .table-exhibit td, .card-val {
                    color: #0F172A !important;
                }
                .watermark-banner {
                    background: #FEE2E2 !important;
                    color: #991B1B !important;
                    border: 1px solid #EF4444 !important;
                }
                .page-break {
                    page-break-before: always;
                }
            }
        </style>
        """

        # Build Exhibit A Rows
        ex_a_rows = "".join([
            f"<tr><td><code>{e(item.field_name)}</code></td><td><strong>{e(item.value_display)}</strong></td><td>{e(item.source_system)}</td><td><code>{e(item.source_record_id)}</code></td></tr>"
            for item in ex.exhibit_a.items
        ])
        ex_a_missing = "".join([f"<div class='missing-banner'>● [MISSING EVIDENCE] {e(m)}</div>" for m in ex.exhibit_a.missing_evidence])

        # Build Exhibit B Rows
        ex_b_rows = "".join([
            f"<tr><td><code>{e(item.field_name)}</code></td><td><strong>{e(item.value_display)}</strong></td><td>{e(item.source_system)}</td><td><code>{e(item.source_record_id)}</code></td></tr>"
            for item in ex.exhibit_b.items
        ])
        ex_b_missing = "".join([f"<div class='missing-banner'>● [MISSING EVIDENCE] {e(m)}</div>" for m in ex.exhibit_b.missing_evidence])

        # Build Exhibit C Rows
        ex_c_rows = "".join([
            f"<tr><td><code>{e(item.field_name)}</code></td><td><strong>{e(item.value_display)}</strong></td><td>{e(item.source_system)}</td><td><code>{e(item.source_record_id)}</code></td></tr>"
            for item in ex.exhibit_c.items
        ])

        # Build Exhibit D Rows
        ex_d_rows = "".join([
            f"<tr><td><code>{e(item.field_name)}</code></td><td><strong>{e(item.value_display)}</strong></td><td>{e(item.source_system)}</td><td><code>{e(item.source_record_id)}</code></td></tr>"
            for item in ex.exhibit_d.items
        ])
        ex_d_missing = "".join([f"<div class='missing-banner'>● [MISSING EVIDENCE] {e(m)}</div>" for m in ex.exhibit_d.missing_evidence])

        # Exhibit E Secondary list
        sec_display = ", ".join([e(s) for s in ex.exhibit_e.secondary_intents]) if ex.exhibit_e.secondary_intents else "None"

        body_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SYVORA Simulated Defense Packet — {e(dossier.dispute_id)}</title>
            {css}
        </head>
        <body>
            <div class="container">
                <div class="watermark-banner">
                    {e(packet.disclaimer)}
                </div>

                <table class="header-table">
                    <tr>
                        <td>
                            <div class="brand-title">SYVORA</div>
                            <div class="brand-sub">Payment Dispute Intelligence &bull; Demonstration Rebuttal Packet</div>
                        </td>
                        <td class="packet-meta">
                            <div>Packet ID: <strong>{e(packet.packet_id)}</strong></div>
                            <div>Dispute Ref: <strong>{e(dossier.dispute_id)}</strong></div>
                            <div>Compiled: {e(packet.generated_at)}</div>
                        </td>
                    </tr>
                </table>

                <!-- Section 1: Case Summary & Economics -->
                <div class="section">
                    <div class="section-title">1. Case Overview & Decision Economics</div>
                    <div class="grid-3" style="margin-bottom: 12px;">
                        <div class="card">
                            <div class="card-label">Transaction Amount</div>
                            <div class="card-val card-val-lg">INR {obs.amount_inr:,.2f}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Reason Code & Issuer</div>
                            <div class="card-val">{e(obs.reason_code)} ({e(obs.issuing_bank)})</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Autonomous Verdict</div>
                            <div class="card-val card-val-lg" style="color: {verdict_color};">{e(ana.decision_verdict)}</div>
                        </div>
                    </div>

                    <div class="grid-3">
                        <div class="card">
                            <div class="card-label">Win Probability P(Win)</div>
                            <div class="card-val">{ana.calibrated_win_probability:.1%}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Expected Value E[EV]</div>
                            <div class="card-val">+INR {ana.expected_value_inr:,.2f}</div>
                        </div>
                        <div class="card">
                            <div class="card-label">Evidence Readiness Index</div>
                            <div class="card-val">{ana.evidence_readiness_score} / 100</div>
                        </div>
                    </div>
                </div>

                <!-- Section 2: Evidentiary Exhibits -->
                <div class="section page-break">
                    <div class="section-title">2. Verified Evidentiary Exhibits</div>

                    <!-- Exhibit A -->
                    <div class="exhibit-box">
                        <div class="exhibit-header">
                            <span class="exhibit-title">{e(ex.exhibit_a.title)}</span>
                            <span class="exhibit-badge {'badge-verified' if ex.exhibit_a.is_authenticated else 'badge-missing'}">
                                {'3DS AUTHENTICATED' if ex.exhibit_a.is_authenticated else 'UNAUTHENTICATED'}
                            </span>
                        </div>
                        <div class="exhibit-body">
                            <table class="table-exhibit">
                                <thead>
                                    <tr><th>Evidence Field</th><th>Observed Value</th><th>Source System</th><th>Record Reference</th></tr>
                                </thead>
                                <tbody>
                                    {ex_a_rows}
                                </tbody>
                            </table>
                            {ex_a_missing}
                        </div>
                    </div>

                    <!-- Exhibit B -->
                    <div class="exhibit-box">
                        <div class="exhibit-header">
                            <span class="exhibit-title">{e(ex.exhibit_b.title)}</span>
                            <span class="exhibit-badge {'badge-verified' if ex.exhibit_b.has_signed_pod else 'badge-missing'}">
                                {'POD SIGNED' if ex.exhibit_b.has_signed_pod else 'POD MISSING'}
                            </span>
                        </div>
                        <div class="exhibit-body">
                            <table class="table-exhibit">
                                <thead>
                                    <tr><th>Evidence Field</th><th>Observed Value</th><th>Source System</th><th>Record Reference</th></tr>
                                </thead>
                                <tbody>
                                    {ex_b_rows}
                                </tbody>
                            </table>
                            {ex_b_missing}
                        </div>
                    </div>

                    <!-- Exhibit C -->
                    <div class="exhibit-box">
                        <div class="exhibit-header">
                            <span class="exhibit-title">{e(ex.exhibit_c.title)}</span>
                            <span class="exhibit-badge badge-verified">MERCHANT RECORDS</span>
                        </div>
                        <div class="exhibit-body">
                            <table class="table-exhibit">
                                <thead>
                                    <tr><th>Evidence Field</th><th>Observed Value</th><th>Source System</th><th>Record Reference</th></tr>
                                </thead>
                                <tbody>
                                    {ex_c_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Exhibit D -->
                    <div class="exhibit-box">
                        <div class="exhibit-header">
                            <span class="exhibit-title">{e(ex.exhibit_d.title)}</span>
                            <span class="exhibit-badge {'badge-verified' if ex.exhibit_d.ip_geo_match else 'badge-missing'}">
                                {'TELEMETRY MATCH' if ex.exhibit_d.ip_geo_match else 'MISMATCH'}
                            </span>
                        </div>
                        <div class="exhibit-body">
                            <table class="table-exhibit">
                                <thead>
                                    <tr><th>Evidence Field</th><th>Observed Value</th><th>Source System</th><th>Record Reference</th></tr>
                                </thead>
                                <tbody>
                                    {ex_d_rows}
                                </tbody>
                            </table>
                            {ex_d_missing}
                        </div>
                    </div>

                    <!-- Exhibit E -->
                    <div class="exhibit-box">
                        <div class="exhibit-header">
                            <span class="exhibit-title">{e(ex.exhibit_e.title)}</span>
                            <span class="exhibit-badge badge-advisory">ADVISORY ONLY &bull; ZERO DECISION INFLUENCE</span>
                        </div>
                        <div class="exhibit-body">
                            <div class="grid-2" style="margin-bottom: 8px;">
                                <div>
                                    <div class="card-label">Primary Claim Classification</div>
                                    <div class="card-val">{e(ex.exhibit_e.primary_intent)}</div>
                                </div>
                                <div>
                                    <div class="card-label">Consistency Status</div>
                                    <div class="card-val">{e(ex.exhibit_e.consistency_status)}</div>
                                </div>
                            </div>
                            <div style="font-size: 11px; margin-bottom: 6px;">
                                <strong>Sanitized Customer Remarks:</strong> <em>"{e(ex.exhibit_e.sanitized_claim_text)}"</em>
                            </div>
                            <div style="font-size: 11px; color: #94A3B8;">
                                <strong>Advisory Explanation:</strong> {e(ex.exhibit_e.advisory_explanation)}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Section 3: Rebuttal Narrative -->
                <div class="section page-break">
                    <div class="section-title">3. Formal Rebuttal Narrative Markdown</div>
                    <div class="rebuttal-narrative">{e(packet.rebuttal_markdown)}</div>
                </div>

                <!-- Section 4: Cryptographic Provenance -->
                <div class="section">
                    <div class="section-title">4. Cryptographic Provenance & Verification</div>
                    <div class="card">
                        <div class="grid-2">
                            <div>
                                <div class="card-label">Cryptographic Provenance Hash</div>
                                <div class="card-val" style="font-size: 11px;">{e(packet.audit_hash)}</div>
                            </div>
                            <div>
                                <div class="card-label">Ledger Signing Status</div>
                                <div class="card-val" style="font-size: 11px; color: #38BDF8;">{e(packet.signing_status)}</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="footer-box">
                    <div>{e(packet.disclaimer)}</div>
                    <div style="margin-top: 4px;">Compiled deterministically by SYVORA Evidence Engine. Simulated provenance trace. Zero live bank network calls executed.</div>
                </div>
            </div>
        </body>
        </html>
        """

        return body_html
