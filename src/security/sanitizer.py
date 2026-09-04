"""
NYAYANTRA — Defensive Input Sanitizer & Prompt Injection Firewall
==============================================================
Sanitizes untrusted customer remarks, transaction memos, and claim text
before ingestion by downstream analytical or dossier-generation layers.

Defends against:
- Prompt injection & instruction hijacking (e.g. "Ignore previous instructions", "System override")
- Delimiter break-outs (e.g. "```", "<system>", "[INST]", "<|im_start|>")
- Malicious control characters (null bytes, ANSI escape codes, BiDi overrides)
- Data exfiltration payloads (URLs, Webhooks, API key scraping attempts)

Guarantees:
- Original source evidence is preserved untouched in parallel
- Sanitized text is clearly tagged with detected threat signatures
- Pure deterministic processing — zero external network calls
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field


class SanitizedTextResult(BaseModel):
    original_text: str
    sanitized_text: str
    is_threat_detected: bool
    threats_detected: List[str]
    original_sha256: str
    sanitized_sha256: str


# ---------------------------------------------------------------------------
# Threat Pattern Signatures
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    # 1. System Overrides & Instruction Hijacking
    (r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b", "SYSTEM_INSTRUCTION_OVERRIDE"),
    (r"(?i)\bsystem\s+override\b", "SYSTEM_OVERRIDE_KEYWORD"),
    (r"(?i)\byou\s+are\s+now\s+(a|an|the|DAN|jailbroken)\b", "PERSONA_HIJACK_ATTEMPT"),
    (r"(?i)\bdisregard\s+(all\s+)?rules\b", "RULE_DISREGARD_ATTEMPT"),
    (r"(?i)\b(act\s+as|pretend\s+to\s+be)\s+a\s+(developer|admin|bank)\b", "ROLEPLAY_PRIVILEGE_ESCALATION"),
    (r"(?i)\bauto[- ]?approve\s+(full\s+)?refund\b", "FINANCIAL_ACTION_FORGERY"),

    # 2. Prompt Delimiter Break-Outs
    (r"```[a-zA-Z]*", "CODE_BLOCK_DELIMITER_INJECTION"),
    (r"(?i)</?(system|sys|instruction|INST|user|assistant)>", "SYSTEM_TAG_DELIMITER_INJECTION"),
    (r"<\|im_start\|>|<\|im_end\|>", "CHATML_TOKEN_INJECTION"),
    (r"---BEGIN(\s+SYSTEM\s+PROMPT)?---", "HEADER_DELIMITER_INJECTION"),

    # 3. Data Exfiltration & Webhooks
    (r"(?i)https?://[^\s<>\"']+", "EXTERNAL_URL_INJECTION"),
    (r"(?i)\b(curl|wget|fetch|xmlhttprequest)\b", "COMMAND_EXECUTION_SYNTAX"),
    (r"(?i)\b(api_key|password|secret|bearer\s+[a-zA-Z0-9_\-\.]+)\b", "CREDENTIAL_PROBE_PATTERN"),
]

# Control character regex (null bytes, ANSI escape, BiDi overrides)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f\u202a-\u202e\u2066-\u2069]")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class InputSanitizer:
    """
    Deterministic input sanitizer guarding LLM and dossier templates
    from adversarial customer claim injection attacks.
    """

    def __init__(self):
        self.compiled_threats = [(re.compile(pattern), tag) for pattern, tag in INJECTION_PATTERNS]

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Computes SHA-256 hash of string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sanitize_claim_text(self, text: Optional[str]) -> SanitizedTextResult:
        """
        Sanitizes untrusted customer dispute remarks / claim text.

        Preserves original text separately and returns detailed threat analysis.
        """
        if text is None or not isinstance(text, str):
            orig = ""
            return SanitizedTextResult(
                original_text=orig,
                sanitized_text="",
                is_threat_detected=False,
                threats_detected=[],
                original_sha256=self._compute_hash(orig),
                sanitized_sha256=self._compute_hash(orig),
            )

        original_text = text
        working_text = text
        threats_found: List[str] = []

        # 1. Strip ANSI escape sequences
        if ANSI_ESCAPE_PATTERN.search(working_text):
            threats_found.append("ANSI_ESCAPE_SEQUENCE_STRIPPED")
            working_text = ANSI_ESCAPE_PATTERN.sub("", working_text)

        # 2. Strip malicious control characters & BiDi overrides
        if CONTROL_CHAR_PATTERN.search(working_text):
            threats_found.append("MALICIOUS_CONTROL_CHARACTERS_STRIPPED")
            working_text = CONTROL_CHAR_PATTERN.sub("", working_text)

        # 3. Detect and neutralize prompt injection signatures
        for pattern, threat_tag in self.compiled_threats:
            if pattern.search(working_text):
                threats_found.append(threat_tag)
                working_text = pattern.sub(f"[FILTERED_{threat_tag}]", working_text)

        # 4. Normalize whitespace
        working_text = " ".join(working_text.split())

        # 5. Wrap inside isolated data boundary tags
        is_threat = len(threats_found) > 0
        final_sanitized = f"<untrusted_customer_claim is_sanitized='{is_threat}'>{working_text}</untrusted_customer_claim>"

        return SanitizedTextResult(
            original_text=original_text,
            sanitized_text=final_sanitized,
            is_threat_detected=is_threat,
            threats_detected=list(set(threats_found)),
            original_sha256=self._compute_hash(original_text),
            sanitized_sha256=self._compute_hash(final_sanitized),
        )
