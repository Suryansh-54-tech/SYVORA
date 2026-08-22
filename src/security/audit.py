"""
SentinelRisk — Cryptographic Audit Ledger
=========================================
Implements an append-only, tamper-evident SHA-256 hash-chained audit ledger
with optional HMAC-SHA256 cryptographic signing for dispute triage events,
evidence snapshots, and decision records.

Mathematical Hash-Chaining & Authentication:
    Payload Hash : H_payload = SHA-256( CanonicalJson(payload) )
    Entry Hash   : H_entry   = SHA-256( H_prev || H_payload || timestamp || entry_id || event_type || dispute_id )
    HMAC Sig     : Sig_entry = HMAC-SHA256( SecretKey, H_entry ) [Optional / Configurable]

Security Posture:
    - UNSIGNED_DEMO Mode : Detects payload tampering, deletion, and reordering.
    - HMAC_SHA256 Mode   : Detects all the above PLUS prevents full-history ledger forgery
                           by requiring knowledge of the application secret key.
"""

import os
import sys
import json
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

GENESIS_HASH = "0" * 64


class AuditEntry(BaseModel):
    entry_id: int
    dispute_id: str
    event_type: str
    timestamp: str
    previous_hash: str
    payload_hash: str
    current_hash: str
    signature_mode: str = Field(default="UNSIGNED_DEMO")
    signature: Optional[str] = Field(default=None)
    payload: Dict[str, Any]


class AuditLedger:
    """
    Append-only cryptographic ledger storing chained dispute events
    with optional HMAC-SHA256 signature authentication.
    """

    def __init__(self, ledger_file: Optional[str] = None, secret_key: Optional[str] = None):
        self.ledger_file = ledger_file or getattr(config, "AUDIT_LEDGER_PATH", os.path.join(config.PROJECT_ROOT, "data", "audit_ledger.jsonl"))
        self.secret_key = secret_key if secret_key is not None else getattr(config, "AUDIT_SECRET_KEY", None)
        self.is_signed_mode = bool(self.secret_key and len(self.secret_key.strip()) > 0)
        self.signing_mode = "HMAC_SHA256" if self.is_signed_mode else "UNSIGNED_DEMO"
        self.entries: List[AuditEntry] = []
        self._load_ledger()

    def _canonical_hash(self, data: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 hash of canonical sorted JSON payload."""
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def _compute_entry_hash(
        self,
        entry_id: int,
        previous_hash: str,
        payload_hash: str,
        timestamp: str,
        event_type: str,
        dispute_id: str
    ) -> str:
        """
        Chained hash computation:
        SHA-256( previous_hash || payload_hash || timestamp || entry_id || event_type || dispute_id )
        """
        combined = f"{previous_hash}:{payload_hash}:{timestamp}:{entry_id}:{event_type}:{dispute_id}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _compute_hmac(self, current_hash: str) -> Optional[str]:
        """Computes HMAC-SHA256 signature if a secret key is configured."""
        if not self.is_signed_mode or not self.secret_key:
            return None
        return hmac.new(
            self.secret_key.encode("utf-8"),
            current_hash.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def append_event(
        self,
        dispute_id: str,
        event_type: str,
        payload: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> AuditEntry:
        """
        Appends a new event payload to the cryptographic hash chain.
        """
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry_id = len(self.entries) + 1
        
        # Previous hash from latest entry or genesis
        prev_hash = self.entries[-1].current_hash if self.entries else GENESIS_HASH
        
        # Hash of the payload
        payload_hash = self._canonical_hash(payload)
        
        # Chained hash of this entry
        curr_hash = self._compute_entry_hash(
            entry_id=entry_id,
            previous_hash=prev_hash,
            payload_hash=payload_hash,
            timestamp=ts,
            event_type=event_type,
            dispute_id=dispute_id
        )

        # HMAC Signature (if secret key configured)
        signature = self._compute_hmac(curr_hash)
        sig_mode = "HMAC_SHA256" if signature is not None else "UNSIGNED_DEMO"

        entry = AuditEntry(
            entry_id=entry_id,
            dispute_id=dispute_id,
            event_type=event_type,
            timestamp=ts,
            previous_hash=prev_hash,
            payload_hash=payload_hash,
            current_hash=curr_hash,
            signature_mode=sig_mode,
            signature=signature,
            payload=payload,
        )

        self.entries.append(entry)
        self._persist_entry(entry)
        return entry

    def verify_integrity(self, require_signed: Optional[bool] = None) -> Tuple[bool, Optional[str]]:
        """
        Verifies the cryptographic integrity and HMAC authenticity of the entire ledger chain.

        Args:
            require_signed: If True, fails if any entry lacks a valid HMAC signature.
                            If None, defaults to True when secret_key is configured.

        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        if not self.entries:
            return True, None

        enforce_hmac = require_signed if require_signed is not None else self.is_signed_mode

        for i, entry in enumerate(self.entries):
            # 1. Verify Entry ID sequence
            expected_id = i + 1
            if entry.entry_id != expected_id:
                return False, f"Entry sequence broken at index {i}: Expected ID {expected_id}, found {entry.entry_id} (Insertion/Deletion detected)"

            # 2. Verify Previous Hash linkage
            expected_prev = GENESIS_HASH if i == 0 else self.entries[i - 1].current_hash
            if entry.previous_hash != expected_prev:
                return False, f"Hash chain broken at entry #{entry.entry_id}: Previous hash mismatch (Reordering or modification detected)"

            # 3. Verify Payload Hash
            recomputed_payload_hash = self._canonical_hash(entry.payload)
            if entry.payload_hash != recomputed_payload_hash:
                return False, f"Payload tampering detected at entry #{entry.entry_id}: Payload hash does not match content"

            # 4. Verify Current Entry Hash
            recomputed_entry_hash = self._compute_entry_hash(
                entry_id=entry.entry_id,
                previous_hash=entry.previous_hash,
                payload_hash=recomputed_payload_hash,
                timestamp=entry.timestamp,
                event_type=entry.event_type,
                dispute_id=entry.dispute_id
            )
            if entry.current_hash != recomputed_entry_hash:
                return False, f"Cryptographic entry hash invalid at entry #{entry.entry_id}: Current hash mismatch"

            # 5. Verify HMAC Signature (if enforced or secret key is present)
            if enforce_hmac:
                if not self.secret_key:
                    return False, f"HMAC verification requested at entry #{entry.entry_id} but no secret key is configured"
                if not entry.signature or entry.signature_mode != "HMAC_SHA256":
                    return False, f"Unsigned entry detected in signed ledger at entry #{entry.entry_id} (Found mode '{entry.signature_mode}')"
                
                recomputed_sig = hmac.new(
                    self.secret_key.encode("utf-8"),
                    entry.current_hash.encode("utf-8"),
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(entry.signature, recomputed_sig):
                    return False, f"HMAC signature forgery detected at entry #{entry.entry_id}: Invalid secret key or signature"

        return True, None

    def get_verification_metadata(self) -> Dict[str, Any]:
        """Returns structured metadata about ledger status, signing mode, and cryptographic authentication."""
        is_valid, error_msg = self.verify_integrity()
        signed_count = sum(1 for e in self.entries if e.signature_mode == "HMAC_SHA256" and e.signature is not None)
        unsigned_count = len(self.entries) - signed_count

        return {
            "is_valid": is_valid,
            "total_entries": len(self.entries),
            "signing_mode": self.signing_mode,
            "is_signed_mode": self.is_signed_mode,
            "is_cryptographically_authenticated": bool(is_valid and self.is_signed_mode and signed_count == len(self.entries) and len(self.entries) > 0),
            "signed_entries_count": signed_count,
            "unsigned_entries_count": unsigned_count,
            "error_message": error_msg,
        }

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Appends entry to JSONL file on disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_file)), exist_ok=True)
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def _load_ledger(self) -> None:
        """Loads and verifies existing ledger file if present."""
        self.entries = []
        if os.path.exists(self.ledger_file):
            with open(self.ledger_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(AuditEntry.model_validate_json(line))

