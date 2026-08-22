"""
SentinelRisk — Deterministic Offline Natural Language Claim Extractor
======================================================================
Extracts structured advisory claim signals from sanitized customer dispute remarks.

Security & Architectural Constraints:
- Consumes sanitized customer remarks ONLY.
- Never reads dispute operational records, evidence fields, or ML feature matrices.
- Pure Python deterministic rule heuristics; zero network calls; zero external dependencies.
- Produces ADVISORY claim packages with `advisory_only=True` strictly enforced.
"""

import hashlib
import re
from typing import List, Optional, Set, Tuple, Union

from src.agent.schemas import (
    ClaimIntent,
    ClaimSignal,
    ClaimSignalPackage,
    CustomerClaimEvidence,
)
from src.nlp.intents import (
    AFFIRMATIVE_DELIVERY_PATTERNS,
    INTENT_PATTERNS,
    NEGATION_WORDS,
)


class DeterministicClaimExtractor:
    """
    Negation-aware deterministic claim intent extractor for customer dispute text.
    """

    @classmethod
    def extract_signals(
        cls,
        sanitized_input: Union[str, CustomerClaimEvidence, None]
    ) -> ClaimSignalPackage:
        """
        Extracts structured advisory dispute signals from sanitized customer remarks.

        Args:
            sanitized_input: A sanitized string or a CustomerClaimEvidence model instance.

        Returns:
            ClaimSignalPackage containing extracted advisory signals, primary intent,
            and provenance hash.
        """
        if sanitized_input is None:
            return cls._empty_package(source_sha256="")

        if isinstance(sanitized_input, CustomerClaimEvidence):
            text = sanitized_input.sanitized_text or ""
            source_sha256 = sanitized_input.sanitized_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
        else:
            text = str(sanitized_input)
            source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

        clean_text = text.strip()
        if not clean_text:
            return cls._empty_package(source_sha256=source_sha256)

        # Tokenize and normalize for pattern analysis
        signals: List[ClaimSignal] = []
        sentences = [s.strip() for s in re.split(r"[.!?;\n]+", clean_text) if s.strip()]

        for intent, patterns in INTENT_PATTERNS.items():
            matched_phrases: List[str] = []
            highest_conf = 0.0
            is_intent_negated = False
            best_snippet = ""

            for sentence in sentences:
                sentence_lower = sentence.lower()

                # Special check for NON_DELIVERY: If sentence explicitly affirms receipt without negative delivery phrases, skip
                if intent == ClaimIntent.NON_DELIVERY:
                    has_affirmative_receipt = any(
                        re.search(aff_p, sentence_lower, re.I) for aff_p in AFFIRMATIVE_DELIVERY_PATTERNS
                    )
                    has_explicit_non_delivery = any(
                        re.search(r"\b(package|item|order|goods|shipment)\s+(?:was\s+)?not\s+received\b", sentence_lower) or
                        re.search(r"\bnever\s+received\s+(?:my|the)?\s*(package|item|order|goods)\b", sentence_lower)
                        for _ in [1]
                    )
                    if has_affirmative_receipt and not has_explicit_non_delivery:
                        continue

                for pattern in patterns:
                    matches = list(pattern.finditer(sentence))
                    for m in matches:
                        match_text = m.group()
                        
                        # Disambiguation: if "not received" or "didn't receive" is immediately followed by "refund", skip for NON_DELIVERY
                        if intent == ClaimIntent.NON_DELIVERY:
                            following_text = sentence[m.end():m.end() + 20].lower()
                            if following_text.strip().startswith("refund") or following_text.strip().startswith("my refund"):
                                continue

                        matched_phrases.append(match_text)
                        
                        # Context snippet (surrounding window)
                        start_pos = max(0, m.start() - 20)
                        end_pos = min(len(sentence), m.end() + 20)
                        best_snippet = sentence[start_pos:end_pos].strip()

                        # Check for local clause negation
                        clause_before = sentence[:m.start()].lower()
                        words_before = clause_before.split()[-3:] if clause_before else []
                        negated_locally = any(w in NEGATION_WORDS for w in words_before)

                        # Evaluate heuristic score based on match specificity
                        conf = 0.85 if len(match_text.split()) > 1 else 0.70
                        if negated_locally:
                            if any(neg in match_text.lower() for neg in ["not", "never", "didn"]):
                                is_intent_negated = True
                                conf = 0.40
                        
                        if conf > highest_conf:
                            highest_conf = conf

            if matched_phrases and highest_conf > 0.50 and not is_intent_negated:
                signal = ClaimSignal(
                    intent=intent,
                    confidence_score=round(highest_conf, 2),
                    matched_keywords=[p.lower() for p in matched_phrases[:3]],
                    matched_phrases=matched_phrases[:3],
                    is_negated=is_intent_negated,
                    context_snippet=best_snippet,
                    advisory_only=True
                )
                signals.append(signal)

        # Determine primary and secondary intents
        if not signals:
            return cls._empty_package(source_sha256=source_sha256)

        # Sort signals by deterministic confidence score descending
        signals.sort(key=lambda s: s.confidence_score, reverse=True)
        primary = signals[0].intent
        secondary = [s.intent for s in signals[1:] if s.intent != primary]

        return ClaimSignalPackage(
            primary_intent=primary,
            secondary_intents=secondary,
            signals=signals,
            source_sanitized_sha256=source_sha256,
            has_structured_claim=True,
            advisory_only=True
        )

    @classmethod
    def _empty_package(cls, source_sha256: str) -> ClaimSignalPackage:
        """Returns an empty advisory claim package when no structured signals are found."""
        return ClaimSignalPackage(
            primary_intent=ClaimIntent.OTHER,
            secondary_intents=[],
            signals=[],
            source_sanitized_sha256=source_sha256,
            has_structured_claim=False,
            advisory_only=True
        )
