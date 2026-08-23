"""
SYVORA — Claim Intent Lexicons and Deterministic Pattern Definitions
=====================================================================
Defines the deterministic matching rules, regular expressions, and negative/positive
trigger vocabularies for standard dispute claim classifications.
"""

import re
from typing import Dict, List, Pattern
from src.agent.schemas import ClaimIntent

# Negation prefixes and modifiers that flip the meaning of adjacent verbs/predicates
NEGATION_WORDS = {
    "not", "never", "no", "n't", "none", "neither", "nor", "hardly", "scarcely",
    "barely", "without", "didnt", "didn't", "havent", "haven't", "dont", "don't",
    "wasnt", "wasn't", "isnt", "isn't", "couldnt", "couldn't", "wouldnt", "would't",
    "cannot", "can't", "cant", "wont", "won't"
}

# Affirmation / Delivery confirmations that explicitly refute non-delivery
AFFIRMATIVE_DELIVERY_PATTERNS = [
    r"\b(did\s+receive|have\s+received|i\s+received|package\s+arrived|item\s+arrived|order\s+arrived|got\s+the\s+package|got\s+the\s+item|got\s+the\s+order)\b",
    r"\b(received\s+it|received\s+the\s+package|received\s+my\s+package|arrived\s+on\s+time)\b",
]

# Core Intent Regex Patterns
INTENT_PATTERNS: Dict[ClaimIntent, List[Pattern]] = {
    ClaimIntent.REFUND_NOT_RECEIVED: [
        re.compile(r"\b(refund\s+(?:was\s+)?(?:not|never)\s+received|refund\s+never\s+arrived)\b", re.I),
        re.compile(r"\b(haven'?t\s+(?:\w+\s+)?received\s+(?:my\s+|the\s+|a\s+)?refund|have\s*(?:\w+\s+)?not\s+(?:\w+\s+)?received\s+(?:my\s+|the\s+|a\s+)?refund|never\s+(?:\w+\s+)?got\s+(?:my\s+|the\s+|a\s+)?refund)\b", re.I),
        re.compile(r"\b(promised\s+(?:a\s+)?refund|refund\s+missing|refund\s+not\s+processed|money\s+not\s+refunded|no\s+refund)\b", re.I),
        re.compile(r"\b(waiting\s+for\s+(?:my\s+|the\s+|a\s+)?refund|where\s+is\s+my\s+refund|credit\s+not\s+received)\b", re.I),
        re.compile(r"\b(return(?:ed)?\s+(?:the\s+item\s+|item\s+)?(?:with\s+)?no\s+refund|return(?:ed)?\s+(?:item\s+|order\s+)?but\s+no\s+(?:money|refund))\b", re.I),
    ],
    ClaimIntent.NON_DELIVERY: [
        re.compile(r"\b(never\s+received|did\s*not\s+receive|didn'?t\s+receive|have\s*not\s+received|haven'?t\s+received)\s+(?:the\s+|my\s+)?(?:package|item|order|goods|shipment|product|delivery)\b", re.I),
        re.compile(r"\b(not\s+delivered|never\s+delivered|was\s*not\s+delivered|wasn'?t\s+delivered|failed\s+to\s+deliver)\b", re.I),
        re.compile(r"\b(never\s+arrived|did\s*not\s+arrive|didn'?t\s+arrive|has\s*not\s+arrived|hasn'?t\s+arrived)\b", re.I),
        re.compile(r"\b(missing\s+(?:package|item|order|shipment|delivery)|lost\s+in\s+transit|empty\s+box)\b", re.I),
        re.compile(r"\b(never\s+got\s+(?:my|the)\s+(?:package|item|order|goods)|didn'?t\s+get\s+(?:my|the)\s+(?:package|item|order|goods))\b", re.I),
        re.compile(r"\b(?:package|item|order|goods|shipment)\s+(?:was\s+)?(?:not|never)\s+received\b", re.I),
        re.compile(r"\b(where\s+is\s+my\s+(?:order|package|shipment)|package\s+was\s+never\s+dropped|undelivered|non-delivery)\b", re.I),
        re.compile(r"\b(did\s*not\s+receive|didn'?t\s+receive|never\s+received)\b", re.I),
    ],
    ClaimIntent.UNAUTHORIZED_TRANSACTION: [
        re.compile(r"\b(unauthorized|unauthorised|unrecognized|unrecognised|fraudulent\s+charge|fraud\s+charge)\b", re.I),
        re.compile(r"\b(didn'?t\s+authorize|did\s*not\s+authorize|did\s*not\s+authorise|didn'?t\s+authorise)\b", re.I),
        re.compile(r"\b(don'?t\s+recognize|do\s*not\s+recognize|don'?t\s+recognise|do\s*not\s+recognise)\b", re.I),
        re.compile(r"\b(never\s+made\s+this|did\s*not\s+make\s+this|didn'?t\s+make\s+this|never\s+bought\s+this)\b", re.I),
        re.compile(r"\b(card\s+was\s+stolen|stolen\s+card|card\s+compromised|card\s+hacked|identity\s+theft)\b", re.I),
        re.compile(r"\b(not\s+me|wasn'?t\s+me|was\s*not\s+me|someone\s+else\s+used\s+my\s+card|account\s+hacked)\b", re.I),
        re.compile(r"\b(unknown\s+transaction|unknown\s+charge|unknown\s+payment|suspicious\s+activity)\b", re.I),
    ],
    ClaimIntent.DUPLICATE_CHARGE: [
        re.compile(r"\b(charged\s+(?:\w+\s+)?(?:twice|2\s*times|multiple\s+times|again)|billed\s+(?:\w+\s+)?(?:twice|2\s*times))\b", re.I),
        re.compile(r"\b(double\s+(?:charged|billing|deduction|charge)|debited\s+(?:\w+\s+)?(?:twice|2\s*times|again))\b", re.I),
        re.compile(r"\b(duplicate\s+(?:charge|transaction|payment|billing|entry))\b", re.I),
        re.compile(r"\b(two\s+charges|multiple\s+charges|charged\s+more\s+than\s+once|paid\s+(?:\w+\s+)?(?:twice|2\s*times))\b", re.I),
        re.compile(r"\b(repeated\s+charge|recurring\s+error)\b", re.I),
    ],
    ClaimIntent.WRONG_AMOUNT: [
        re.compile(r"\b(wrong\s+amount|incorrect\s+amount|overcharged|over-charged|charged\s+extra)\b", re.I),
        re.compile(r"\b(charged\s+more\s+than|billed\s+more\s+than|billed\s+incorrectly|incorrect\s+charge)\b", re.I),
        re.compile(r"\b(amount\s+mismatch|amount\s+is\s+wrong|price\s+difference|extra\s+fee\s+charged)\b", re.I),
        re.compile(r"\b(charged\s+higher|higher\s+than\s+agreed|billed\s+extra|excessive\s+charge)\b", re.I),
    ],
    ClaimIntent.CANCELLATION: [
        re.compile(r"\b(cancel(?:led|ed|ing)?\s+(?:the\s+|my\s+)?(?:order|subscription|service|account|membership))\b", re.I),
        re.compile(r"\b(requested\s+(?:a\s+)?cancellation|cancellation\s+requested|cancellation\s+was\s+approved|asked\s+to\s+cancel)\b", re.I),
        re.compile(r"\b(charged\s+after\s+cancel(?:lation|ing|ed|led)|billed\s+after\s+cancel(?:ling|ing|ed|led)|recurring\s+fee\s+still\s+deducted)\b", re.I),
        re.compile(r"\b(unsubscribed|auto-renewed\s+after\s+cancel)\b", re.I),
    ],
}
