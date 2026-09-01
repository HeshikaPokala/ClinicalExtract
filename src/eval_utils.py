"""Shared scoring utilities for comparing extracted diagnoses/medications against ground truth."""

import re

_PAREN_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_LEADING_DRUG_NAME_RE = re.compile(r"^([a-z][a-z\s\-]*?)(?=\s+[\d.]|\s*$)")


def normalize_condition(text: str) -> str:
    """'Streptococcal sore throat (disorder)' -> 'streptococcal sore throat'"""
    text = text.lower().strip()
    text = _PAREN_TAG_RE.sub("", text)
    text = _NON_ALNUM_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_medication(text: str) -> str:
    """'lisinopril 10 MG Oral Tablet' -> 'lisinopril' (drop dosage/form, keep drug name)."""
    text = text.lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    m = _LEADING_DRUG_NAME_RE.match(text)
    name = m.group(1).strip() if m else text
    return name if name else text


def fuzzy_match(a: str, b: str) -> bool:
    """True if normalized strings are equal or one contains the other as a whole-word match."""
    if not a or not b:
        return False
    if a == b:
        return True
    return (a in b or b in a) and min(len(a), len(b)) >= 4


def score_set(predicted: list, gold: list, normalize_fn) -> dict:
    """Set-based precision/recall/F1 between predicted and gold strings, using fuzzy matching."""
    pred_norm = [normalize_fn(p) for p in predicted if isinstance(p, str) and p.strip()]
    gold_norm = [normalize_fn(g) for g in gold if isinstance(g, str) and g.strip()]

    matched_gold = set()
    true_positives = 0
    for p in pred_norm:
        for i, g in enumerate(gold_norm):
            if i in matched_gold:
                continue
            if fuzzy_match(p, g):
                matched_gold.add(i)
                true_positives += 1
                break

    precision = true_positives / len(pred_norm) if pred_norm else (1.0 if not gold_norm else 0.0)
    recall = true_positives / len(gold_norm) if gold_norm else (1.0 if not pred_norm else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "n_predicted": len(pred_norm),
        "n_gold": len(gold_norm),
    }


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
