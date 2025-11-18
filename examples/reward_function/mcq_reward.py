import re
from typing import Any, Optional, Dict

CHOICES = ["A", "B", "C", "D"]
LETTER_RE = re.compile(r"(?:Final\s*Answer|Answer)\s*:\s*([A-D])\b", re.IGNORECASE)


def _extract_letter(s: str) -> Optional[str]:
    # First try to extract from <answer> tags
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: look for "Answer:" or "Final Answer:" pattern
    m2 = LETTER_RE.search(s)
    if m2:
        return m2.group(1).upper()
    # Final fallback: bare single letter on a line
    m3 = re.search(r"^[\s>*-]*([A-D])\s*$", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    if m3:
        return m3.group(1).upper()
    return None


def format_reward(response: str) -> float:
    """
    Check if response contains required format: <think> tags and answer in <answer> tags with valid letter (A-D).
    Returns 1.0 if format is correct, 0.0 otherwise.
    """
    pattern = re.compile(r"<think>.*</think>.*<answer>\s*[A-D]\s*</answer>", re.IGNORECASE | re.DOTALL)
    format_match = re.search(pattern, response or "")
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Check if extracted letter matches ground truth.
    Returns 1.0 if correct, 0.0 otherwise.
    """
    pred = _extract_letter(response or "")
    if pred is None:
        return 0.0
    return 1.0 if pred == ground_truth else 0.0


def compute_score(
    reward_inputs: list[Dict[str, Any]],
    format_weight: float = 0.1
) -> list[Dict[str, float]]:
    """
    Compute scores for multiple-choice questions.

    Args:
        reward_inputs: List of dicts with 'response' and 'ground_truth' keys
        format_weight: Weight for format reward (default 0.1)

    Returns:
        List of dicts with 'overall', 'format', and 'accuracy' scores
    """
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
