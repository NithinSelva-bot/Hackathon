"""
tenseai.py — Tone analysis and fairness scoring for HireFlip.

Replace the stub implementations with your real logic / model calls.
"""

# ---------------------------------------------------------------------------
# Simple keyword lists you can expand
# ---------------------------------------------------------------------------

_MASCULINE_CODED = [
    "ninja", "rockstar", "dominant", "assertive", "competitive",
    "chairman", "manpower", "he/she", "his/her", "brotherhood",
]

_FEMININE_CODED = [
    "nurturing", "supportive", "collaborative", "empathetic",
    "gentle", "compassionate",
]

_AGE_CODED = [
    "young", "energetic", "recent graduate", "fresh", "digital native",
    "millennials", "gen z",
]

_EXCLUSIONARY = [
    "culture fit", "native speaker", "able-bodied",
    "must be able to lift", "unrestricted",
]

ALL_BIAS_KEYWORDS: list[str] = (
    _MASCULINE_CODED + _FEMININE_CODED + _AGE_CODED + _EXCLUSIONARY
)


def analyze_tone(text: str) -> str:
    """
    Mock implementation: Analyzes the tone of the given text using keyword matching.

    Returns one of: "aggressive", "neutral", "welcoming"

    This is a mock and does not call any real API. Later, replace with real TenseAI API call.
    """
    text_lower = text.lower()

    # Keywords for aggressive tone
    aggressive_keywords = ["angry", "hate", "fight", "attack", "rage", "violent", "aggressive", "hostile"]

    # Keywords for welcoming tone
    welcoming_keywords = ["welcome", "hello", "friendly", "kind", "warm", "inviting", "helpful", "supportive"]

    # Check for aggressive
    if any(kw in text_lower for kw in aggressive_keywords):
        return "aggressive"

    # Check for welcoming
    if any(kw in text_lower for kw in welcoming_keywords):
        return "welcoming"

    # Default to neutral
    return "neutral"


def calculate_fairness_score(groq_score: int, tone: str) -> int:
    """
    Combines the Groq bias score with the tone signal into a 0-100 fairness score.
    Higher = fairer / less biased.

    Args:
        groq_score: Raw score from detect_bias() (0-100, where 100 = very biased).
        tone:       One of "inclusive" | "neutral" | "exclusive" | "aggressive".

    Returns:
        int: Fairness score 0-100.
    """
    tone_penalty = {
        "inclusive":  0,
        "neutral":    5,
        "exclusive": 15,
        "aggressive": 25,
    }

    # Groq score is treated as a bias score; invert it for fairness.
    base_fairness = 100 - max(0, min(100, groq_score))
    penalty = tone_penalty.get(tone, 10)
    return max(0, min(100, base_fairness - penalty))


def get_bias_keywords(job_description: str) -> list[str]:
    """
    Scans the job description and returns all matched bias-indicator keywords.

    Returns:
        A deduplicated list of flagged words/phrases found in the JD.
    """
    jd_lower = job_description.lower()
    found = [kw for kw in ALL_BIAS_KEYWORDS if kw in jd_lower]
    return list(dict.fromkeys(found))  # preserve order, deduplicate
