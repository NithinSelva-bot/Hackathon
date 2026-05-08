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


def analyze_tone(job_description: str) -> str:
    """
    Classifies the overall tone of the job description.

    Returns one of: "inclusive", "neutral", "exclusive", "aggressive"

    Replace this with a real classifier (e.g. a fine-tuned model or
    another Groq prompt) for production use.
    """
    jd_lower = job_description.lower()

    masculine_hits = sum(1 for w in _MASCULINE_CODED if w in jd_lower)
    exclusionary_hits = sum(1 for w in _EXCLUSIONARY if w in jd_lower)

    if masculine_hits >= 4 or exclusionary_hits >= 2:
        return "aggressive"
    if masculine_hits >= 2 or exclusionary_hits == 1:
        return "exclusive"
    if masculine_hits == 1:
        return "neutral"
    return "inclusive"


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
