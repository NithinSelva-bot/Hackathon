import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def detect_bias(job_description: str) -> dict:
    """
    Analyzes a job description for bias across gender, age, cultural, and tone dimensions.

    Args:
        job_description: The raw job description text to analyze.

    Returns:
        A dict with biased_words, fairness_score, categories breakdown, and explanation.
    """
    system_prompt = (
        "You are a bias detection expert. Analyze job descriptions for gender, age, cultural, and tone bias."
    )

    user_prompt = f"""Analyze the following job description for bias and return ONLY a valid JSON object with no extra text or markdown.

The JSON must follow this exact structure:
{{
  "biased_words": ["word1", "word2"],
  "fairness_score": <integer 0-100, where 100 is fully fair>,
  "categories": {{
    "gender": <integer 0-100, bias severity>,
    "age": <integer 0-100, bias severity>,
    "cultural": <integer 0-100, bias severity>,
    "tone": <integer 0-100, bias severity>
  }},
  "explanation": "<brief explanation of detected bias>"
}}

Job Description:
{job_description}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON response: {e}",
            "biased_words": [],
            "fairness_score": -1,
            "categories": {"gender": -1, "age": -1, "cultural": -1, "tone": -1},
            "explanation": "Could not analyze the job description due to a parsing error.",
        }
    except Exception as e:
        return {
            "error": str(e),
            "biased_words": [],
            "fairness_score": -1,
            "categories": {"gender": -1, "age": -1, "cultural": -1, "tone": -1},
            "explanation": "An unexpected error occurred during bias detection.",
        }


def rewrite_jd(job_description: str) -> str:
    """
    Rewrites a job description to remove bias while preserving all job requirements.

    Args:
        job_description: The original job description text.

    Returns:
        The rewritten, bias-free job description as a plain string.
    """
    system_prompt = (
        "You are an expert HR writer specializing in inclusive, bias-free job descriptions. "
        "Your rewrites retain all original job requirements and responsibilities while eliminating "
        "any gender, age, cultural, or tone bias. Return only the rewritten job description — "
        "no commentary, no explanations, no markdown."
    )

    user_prompt = f"""Rewrite the following job description to remove all bias. 
Keep the same job requirements, responsibilities, and qualifications.
Return only the rewritten job description text — nothing else.

Original Job Description:
{job_description}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error rewriting job description: {e}"


# ── Quick smoke test ──────────────────────────────────────────────────────────

SAMPLE_JD = """
We are looking for a young, energetic rockstar developer to join our brotherhood.
The ideal candidate is a recent graduate (0-2 years experience) who can hustle hard
and dominate in a fast-paced environment. You must be a native English speaker with
strong manpower to handle aggressive deadlines. We need a guy who can own the codebase
and crush it every day. Perks include beer Fridays and a ping-pong table.

Requirements:
- Proficiency in Python and JavaScript
- Experience with REST APIs
- Strong problem-solving skills
- Bachelor's degree in Computer Science
"""

if __name__ == "__main__":
    print("=" * 60)
    print("HireFlip — AI Bias Detector")
    print("=" * 60)

    print("\n📋 Original Job Description:")
    print(SAMPLE_JD)

    print("\n🔍 Running bias detection...")
    bias_report = detect_bias(SAMPLE_JD)

    if "error" in bias_report:
        print(f"  ⚠️  Error: {bias_report['error']}")
    else:
        print(f"\n  Fairness Score : {bias_report.get('fairness_score')}/100")
        print(f"  Biased Words   : {', '.join(bias_report.get('biased_words', []))}")
        print(f"\n  Category Scores (higher = more biased):")
        for category, score in bias_report.get("categories", {}).items():
            bar = "█" * (score // 10) + "░" * (10 - score // 10)
            print(f"    {category:<10} [{bar}] {score}")
        print(f"\n  Explanation:\n  {bias_report.get('explanation')}")

    print("\n✍️  Rewriting job description to remove bias...")
    rewritten = rewrite_jd(SAMPLE_JD)

    print("\n✅ Rewritten Job Description:")
    print(rewritten)
    print("\n" + "=" * 60)
