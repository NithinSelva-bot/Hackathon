import os
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_MOCK_GROQ = os.getenv("USE_MOCK_GROQ", "false").lower() in ("1", "true", "yes")

if not GROQ_API_KEY:
    print("[groq_ai] GROQ_API_KEY is not configured. Using mock bias detection.")
    USE_MOCK_GROQ = True

if not USE_MOCK_GROQ:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

def _create_chat_completion(prompt: str):
    try:
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as exc:
        print(f"[groq_ai] Groq API request failed: {exc!r}")
        print("[groq_ai] Falling back to mock bias detection.")
        return None


def _mock_detect_bias(job_description: str) -> dict:
    jd = job_description.lower()
    categories = {
        "gender": [],
        "age": [],
        "race": [],
        "disability": [],
    }

    if any(word in jd for word in ["he", "she", "him", "her", "ninja", "rockstar", "chairman"]):
        categories["gender"].append("gendered language")
    if any(word in jd for word in ["young", "energetic", "fresh", "recent graduate"]):
        categories["age"].append("age-biased language")
    if any(word in jd for word in ["native speaker", "able-bodied", "culture fit"]):
        categories["disability"].append("ableist or exclusionary language")
    if any(word in jd for word in ["black", "white", "asian", "latino", "hispanic"]):
        categories["race"].append("race-related language")

    explanation = "This is a mock bias analysis based on keyword matching. Replace with the real Groq API when ready."
    rewritten_jd = job_description
    if categories["gender"] or categories["age"] or categories["race"] or categories["disability"]:
        rewritten_jd = job_description.replace("ninja", "team member").replace("rockstar", "strong candidate")

    return {
        "categories": categories,
        "explanation": explanation,
        "rewritten_jd": rewritten_jd,
        "groq_score": 50,
    }


def detect_bias(job_description: str) -> dict:
    prompt = f"""
You are an expert HR bias detector. Analyze this job description for bias.

Job Description:
{job_description}

Respond ONLY with a valid JSON object in this exact format:
{{
    "categories": {{
        "gender": ["list of gendered words found"],
        "age": ["list of age-biased words found"],
        "race": ["list of racially biased words found"],
        "disability": ["list of ableist words found"]
    }},
    "explanation": "A clear explanation of the bias found",
    "rewritten_jd": "A fully rewritten unbiased version of the job description",
    "groq_score": 50
}}

groq_score should be 0-100 where 100 means extremely biased, 0 means no bias.
"""
    if USE_MOCK_GROQ:
        return _mock_detect_bias(job_description)

    response = _create_chat_completion(prompt)
    if response is None:
        return _mock_detect_bias(job_description)

    text = response.choices[0].message.content
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def rewrite_jd(job_description: str) -> str:
    prompt = f"""
Rewrite this job description to be completely unbiased, inclusive, and welcoming to all candidates.
Remove any gendered, ageist, racist, or ableist language.

Job Description:
{job_description}

Return only the rewritten job description, nothing else.
"""
    if USE_MOCK_GROQ:
        return job_description.replace("ninja", "team member").replace("rockstar", "strong candidate")

    response = _create_chat_completion(prompt)
    if response is None:
        return job_description.replace("ninja", "team member").replace("rockstar", "strong candidate")
    return response.choices[0].message.content.strip()