import os
from groq import Groq
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. Set it in Render environment variables or in .env for local testing."
    )

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
        raise RuntimeError(
            "Failed to connect to Groq API. Check GROQ_API_KEY and network access."
        ) from exc

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
    response = _create_chat_completion(prompt)
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
    response = _create_chat_completion(prompt)
    return response.choices[0].message.content.strip()