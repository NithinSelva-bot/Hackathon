import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()

from groq_ai import detect_bias, rewrite_jd
from tenseai import analyze_tone, calculate_fairness_score, get_bias_keywords

app = FastAPI(
    title="HireFlip API",
    description="AI-powered bias detector for job descriptions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobDescriptionRequest(BaseModel):
    job_description: str

@app.get("/health")
async def health():
    print("[GET /health] Health check")
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(request: JobDescriptionRequest):
    jd = request.job_description
    print(f"[POST /analyze] JD length: {len(jd)} chars")
    if len(jd.strip()) < 50:
        raise HTTPException(status_code=422, detail="JD must be at least 50 characters.")
    try:
        bias_result = detect_bias(jd)
        tone = analyze_tone(jd)
        groq_score = bias_result.get("groq_score", 50)
        fairness_score = calculate_fairness_score(groq_score, tone)
        biased_words = get_bias_keywords(jd)
        return {
            "fairness_score": fairness_score,
            "biased_words": biased_words,
            "categories": bias_result.get("categories", {}),
            "explanation": bias_result.get("explanation", ""),
            "rewritten_jd": bias_result.get("rewritten_jd", ""),
            "tone": tone,
        }
    except Exception as exc:
        print(f"[POST /analyze] ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/rewrite")
async def rewrite(request: JobDescriptionRequest):
    jd = request.job_description
    print(f"[POST /rewrite] JD length: {len(jd)} chars")
    if len(jd.strip()) < 50:
        raise HTTPException(status_code=422, detail="JD must be at least 50 characters.")
    try:
        rewritten = rewrite_jd(jd)
        return {"rewritten_jd": rewritten}
    except Exception as exc:
        print(f"[POST /rewrite] ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)