from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env
load_dotenv()

# Local module imports (must exist in the same directory)
#from groq_ai import detect_bias, rewrite_jd
from tenseai import analyze_tone, calculate_fairness_score, get_bias_keywords

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HireFlip API",
    description="AI-powered bias detector and rewriter for job descriptions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins for React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class JobDescriptionRequest(BaseModel):
    job_description: str

class AnalyzeResponse(BaseModel):
    fairness_score: int
    biased_words: list
    categories: dict
    explanation: str
    rewritten_jd: str
    tone: str

class RewriteResponse(BaseModel):
    rewritten_jd: str

class HealthResponse(BaseModel):
    status: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Simple liveness check."""
    print("[GET /health] Health check requested")
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(request: JobDescriptionRequest):
    """
    Analyze a job description for bias and return a comprehensive report.

    - Validates that the JD is at least 50 characters.
    - Calls groq_ai.detect_bias() for AI-powered bias detection.
    - Calls tenseai.analyze_tone() for tone classification.
    - Calls tenseai.calculate_fairness_score() to produce a 0-100 score.
    - Calls tenseai.get_bias_keywords() for flagged word extraction.
    """
    jd = request.job_description
    print(f"[POST /analyze] Incoming request — JD length: {len(jd)} chars")

    # --- Validation ---
    if len(jd.strip()) < 50:
        print("[POST /analyze] Validation failed: JD too short")
        raise HTTPException(
            status_code=422,
            detail="Job description must be at least 50 characters long.",
        )

    try:
        print("[POST /analyze] Calling detect_bias()...")
        bias_result = detect_bias(jd)
        # Expected shape from groq_ai: {
        #   "categories": dict, "explanation": str,
        #   "rewritten_jd": str, "groq_score": int  (optional)
        # }

        print("[POST /analyze] Calling analyze_tone()...")
        tone = analyze_tone(jd)

        print("[POST /analyze] Calling calculate_fairness_score()...")
        groq_score = bias_result.get("groq_score", 50)
        fairness_score = calculate_fairness_score(groq_score, tone)

        print("[POST /analyze] Calling get_bias_keywords()...")
        biased_words = get_bias_keywords(jd)

        response = {
            "fairness_score": fairness_score,
            "biased_words": biased_words,
            "categories": bias_result.get("categories", {}),
            "explanation": bias_result.get("explanation", ""),
            "rewritten_jd": bias_result.get("rewritten_jd", ""),
            "tone": tone,
        }

        print(
            f"[POST /analyze] Done — score: {fairness_score}, "
            f"tone: {tone}, flagged words: {len(biased_words)}"
        )
        return response

    except HTTPException:
        raise  # Re-raise validation errors as-is
    except Exception as exc:
        print(f"[POST /analyze] ERROR: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)}",
        )


@app.post("/rewrite", response_model=RewriteResponse, tags=["Rewrite"])
async def rewrite(request: JobDescriptionRequest):
    """
    Return an unbiased rewrite of the provided job description.
    """
    jd = request.job_description
    print(f"[POST /rewrite] Incoming request — JD length: {len(jd)} chars")

    if len(jd.strip()) < 50:
        print("[POST /rewrite] Validation failed: JD too short")
        raise HTTPException(
            status_code=422,
            detail="Job description must be at least 50 characters long.",
        )

    try:
        print("[POST /rewrite] Calling rewrite_jd()...")
        rewritten = rewrite_jd(jd)
        print("[POST /rewrite] Done — rewrite complete")
        return {"rewritten_jd": rewritten}

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[POST /rewrite] ERROR: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Rewrite failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
