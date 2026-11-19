"""FastAPI application entry point for the Karmanometer resume analyzer.

The ``/analyse`` endpoint accepts a PDF upload and a job description as
multipart form fields, then delegates extraction, cleaning, scoring, and gap
analysis to the standalone pipeline modules.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline.step4_keywordGap import analyze_keyword_gaps
from pipeline.step1_pdfExtractor import extract_text_from_pdf
from pipeline.step5_schemas import AnalyseResponse, ErrorResponse, HealthResponse, ScoringMethod
from pipeline.step3_scorer import APPROVED_SKILLS, compute_score
from pipeline.step2_textCleaner import clean_text


_MAX_PDF_BYTES = 5 * 1024 * 1024
_ALLOWED_PDF_CONTENT_TYPES = frozenset(
    {"application/pdf", "application/octet-stream"}
)

app = FastAPI(
    title="Karmanometer API",
    description="Resume-to-Job-Description matching and keyword gap analysis API.",
    version="1.0.0",
)

_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_frontend_origin = os.getenv("FRONTEND_ORIGIN")
if _frontend_origin and _frontend_origin not in _allowed_origins:
    _allowed_origins.append(_frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Report API health and the number of active taxonomy skills."""
    return HealthResponse(status="ok", skills_loaded=len(APPROVED_SKILLS))


@app.post(
    "/analyse",
    response_model=AnalyseResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    tags=["Analysis"],
)
async def analyse_resume(
    resume_file: UploadFile = File(
        ...,
        description="Candidate resume in PDF format (max 5 MB).",
    ),
    job_description: str = Form(
        ...,
        description="Target job description text.",
    ),
    scoring_method: ScoringMethod = Form(
        "cosine",
        description="Primary scoring formula ('cosine' or 'coverage').",
    ),
) -> AnalyseResponse:
    """Analyze an uploaded resume PDF against a submitted job description."""
    filename = resume_file.filename or ""
    content_type = (resume_file.content_type or "").strip().lower()
    if (
        not filename.lower().endswith(".pdf")
        or content_type not in _ALLOWED_PDF_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files (.pdf) are supported.",
        )

    file_bytes = await resume_file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF file is empty.",
        )
    if len(file_bytes) > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=400,
            detail="PDF file size exceeds the 5 MB limit.",
        )

    if len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Job description is too short. Please provide at least 20 characters.",
        )

    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract readable text from the uploaded PDF. Ensure it is "
                "not password-protected or a scanned image."
            ),
        ) from exc
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract readable text from the uploaded PDF. Ensure it is "
                "not password-protected or a scanned image."
            ),
        )

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)
    if not cleaned_resume.strip():
        raise HTTPException(
            status_code=400,
            detail="Resume contains no valid text after cleaning.",
        )
    if not cleaned_jd.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description contains no valid text after cleaning.",
        )

    cosine_score = compute_score(cleaned_resume, cleaned_jd, method="cosine")
    coverage_score = compute_score(cleaned_resume, cleaned_jd, method="coverage")
    if scoring_method == "cosine":
        primary_score = cosine_score
        secondary_score = coverage_score
    else:
        primary_score = coverage_score
        secondary_score = cosine_score

    gap_report = analyze_keyword_gaps(cleaned_resume, cleaned_jd)
    matched_skills = gap_report.get("matched_skills", [])
    missing_skills = gap_report.get("missing_skills", [])
    missing_keywords = gap_report.get("missing_keywords", [])
    matched_skill_count = len(matched_skills)
    jd_skill_count = matched_skill_count + len(missing_skills)

    return AnalyseResponse(
        score=primary_score,
        secondary_score=secondary_score,
        scoring_method=scoring_method,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        missing_keywords=missing_keywords,
        jd_skill_count=jd_skill_count,
        matched_skill_count=matched_skill_count,
        resume_char_count=len(resume_text),
        jd_char_count=len(job_description),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
