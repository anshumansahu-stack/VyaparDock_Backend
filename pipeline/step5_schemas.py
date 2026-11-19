"""Standalone Pydantic response schemas for the Karmanometer API.

The FastAPI ``main.py`` module uses :class:`AnalyseResponse` for successful
analysis responses, :class:`HealthResponse` for the health endpoint, and
:class:`ErrorResponse` to document human-readable HTTP error details. The
``/analyse`` request remains multipart form data and is intentionally not
represented by a JSON request model here.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ScoringMethod = Literal["cosine", "coverage"]
"""Supported scoring formulas selectable by the API client."""


class AnalyseResponse(BaseModel):
    """Validated JSON payload returned after resume-to-JD analysis.

    ``main.py`` constructs this model after parsing the uploaded resume and
    submitted job description. It includes the primary and alternate scores,
    skill and keyword gaps, and optional input-size metadata for the UI.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Primary resume-to-JD match score on a 0.0 to 10.0 scale.",
    )
    secondary_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description=(
            "Alternate formula match score (0.0 to 10.0) so the UI can toggle "
            "between cosine and coverage without re-uploading."
        ),
    )
    scoring_method: ScoringMethod = Field(
        default="cosine",
        description="Formula used to compute the primary score ('cosine' or 'coverage').",
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Approved skills found in both the resume and the job description.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Approved skills required by the job description that are missing "
            "from the resume."
        ),
    )
    missing_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Important non-skill contextual terms from the job description "
            "missing from the resume."
        ),
    )
    jd_skill_count: int = Field(
        ...,
        ge=0,
        description="Total number of recognized skills extracted from the job description.",
    )
    matched_skill_count: int = Field(
        ...,
        ge=0,
        description="Number of recognized job description skills matched in the resume.",
    )
    resume_char_count: int | None = Field(
        default=None,
        ge=0,
        description="Character count of extracted resume text.",
    )
    jd_char_count: int | None = Field(
        default=None,
        ge=0,
        description="Character count of the submitted job description text.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "score": 7.4,
                "secondary_score": 6.7,
                "scoring_method": "cosine",
                "matched_skills": ["python", "sql"],
                "missing_skills": ["docker"],
                "missing_keywords": ["pipelines", "stakeholders"],
                "jd_skill_count": 3,
                "matched_skill_count": 2,
                "resume_char_count": 3420,
                "jd_char_count": 1850,
            }
        }
    )


class HealthResponse(BaseModel):
    """Health endpoint payload returned by ``main.py``."""

    status: str = Field(default="ok", description="Service health status.")
    skills_loaded: int = Field(
        ...,
        ge=0,
        description="Total count of active skills loaded into memory.",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ok", "skills_loaded": 512}}
    )


class ErrorResponse(BaseModel):
    """Documented error payload matching FastAPI HTTPException details."""

    detail: str = Field(
        ...,
        description="Human-readable error explanation matching FastAPI HTTPException responses.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"detail": "Uploaded file must be a valid PDF."}
        }
    )