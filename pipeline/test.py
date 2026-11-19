from pathlib import Path
import sys

# Ensure VyaparDock_Backend/ (parent of pipeline/) is on sys.path so `import main` works
PIPELINE_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = PIPELINE_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient
from main import app
from pydantic import ValidationError
from step1_pdfExtractor import extract_text_from_pdf
from step2_textCleaner import clean_text
from step3_scorer import compute_score
from step4_keywordGap import analyze_keyword_gaps
from step5_schemas import AnalyseResponse, ErrorResponse, HealthResponse

sys.stdout.reconfigure(encoding="utf-8")

CV_PATH1 = "VyaparDock_Backend/pipeline/testcv/Resume Anshuman updated.pdf"
CV_PATH2 = "VyaparDock_Backend/pipeline/testcv/23CHB0A03-CV.pdf"

JD_TEXT = """
We are hiring a Data Analyst Intern. You will analyse business data using Python and SQL,
build interactive dashboards in Power BI, and present insights to stakeholders.

Requirements:
- Strong skills in Python (Pandas, NumPy, Matplotlib) and SQL (PostgreSQL or MySQL)
- Experience with Power BI or Tableau for dashboards
- Familiarity with scikit-learn and basic machine learning
- Excel for data cleaning and reporting
- Good communication and analytical thinking
- Git and GitHub for version control
"""


## Step 1: Extract the text from the PDF
with open(CV_PATH1, "rb") as f:
    file_bytes1 = f.read()

resume_text = extract_text_from_pdf(file_bytes1)
assert isinstance(resume_text, str) and resume_text.strip(), "Step 1 failed: no text extracted"

print("===== STEP 1: EXTRACTED TEXT =====")
print(f"chars: {len(resume_text)}")
print(resume_text)

# with open(CV_PATH2, "rb") as f:
#     file_bytes2 = f.read()

# resume_text = extract_text_from_pdf(file_bytes2)
# assert isinstance(resume_text, str) and resume_text.strip(), "Step 1 failed: no text extracted"

# print("===== STEP 1: EXTRACTED TEXT =====")
# print(f"chars: {len(resume_text)}")
# print(resume_text)

## Step 2: Clean the resume text and the job description with the same function
cleaned_resume = clean_text(resume_text)
cleaned_jd = clean_text(JD_TEXT)

assert isinstance(cleaned_resume, str) and cleaned_resume.strip(), "Step 2 failed: cleaned resume is empty"
assert isinstance(cleaned_jd, str) and cleaned_jd.strip(), "Step 2 failed: cleaned JD is empty"

print("\n===== STEP 2: CLEANED TEXT =====")
print(f"resume tokens: {len(cleaned_resume.split())}")
print(cleaned_resume)
print(f"\nJD tokens: {len(cleaned_jd.split())}")
print(cleaned_jd)

# import re

# for label, source, cleaned in [("resume", resume_text, cleaned_resume), ("JD", JD_TEXT, cleaned_jd)]:
#     baseline = set(re.findall(r"[a-z0-9]+", source.lower()))
#     unknown = sorted(t for t in set(cleaned.split()) if t not in baseline)
#     print(f"{label}: tokens not found as whole words in the raw text -> {unknown}")

## Step 3: Score the cleaned resume against the cleaned job description
score = compute_score(cleaned_resume, cleaned_jd)

assert isinstance(score, (int, float)), "Step 3 failed: score is not a number"
assert 0 <= score <= 10, "Step 3 failed: score is outside 0-10"

print("\n===== STEP 3: SCORE =====")
print(f"score (0-10): {score}")

## Step 4: Identify matched skills, missing skills, and missing keywords
gap_report = analyze_keyword_gaps(cleaned_resume, cleaned_jd)

assert isinstance(gap_report, dict), "Step 4 failed: output is not a dict"
for key in ("matched_skills", "missing_skills", "missing_keywords"):
    assert key in gap_report and isinstance(gap_report[key], list), f"Step 4 failed: invalid '{key}'"

print("\n===== STEP 4: KEYWORD & SKILL GAP =====")
print(f"matched_skills ({len(gap_report['matched_skills'])}): {gap_report['matched_skills']}")
print(f"missing_skills ({len(gap_report['missing_skills'])}): {gap_report['missing_skills']}")
print(f"missing_keywords ({len(gap_report['missing_keywords'])}): {gap_report['missing_keywords']}")


## Step 5: Validate Pydantic schemas against pipeline outputs
def test_step5_schemas(
    primary_score: float,
    secondary_score: float,
    gaps: dict[str, list[str]],
    raw_resume: str,
    raw_jd: str,
) -> None:
    matched = gaps["matched_skills"]
    missing = gaps["missing_skills"]

    response = AnalyseResponse(
        score=primary_score,
        secondary_score=secondary_score,
        scoring_method="cosine",
        matched_skills=matched,
        missing_skills=missing,
        missing_keywords=gaps["missing_keywords"],
        jd_skill_count=len(matched) + len(missing),
        matched_skill_count=len(matched),
        resume_char_count=len(raw_resume),
        jd_char_count=len(raw_jd),
    )

    serialized = response.model_dump()
    assert serialized["score"] == primary_score, "Step 5 failed: score mismatch"
    assert serialized["secondary_score"] == secondary_score, "Step 5 failed: secondary_score mismatch"
    assert serialized["jd_skill_count"] == len(matched) + len(missing), "Step 5 failed: jd_skill_count mismatch"
    assert serialized["matched_skill_count"] == len(matched), "Step 5 failed: matched_skill_count mismatch"

    health = HealthResponse(status="ok", skills_loaded=506)
    assert health.status == "ok" and health.skills_loaded == 506, "Step 5 failed: HealthResponse invalid"

    error = ErrorResponse(detail="Job description contains no recognised skills")
    assert error.detail, "Step 5 failed: ErrorResponse invalid"

    # Verify strict boundary enforcement (score > 10.0 must fail)
    try:
        AnalyseResponse(
            score=10.5,
            jd_skill_count=1,
            matched_skill_count=1,
        )
        raise AssertionError("Step 5 failed: AnalyseResponse accepted score > 10.0")
    except ValidationError:
        pass

    print("\n===== STEP 5: PYDANTIC SCHEMAS VALIDATED =====")
    print(response.model_dump_json(indent=2))


coverage_score = compute_score(cleaned_resume, cleaned_jd, method="coverage")
test_step5_schemas(score, coverage_score, gap_report, resume_text, JD_TEXT)


## Step 6: End-to-End FastAPI HTTP Verification via TestClient
def test_step6_fastapi_endpoints(pdf_bytes: bytes, jd_text: str) -> None:
    client = TestClient(app)

    # 1. Verify GET /health
    health_res = client.get("/health")
    assert health_res.status_code == 200, f"Step 6 failed: /health returned {health_res.status_code}"
    health_json = health_res.json()
    assert health_json["status"] == "ok", "Step 6 failed: /health status != 'ok'"
    assert health_json["skills_loaded"] > 0, "Step 6 failed: skills_loaded <= 0"

    # 2. Verify POST /analyse (default cosine mode)
    cosine_res = client.post(
        "/analyse",
        files={"resume_file": ("resume.pdf", pdf_bytes, "application/pdf")},
        data={"job_description": jd_text, "scoring_method": "cosine"},
    )
    assert cosine_res.status_code == 200, f"Step 6 failed: /analyse (cosine) returned {cosine_res.status_code}: {cosine_res.text}"
    cosine_json = cosine_res.json()
    assert cosine_json["scoring_method"] == "cosine"
    assert cosine_json["score"] == score
    assert cosine_json["secondary_score"] == coverage_score

    # 3. Verify POST /analyse (coverage mode swaps primary and secondary scores)
    cov_res = client.post(
        "/analyse",
        files={"resume_file": ("resume.pdf", pdf_bytes, "application/pdf")},
        data={"job_description": jd_text, "scoring_method": "coverage"},
    )
    assert cov_res.status_code == 200, f"Step 6 failed: /analyse (coverage) returned {cov_res.status_code}"
    cov_json = cov_res.json()
    assert cov_json["scoring_method"] == "coverage"
    assert cov_json["score"] == coverage_score
    assert cov_json["secondary_score"] == score

    # 4. Verify 400 error on non-PDF upload
    bad_file_res = client.post(
        "/analyse",
        files={"resume_file": ("resume.txt", b"plain text", "text/plain")},
        data={"job_description": jd_text},
    )
    assert bad_file_res.status_code == 400, f"Step 6 failed: expected 400 for .txt file, got {bad_file_res.status_code}"

    # 5. Verify 400 error on overly short JD
    short_jd_res = client.post(
        "/analyse",
        files={"resume_file": ("resume.pdf", pdf_bytes, "application/pdf")},
        data={"job_description": "Too short"},
    )
    assert short_jd_res.status_code == 400, f"Step 6 failed: expected 400 for short JD, got {short_jd_res.status_code}"

    # 6. Verify CORS header for http://localhost:5173
    cors_res = client.options(
        "/analyse",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert cors_res.headers.get("access-control-allow-origin") == "http://localhost:5173", (
        "Step 6 failed: CORS origin header missing for http://localhost:5173"
    )

    print("\n===== STEP 6: FASTAPI ENDPOINTS VALIDATED =====")
    print(f"/health -> {health_json}")
    print(f"/analyse (cosine) -> score={cosine_json['score']}, secondary={cosine_json['secondary_score']}")
    print(f"/analyse (coverage) -> score={cov_json['score']}, secondary={cov_json['secondary_score']}")
    print("Error handling (400s) & CORS (http://localhost:5173) -> PASSED")


test_step6_fastapi_endpoints(file_bytes1, JD_TEXT)
