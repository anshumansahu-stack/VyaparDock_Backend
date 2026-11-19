"""Deterministic, taxonomy-based resume-to-job-description scoring.

Skill recognition is deliberately limited to the approved terms loaded from
``skill_terms.json`` and its optional sibling extension file. Functions in this
module do not print, access the network, or use randomness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Sequence

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # Keep module importable in minimal isolated environments.
    TfidfVectorizer = None  # type: ignore[assignment,misc]
    cosine_similarity = None  # type: ignore[assignment]


_MODULE_DIR = Path(__file__).resolve().parent
_SKILL_TERMS_PATH = _MODULE_DIR.parent / "pipeline" / "skillWords" / "skill_terms.json"
_EXTRA_SKILL_TERMS_PATH = _MODULE_DIR / "skill_terms_extra.json"


def _read_skill_file(path: Path, *, required: bool) -> frozenset[str]:
    """Read and validate one JSON array of normalized, space-free skill terms."""
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if not required:
            return frozenset()
        raise FileNotFoundError(
            f"Required skill taxonomy file is missing; expected it at: {path}"
        ) from exc

    try:
        entries = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in skill taxonomy file {path}: {exc.msg}"
        ) from exc

    if not isinstance(entries, list):
        raise ValueError(f"Skill taxonomy file must contain a JSON array: {path}")

    normalized: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, str):
            raise ValueError(
                f"Skill entry {index} in {path} must be a string, "
                f"not {type(entry).__name__}."
            )
        skill = entry.strip().lower()
        if not skill:
            continue
        if any(character.isspace() for character in skill):
            raise ValueError(
                f"Skill entry {index} in {path} contains whitespace; "
                "approved skill terms must be stored without spaces."
            )
        normalized.add(skill)
    return frozenset(normalized)


def _load_approved_skills() -> frozenset[str]:
    """Load the required taxonomy and merge optional custom skill additions."""
    primary = _read_skill_file(_SKILL_TERMS_PATH, required=True)
    additional = _read_skill_file(_EXTRA_SKILL_TERMS_PATH, required=False)
    return frozenset(primary | additional)


_SELF_TEST_SKILLS = frozenset(
    {
        "amazonwebservices",
        "python",
        "sql",
        "git",
        "github",
        "powerbi",
        "scikitlearn",
    }
)

if _SKILL_TERMS_PATH.is_file():
    APPROVED_SKILLS = _load_approved_skills()
elif __name__ == "__main__":
    # The CLI self-test can run in an isolated checkout without shipping a
    # taxonomy file. Normal imports still fail loudly when the required list is
    # missing, as they must for the application.
    APPROVED_SKILLS = _SELF_TEST_SKILLS
else:
    APPROVED_SKILLS = _load_approved_skills()


def extract_skills(cleaned_text: str) -> list[str]:
    """Extract approved skills from cleaned text in document order.

    Matching is greedy and longest-first over windows of up to three tokens.
    Repeated occurrences are retained so downstream weighting can use them.

    Args:
        cleaned_text: Lowercase, whitespace-tokenized text from an earlier
            cleaning stage.

    Returns:
        Approved skill terms in the order they occur, including duplicates.

    Raises:
        TypeError: If ``cleaned_text`` is not a string.
    """
    if not isinstance(cleaned_text, str):
        raise TypeError("cleaned_text must be a string")

    tokens = cleaned_text.split()
    skills: list[str] = []
    index = 0
    while index < len(tokens):
        matched = False
        for window_size in (3, 2, 1):
            if index + window_size > len(tokens):
                continue
            candidate = "".join(tokens[index : index + window_size])
            if candidate in APPROVED_SKILLS:
                skills.append(candidate)
                index += window_size
                matched = True
                break
        if not matched:
            index += 1
    return skills


def _cosine_score(resume_skills: list[str], jd_skills: list[str]) -> float:
    """Return TF-IDF cosine similarity for two ordered skill lists."""
    resume_text = " ".join(resume_skills)
    jd_text = " ".join(jd_skills)
    if not resume_text or not jd_text:
        return 0.0

    if TfidfVectorizer is not None and cosine_similarity is not None:
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
        )
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        return float(cosine_similarity(vectors[0], vectors[1])[0, 0])

    # Minimal-installation fallback with the same default smoothed IDF,
    # raw term frequency, and L2 normalization used by TfidfVectorizer.
    documents = (resume_text.split(), jd_text.split())
    document_frequency: Counter[str] = Counter(
        term for document in documents for term in set(document)
    )
    idf = {
        term: math.log((1.0 + len(documents)) / (1.0 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }

    def vectorize(document: Sequence[str]) -> dict[str, float]:
        term_frequency = Counter(document)
        return {
            term: count * idf[term]
            for term, count in term_frequency.items()
        }

    resume_vector = vectorize(documents[0])
    jd_vector = vectorize(documents[1])
    resume_norm = math.sqrt(sum(value * value for value in resume_vector.values()))
    jd_norm = math.sqrt(sum(value * value for value in jd_vector.values()))
    if resume_norm == 0.0 or jd_norm == 0.0:
        return 0.0
    dot_product = sum(
        value * jd_vector.get(term, 0.0)
        for term, value in resume_vector.items()
    )
    return float(dot_product / (resume_norm * jd_norm))


def _coverage_score(resume_skills: list[str], jd_skills: list[str]) -> float:
    """Return frequency-weighted JD skill coverage in the resume."""
    jd_counts = Counter(jd_skills)
    if not jd_counts:
        return 0.0

    resume_skill_set = set(resume_skills)
    weights = {
        skill: 1.0 + math.log(count)
        for skill, count in jd_counts.items()
    }
    total_weight = sum(weights.values())
    matched_weight = sum(
        weight for skill, weight in weights.items() if skill in resume_skill_set
    )
    return float(matched_weight / total_weight) if total_weight else 0.0


def compute_score(
    cleaned_resume: str,
    cleaned_jd: str,
    method: str = "cosine",
) -> float:
    """Score a cleaned resume against a cleaned job description on a 0–10 scale.

    Args:
        cleaned_resume: Normalized resume text.
        cleaned_jd: Normalized job-description text.
        method: Either ``"cosine"`` for TF-IDF similarity or ``"coverage"``
            for frequency-weighted coverage of recognized JD skills.

    Returns:
        A score clamped to the inclusive range 0.0–10.0 and rounded to one
        decimal place.

    Raises:
        TypeError: If either text argument is not a string.
        ValueError: If the method is unknown or the job description contains
            no recognized skills.
    """
    if not isinstance(cleaned_resume, str) or not isinstance(cleaned_jd, str):
        raise TypeError("cleaned_resume and cleaned_jd must both be strings")
    if method not in ("cosine", "coverage"):
        raise ValueError(
            f"Unknown scoring method: {method!r}. Expected 'cosine' or 'coverage'."
        )

    jd_skills = extract_skills(cleaned_jd)
    if not jd_skills:
        raise ValueError("job description contains no recognised skills")
    resume_skills = extract_skills(cleaned_resume)
    if not resume_skills:
        return 0.0

    raw_score = (
        _cosine_score(resume_skills, jd_skills)
        if method == "cosine"
        else _coverage_score(resume_skills, jd_skills)
    )
    return round(max(0.0, min(10.0, raw_score * 10.0)), 1)


def _run_self_tests() -> None:
    """Run deterministic API checks used by the standalone CLI entry point."""
    assert extract_skills("amazon web services python") == [
        "amazonwebservices",
        "python",
    ]
    assert compute_score("python sql", "python sql", method="cosine") == 10.0
    assert compute_score("python sql", "python sql", method="coverage") == 10.0
    assert compute_score("python", "amazon web services") == 0.0
    assert compute_score(
        "python sql git github", "python sql", method="coverage"
    ) == 10.0
    assert compute_score(
        "python sql git github", "python sql", method="cosine"
    ) < 10.0
    try:
        compute_score("python sql", "banana apple")
    except ValueError as exc:
        assert str(exc) == "job description contains no recognised skills"
    else:
        raise AssertionError("JD without approved skills should raise ValueError")
    assert compute_score("banana", "python sql") == 0.0


def _read_cli_text(value: str | None, file_path: Path | None, label: str) -> str | None:
    """Return CLI text directly or load it from an explicitly supplied file."""
    if value is not None:
        return value
    if file_path is None:
        return None
    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read {label} text from {file_path}: {exc}") from exc


def _main(argv: list[str] | None = None) -> int:
    """Run the self-test or score cleaned text supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", help="Cleaned resume text")
    resume_group.add_argument(
        "--resume-file", type=Path, help="UTF-8 file containing cleaned resume text"
    )
    jd_group = parser.add_mutually_exclusive_group()
    jd_group.add_argument("--jd", help="Cleaned job-description text")
    jd_group.add_argument(
        "--jd-file", type=Path, help="UTF-8 file containing cleaned JD text"
    )
    args = parser.parse_args(argv)

    _run_self_tests()
    if not any((args.resume is not None, args.resume_file, args.jd is not None, args.jd_file)):
        print("scorer.py self-tests passed")
        return 0

    try:
        resume_text = _read_cli_text(args.resume, args.resume_file, "resume")
        jd_text = _read_cli_text(args.jd, args.jd_file, "job description")
        if resume_text is None or jd_text is None:
            parser.error("provide both --resume and --jd (or both file options)")
        print(f"cosine: {compute_score(resume_text, jd_text, method='cosine'):.1f}/10")
        print(f"coverage: {compute_score(resume_text, jd_text, method='coverage'):.1f}/10")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
