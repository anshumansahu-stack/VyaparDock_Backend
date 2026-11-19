"""Find skill and contextual-keyword gaps between cleaned resume and JD text.

Skill recognition is delegated to :mod:`scorer`; this module only formats
and compares the recognized skills and ranks non-skill terms from the JD.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Sequence

try:
    from step3_scorer import APPROVED_SKILLS, extract_skills
except ImportError:  # Support direct execution as ``python keyword_gap.py``.
    from step3_scorer import APPROVED_SKILLS, extract_skills

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # Use a deterministic standard-library fallback if absent.
    TfidfVectorizer = None  # type: ignore[assignment,misc]


_TOKEN_RE = re.compile(r"(?u)\b\w+\b")
BOILERPLATE_NON_SKILLS: frozenset[str] = frozenset(
    {
        "hiring",
        "intern",
        "internship",
        "candidate",
        "applicant",
        "role",
        "position",
        "team",
        "work",
        "working",
        "experience",
        "requirements",
        "responsibilities",
    }
)
_FILLER_BIGRAM_TOKENS: frozenset[str] = frozenset(
    {
        "using",
        "present",
        "build",
        "experience",
        "good",
        "strong",
        "basic",
        "familiarity",
        "hiring",
        "intern",
        "requirements",
        "skills",
    }
)


def _surface_form_for_skill(skill: str, cleaned_jd: str) -> str:
    """Use a spaced JD span for display when it maps to a canonical skill key."""
    tokens = cleaned_jd.split()
    spaced_matches: list[tuple[int, int, str]] = []
    for start in range(len(tokens)):
        for width in (3, 2):
            if start + width > len(tokens):
                continue
            span = tokens[start : start + width]
            if "".join(span).casefold() == skill.casefold():
                spaced_matches.append((start, -width, " ".join(span)))
    if spaced_matches:
        return min(spaced_matches)[2]
    return skill


def _skill_surface_tokens(cleaned_jd: str, skills: set[str]) -> set[str]:
    """Return component tokens from spaced surface forms of recognized skills."""
    components: set[str] = set()
    for skill in skills:
        surface = _surface_form_for_skill(skill, cleaned_jd)
        components.update(token.casefold() for token in surface.split())
    return components


def _first_appearance_index(term: str, cleaned_jd: str) -> int:
    """Return the first JD token offset for a unigram or contiguous n-gram."""
    tokens = [token.casefold() for token in _TOKEN_RE.findall(cleaned_jd)]
    phrase = tuple(token.casefold() for token in term.split())
    width = len(phrase)
    for start in range(max(0, len(tokens) - width + 1)):
        if tuple(tokens[start : start + width]) == phrase:
            return start
    return len(tokens)


def _contains_ngram(tokens: Sequence[str], ngram: Sequence[str]) -> bool:
    """Return whether an exact contiguous token sequence occurs in ``tokens``."""
    width = len(ngram)
    if width == 0 or width > len(tokens):
        return False
    return any(
        tuple(tokens[start : start + width]) == tuple(ngram)
        for start in range(len(tokens) - width + 1)
    )


def _fallback_tfidf_candidates(
    cleaned_resume: str,
    cleaned_jd: str,
) -> list[tuple[str, float]]:
    """Rank JD unigrams and bigrams with smoothed two-document TF-IDF."""
    documents = [
        _TOKEN_RE.findall(cleaned_resume),
        _TOKEN_RE.findall(cleaned_jd),
    ]
    document_terms: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        terms: Counter[str] = Counter()
        for width in (1, 2):
            for start in range(len(tokens) - width + 1):
                terms[" ".join(tokens[start : start + width])] += 1
        document_terms.append(terms)
        document_frequency.update(terms.keys())

    jd_terms = document_terms[1]
    if not jd_terms:
        return []
    idf = {
        term: math.log((1.0 + len(documents)) / (1.0 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }
    jd_weights = {
        term: (1.0 + math.log(frequency)) * idf[term]
        for term, frequency in jd_terms.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in jd_weights.values()))
    if norm:
        jd_weights = {term: weight / norm for term, weight in jd_weights.items()}
    return sorted(
        jd_weights.items(),
        key=lambda item: (
            -item[1],
            len(item[0].split()),
            _first_appearance_index(item[0], cleaned_jd),
        ),
    )


def _rank_jd_candidates(
    cleaned_resume: str,
    cleaned_jd: str,
) -> list[tuple[str, float]]:
    """Fit TF-IDF to resume and JD, returning the JD row's ranked features."""
    if TfidfVectorizer is None:
        return _fallback_tfidf_candidates(cleaned_resume, cleaned_jd)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
    )
    try:
        matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
    except ValueError:
        # An input with no word tokens has no vocabulary to rank.
        return []

    feature_names = vectorizer.get_feature_names_out()
    jd_row = matrix[1].tocoo()
    candidates = [
        (str(feature_names[index]), float(weight))
        for index, weight in zip(jd_row.col, jd_row.data)
    ]
    return sorted(
        candidates,
        key=lambda item: (
            -item[1],
            len(item[0].split()),
            _first_appearance_index(item[0], cleaned_jd),
        ),
    )


def _is_skill_related_candidate(
    term: str,
    skill_component_tokens: set[str],
) -> bool:
    """Exclude an approved skill or a keyword overlapping its spaced parts."""
    if term.replace(" ", "").casefold() in APPROVED_SKILLS:
        return True
    return bool({token.casefold() for token in term.split()} & skill_component_tokens)


def _is_boilerplate_skill(skill: str, surface: str) -> bool:
    """Identify generic recruitment words accidentally included as skills."""
    canonical = skill.replace(" ", "").casefold()
    surface_tokens = [token.casefold() for token in surface.split()]
    return canonical in BOILERPLATE_NON_SKILLS or (
        len(surface_tokens) == 1 and surface_tokens[0] in BOILERPLATE_NON_SKILLS
    )


def _rank_missing_keywords(
    cleaned_resume: str,
    cleaned_jd: str,
    skills: set[str],
    top_k_keywords: int,
) -> list[str]:
    """Select absent, non-skill JD terms while collapsing redundant phrases."""
    if top_k_keywords == 0:
        return []

    resume_tokens = [token.casefold() for token in _TOKEN_RE.findall(cleaned_resume)]
    skill_component_tokens = _skill_surface_tokens(cleaned_jd, skills)
    selected_scores: dict[str, float] = {}

    for term, score in _rank_jd_candidates(cleaned_resume, cleaned_jd):
        phrase_tokens = tuple(token.casefold() for token in term.split())
        if len(phrase_tokens) == 2 and any(
            token in _FILLER_BIGRAM_TOKENS for token in phrase_tokens
        ):
            continue
        if len(phrase_tokens) == 1 and phrase_tokens[0] in (
            BOILERPLATE_NON_SKILLS | _FILLER_BIGRAM_TOKENS
        ):
            continue
        if _is_skill_related_candidate(term, skill_component_tokens):
            continue
        if _contains_ngram(resume_tokens, phrase_tokens):
            continue

        if len(phrase_tokens) == 1:
            token = phrase_tokens[0]
            if any(
                score_for_bigram >= score and token in bigram.split()
                for bigram, score_for_bigram in selected_scores.items()
                if " " in bigram
            ):
                continue
        else:
            for token in phrase_tokens:
                if token in selected_scores and score > selected_scores[token]:
                    del selected_scores[token]

        selected_scores[term] = score

    return list(selected_scores)[:top_k_keywords]


def analyze_keyword_gaps(
    cleaned_resume: str,
    cleaned_jd: str,
    top_k_keywords: int = 15,
) -> dict[str, list[str]]:
    """Return missing/matched skills and missing contextual JD keywords.

    Skill identification and ordering of skill occurrences come directly from
    :func:`extract_skills` imported from ``scorer``. JD skills are ranked by
    occurrence count, and displayed using spaced forms found in the JD.

    Args:
        cleaned_resume: Normalized resume text.
        cleaned_jd: Normalized job-description text.
        top_k_keywords: Maximum number of contextual terms to return.

    Returns:
        A dictionary with ``missing_skills``, ``matched_skills``, and
        ``missing_keywords`` lists.

    Raises:
        TypeError: If either text input is not a string or ``top_k_keywords``
            is not a non-negative integer.
    """
    if not isinstance(cleaned_resume, str) or not isinstance(cleaned_jd, str):
        raise TypeError("cleaned_resume and cleaned_jd must both be strings")
    if (
        isinstance(top_k_keywords, bool)
        or not isinstance(top_k_keywords, int)
        or top_k_keywords < 0
    ):
        raise TypeError("top_k_keywords must be a non-negative integer")

    empty_result = {
        "missing_skills": [],
        "matched_skills": [],
        "missing_keywords": [],
    }
    if not cleaned_jd.strip():
        return empty_result

    jd_skills = extract_skills(cleaned_jd)
    resume_skills = extract_skills(cleaned_resume)
    resume_skill_set = set(resume_skills)
    jd_skill_counts = Counter(jd_skills)
    first_appearance: dict[str, int] = {}
    for index, skill in enumerate(jd_skills):
        first_appearance.setdefault(skill, index)
    ordered_unique_skills = sorted(
        jd_skill_counts,
        key=lambda skill: (-jd_skill_counts[skill], first_appearance[skill]),
    )

    skill_surfaces = {
        skill: _surface_form_for_skill(skill, cleaned_jd)
        for skill in ordered_unique_skills
    }
    non_boilerplate_missing_skills = [
        skill
        for skill in ordered_unique_skills
        if not _is_boilerplate_skill(skill, skill_surfaces[skill])
    ]
    missing_skills = [
        skill_surfaces[skill]
        for skill in non_boilerplate_missing_skills
        if skill not in resume_skill_set
    ]
    matched_skills = [
        skill_surfaces[skill]
        for skill in ordered_unique_skills
        if skill in resume_skill_set
    ]
    skills_for_keyword_filter = {
        skill
        for skill in ordered_unique_skills
        if skill in resume_skill_set or skill in non_boilerplate_missing_skills
    }
    missing_keywords = _rank_missing_keywords(
        cleaned_resume,
        cleaned_jd,
        skills_for_keyword_filter,
        top_k_keywords,
    )
    return {
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "missing_keywords": missing_keywords,
    }


def _run_self_tests() -> None:
    """Run deterministic behavior checks when the taxonomy is available."""
    exact = analyze_keyword_gaps(
        "python sql power bi tableau version control",
        "python sql power bi tableau version control",
    )
    assert exact["missing_skills"] == []
    assert exact["missing_keywords"] == []

    partial = analyze_keyword_gaps(
        "python sql",
        "python sql power bi tableau version control",
    )
    assert {"python", "sql"}.issubset(set(partial["matched_skills"]))
    if "powerbi" in APPROVED_SKILLS:
        assert "power bi" in partial["missing_skills"] or "powerbi" in partial[
            "missing_skills"
        ]
    if "tableau" in APPROVED_SKILLS:
        assert "tableau" in partial["missing_skills"]
    assert "version control" in partial["missing_keywords"]
    assert "version control" not in partial["missing_skills"]

    disjoint = analyze_keyword_gaps("banana apple", "python sql")
    assert disjoint["missing_skills"] == [
        _surface_form_for_skill(skill, "python sql")
        for skill in extract_skills("python sql")
    ]
    assert isinstance(disjoint["missing_keywords"], list)

    boilerplate = analyze_keyword_gaps("python sql", "hiring python sql")
    assert "hiring" not in boilerplate["missing_skills"]


def _read_cli_text(value: str | None, file_path: Path | None, label: str) -> str | None:
    """Return supplied text or read cleaned text from a UTF-8 file."""
    if value is not None:
        return value
    if file_path is None:
        return None
    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read {label} text from {file_path}: {exc}") from exc


def _main(argv: list[str] | None = None) -> int:
    """Run self-tests or inspect keyword gaps for supplied cleaned text."""
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
    parser.add_argument("--top-k-keywords", type=int, default=15)
    args = parser.parse_args(argv)

    _run_self_tests()
    if not any((args.resume is not None, args.resume_file, args.jd is not None, args.jd_file)):
        print("keyword_gap.py self-tests passed")
        return 0

    try:
        resume_text = _read_cli_text(args.resume, args.resume_file, "resume")
        jd_text = _read_cli_text(args.jd, args.jd_file, "job description")
        if resume_text is None or jd_text is None:
            parser.error("provide both --resume and --jd (or both file options)")
        result = analyze_keyword_gaps(
            resume_text,
            jd_text,
            top_k_keywords=args.top_k_keywords,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
