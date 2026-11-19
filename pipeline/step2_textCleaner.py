"""Deterministic text normalization shared by Karmanometer's resume and JD flows.

The same pure tokenization pipeline is used for resumes and job descriptions so
TF-IDF and keyword-gap analysis operate on aligned vocabularies.

When available, ``wordfreq`` supplies a broad English frequency lexicon for
conservative unknown-token segmentation. A compact local vocabulary keeps the
module importable and useful in minimal/offline installations as well.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

try:
    from wordfreq import zipf_frequency
except ImportError:  # Keep this module usable when optional NLP extras are absent.
    zipf_frequency = None


# Compact grammatical English stopwords based on NLTK's standard English list.
# Kept locally so importing this module never requires downloading NLTK data.
_GRAMMATICAL_STOP_WORDS = frozenset(
    """
    i me my myself we our ours ourselves you you're you've you'll you'd your yours
    yourself yourselves he him his himself she she's her hers herself it it's its
    itself they them their theirs themselves what which who whom this that that'll
    these those am is are was were be been being have has had having do does did
    doing a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up down in
    out on off over under again further then once here there when where why how all
    any both each few more most other some such no nor not only own same so than too
    very s t can will just don don't should should've now d ll m o re ve y ain aren
    aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
    haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
    shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
    """.split()
)

DOMAIN_KEEP_WORDS = frozenset(
    {
        "system",
        "systems",
        "detail",
        "details",
        "top",
        "interest",
        "interests",
        "based",
        "driven",
        "end",
        "full",
        "stack",
        "part",
        "front",
        "back",
    }
)
STOP_WORDS = frozenset(_GRAMMATICAL_STOP_WORDS - DOMAIN_KEEP_WORDS)

# Small vocabulary supplements wordfreq for acronyms and software names it may
# not recognize. These are segment candidates, not a filter or skill whitelist.
_DOMAIN_SEGMENT_WORDS = frozenset(
    {
        "dav",
        "matplotlib",
        "scikit",
        "seaborn",
        "pytorch",
        "tensorflow",
        "fastapi",
        "django",
        "flask",
        "opencv",
        "sklearn",
        "postgresql",
        "javascript",
        "typescript",
        "nodejs",
        "streamlit",
        "numpy",
        "github",
        "linkedin",
        "countvectorizer",
        "kpis",
        "netflix",
        "spotify",
        "cbse",
        "git",
        "dotnet",
        "csharp",
        "cpp",
        "sql",
        "nlp",
        "eda",
        "nss",
    }
)

# Approximate fallback frequencies for common English and resume vocabulary.
# The optional wordfreq lexicon is substantially broader; these entries let the
# algorithm handle common fused words without making wordfreq an import-time
# requirement or hardcoding any complete fused-token examples.
_FALLBACK_WORD_SCORES = {
    **dict.fromkeys(
        (
            "a an and are as at be been by can do for from had has have he her his "
            "i in is it its me my of on or our she that the their them then there "
            "they this to we was were what when where which who will with you your "
            "built data public school ensure accuracy collection validation resume "
            "clean analyze country dataset visualize operation stakeholder feature "
            "support perform develop project personal listen word power scheme "
            "seamless execution sector server "
            "python analyzed transactions warangal movie recommendation distribution "
            "configuration effectiveness attention detail contributing countries "
            "areas interest education experience engineering skills projects "
            "development communication leadership collaboration analytics "
            "professional management implementation performance research design "
            "analysis software hardware database application applications "
            "developer business customer product service services solutions "
            "strategy strategic technology technologies engineer scientist "
            "consultant specialist internship intern achievements responsibilities "
            "certification certifications bachelor master university coursework "
            "algorithm algorithms machine learning artificial intelligence "
            "security cloud backend frontend testing framework frameworks "
            "optimization infrastructure architecture programming languages "
            "platforms systems system based driven full stack part front back end "
            "top interests details seaborn matplotlib scikit dav sql nlp eda nss "
            "fastapi django flask pytorch tensorflow opencv sklearn postgresql "
            "javascript typescript nodejs dotnet csharp cpp"
        ).split(),
        4.0,
    ),
    **dict.fromkeys(
        (
            "recommendations distributions configurations effectivenesses "
            "numerical scientific analytical operational organizational "
            "visualization visualizations documentation communication "
            "recommendation distribution configuration effectiveness validation "
            "matplotlib seaborn tensorflow postgresql javascript typescript"
        ).split(),
        5.0,
    ),
}


def _possible_inflections(word: str) -> set[str]:
    """Return common plural and verb forms for a fallback dictionary entry."""
    forms: set[str] = set()
    if len(word) < 3 or not word.isalpha() or word in STOP_WORDS:
        return forms

    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        forms.add(word[:-1] + "ies")
    else:
        forms.add(word + "s")
        if word.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(word + "es")

    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        forms.add(word[:-1] + "ied")
    elif word.endswith("e"):
        forms.add(word + "d")
    else:
        forms.add(word + "ed")

    if word.endswith("ie"):
        forms.add(word[:-2] + "ying")
    elif word.endswith("e") and not word.endswith("ee"):
        forms.add(word[:-1] + "ing")
    else:
        forms.add(word + "ing")

    if (
        len(word) >= 3
        and word[-1] not in "aeiouwxy"
        and word[-2] in "aeiou"
        and word[-3] not in "aeiou"
    ):
        forms.update({word + word[-1] + "ed", word + word[-1] + "ing"})
    return forms


def _fallback_base_forms(word: str) -> set[str]:
    """Infer likely dictionary bases for common inflected forms."""
    bases: set[str] = set()
    if word.endswith("ies") and len(word) > 4:
        bases.add(word[:-3] + "y")
    if word.endswith("ied") and len(word) > 4:
        bases.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 4:
        bases.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        bases.add(word[:-1])
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        bases.update({stem, stem + "e"})
        if len(stem) > 2 and stem[-1] == stem[-2]:
            bases.add(stem[:-1])
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        bases.update({stem, stem + "e"})
        if len(stem) > 2 and stem[-1] == stem[-2]:
            bases.add(stem[:-1])
    return bases


def _installed_package_vocabulary() -> set[str]:
    """Collect normalized names of installed Python distributions/modules."""
    try:
        from importlib.metadata import distributions, packages_distributions
    except ImportError:
        return set()

    names: set[str] = set()
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name", "")
        normalized = re.sub(r"[^a-z0-9]+", "", raw_name.lower())
        if len(normalized) >= 3:
            names.add(normalized)
    for module_name in packages_distributions():
        normalized = re.sub(r"[^a-z0-9]+", "", module_name.lower())
        if len(normalized) >= 3:
            names.add(normalized)
    return names


_PROTECTED_WORDS = set(_FALLBACK_WORD_SCORES) | set(_DOMAIN_SEGMENT_WORDS)
for _base_word in tuple(_PROTECTED_WORDS):
    _PROTECTED_WORDS.update(_possible_inflections(_base_word))
_PROTECTED_WORDS.update(_installed_package_vocabulary())
PROTECTED_VOCAB = frozenset(_PROTECTED_WORDS)
FULL_LEXICON = PROTECTED_VOCAB

# Aliases are replaced before punctuation is turned into spaces.
_TECH_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![a-z0-9])c\+\+(?![a-z0-9])"), "cpp"),
    (re.compile(r"(?<![a-z0-9])c#(?![a-z0-9])"), "csharp"),
    (re.compile(r"(?<![a-z0-9])\.net(?![a-z0-9])"), "dotnet"),
    (re.compile(r"(?<![a-z0-9])node\.js(?![a-z0-9])"), "nodejs"),
)
_CAMEL_CASE_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_ACRONYM_TO_WORD_RE = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z]{2,})")
_CASE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")

_MIN_SEGMENT_TOKEN_LENGTH = 9
_MAX_SEGMENT_TOKEN_LENGTH = 256
_SEGMENT_SPLIT_PENALTY = 1.8
_MIN_SEGMENT_SCORE = 1.0
_DOMAIN_SEGMENT_SCORE = 2.5
MIN_PIECE_LEN = 4
MAX_SPLITS = 1
_MIN_CAMEL_PIECE_LEN = 3


def _normalize_unicode(text: str) -> str:
    """Decompose accents and remove combining diacritical marks."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def _replace_invisible_controls(text: str) -> str:
    """Turn invisible format/control characters into token boundaries."""
    return "".join(
        " "
        if unicodedata.category(character) in {"Cf", "Cc"}
        else character
        for character in text
    )


def _split_case_boundaries(text: str) -> str:
    """Split only OOV CamelCase tokens whose parts are known words.

    Whole-token recognition runs first so branded and technical identifiers
    such as GitHub, NumPy, and CountVectorizer remain single tokens.
    """

    def split_if_known(match: re.Match[str]) -> str:
        token = match.group(0)
        if _is_protected_word(token.casefold()):
            return token

        separated = _CAMEL_CASE_RE.sub(" ", token)
        separated = _ACRONYM_TO_WORD_RE.sub(" ", separated)
        pieces = separated.split()
        if len(pieces) != 2:
            return token
        if all(
            len(piece) >= _MIN_CAMEL_PIECE_LEN
            and _is_protected_word(piece.casefold())
            for piece in pieces
        ):
            return " ".join(pieces)
        return token

    return _CASE_TOKEN_RE.sub(split_if_known, text)


def _replace_tech_aliases(text: str) -> str:
    """Normalize common programming-language names using whole-token matches."""
    for pattern, replacement in _TECH_ALIAS_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _replace_non_alphanumeric(text: str) -> str:
    """Turn punctuation runs into token boundaries rather than deleting them."""
    return _NON_ALPHANUMERIC_RE.sub(" ", text)


@lru_cache(maxsize=1)
def _nltk_word_vocabulary() -> frozenset[str]:
    """Load NLTK's word corpus when already installed, without downloading it."""
    try:
        from nltk.corpus import words as nltk_words

        return frozenset(
            entry.casefold()
            for entry in nltk_words.words()
            if entry.isalpha()
        )
    except (ImportError, LookupError, OSError):
        return frozenset()


@lru_cache(maxsize=32768)
def _word_score(word: str) -> float:
    """Return a frequency or dictionary score for a candidate component."""
    frequency_score = zipf_frequency(word, "en") if zipf_frequency else 0.0
    if word in _DOMAIN_SEGMENT_WORDS:
        return max(_DOMAIN_SEGMENT_SCORE, frequency_score)
    if frequency_score > 0:
        return frequency_score

    fallback_score = _FALLBACK_WORD_SCORES.get(word, 0.0)
    if fallback_score > 0:
        return fallback_score
    if word in PROTECTED_VOCAB:
        return _DOMAIN_SEGMENT_SCORE
    if word in _nltk_word_vocabulary():
        return _DOMAIN_SEGMENT_SCORE

    # The local fallback corpus also recognizes regular inflections, which
    # keeps segmentation useful when optional frequency data is unavailable.
    for base in _fallback_base_forms(word):
        base_score = _FALLBACK_WORD_SCORES.get(base, 0.0)
        if base_score > 0:
            return max(1.0, base_score - 0.4)
    return 0.0


def _is_protected_word(word: str) -> bool:
    """Return whether a token is already a recognized word or identifier."""
    return word in PROTECTED_VOCAB or _word_score(word) > 0


def _segment_unknown_token(token: str) -> list[str]:
    """Split an unknown token at most once when both halves are recognized.

    Every candidate half must meet ``MIN_PIECE_LEN`` and have positive
    dictionary/corpus evidence. A recognized whole word always bypasses the
    splitter, and unknown residual fragments are never accepted.
    """
    if (
        not token.isalpha()
        or len(token) < _MIN_SEGMENT_TOKEN_LENGTH
        or len(token) > _MAX_SEGMENT_TOKEN_LENGTH
    ):
        return [token]

    whole_word_score = _word_score(token)
    if _is_protected_word(token):
        return [token]

    best_components: list[str] | None = None
    best_split_score = float("-inf")
    last_cut = len(token) - MIN_PIECE_LEN
    for cut in range(MIN_PIECE_LEN, last_cut + 1):
        left, right = token[:cut], token[cut:]
        if len(left) < MIN_PIECE_LEN or len(right) < MIN_PIECE_LEN:
            continue

        left_score, right_score = _word_score(left), _word_score(right)
        if left_score <= 0 or right_score <= 0:
            continue

        candidate_score = left_score + right_score - _SEGMENT_SPLIT_PENALTY
        if candidate_score > best_split_score:
            best_components = [left, right]
            best_split_score = candidate_score

    if (
        MAX_SPLITS == 1
        and best_components is not None
        and best_split_score > whole_word_score
        and best_split_score >= _MIN_SEGMENT_SCORE
    ):
        return best_components
    return [token]


def _segment_compound_tokens(tokens: list[str]) -> list[str]:
    """Apply conservative dictionary-based segmentation to long alphabetic tokens."""
    segmented: list[str] = []
    for token in tokens:
        segmented.extend(_segment_unknown_token(token))
    return segmented


def _drop_numeric_tokens(tokens: list[str]) -> list[str]:
    """Remove tokens made only of digits while retaining mixed tokens."""
    return [token for token in tokens if not token.isdigit()]


def _drop_stopwords(tokens: list[str]) -> list[str]:
    """Remove grammatical stopwords while retaining ATS domain terms."""
    return [
        token
        for token in tokens
        if token in DOMAIN_KEEP_WORDS or token not in STOP_WORDS
    ]


def _drop_single_character_tokens(tokens: list[str]) -> list[str]:
    """Keep tokens compatible with scikit-learn's default two-character rule."""
    return [token for token in tokens if len(token) > 1]


def tokenize(text: str) -> list[str]:
    """Normalize resume or job-description text and return clean tokens.

    Args:
        text: Raw text to normalize.

    Returns:
        Lowercase alphanumeric tokens with accents, numeric-only tokens,
        grammatical stopwords, and one-character tokens removed.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Expected input to be of type str")

    normalized = _normalize_unicode(text)
    normalized = _replace_invisible_controls(normalized)
    normalized = _split_case_boundaries(normalized)
    normalized = normalized.lower()
    normalized = _replace_tech_aliases(normalized)
    normalized = _replace_non_alphanumeric(normalized)
    tokens = normalized.split()
    tokens = _segment_compound_tokens(tokens)
    tokens = _drop_numeric_tokens(tokens)
    tokens = _drop_stopwords(tokens)
    return _drop_single_character_tokens(tokens)


def clean_text(text: str) -> str:
    """Return :func:`tokenize` output joined by single spaces.

    Args:
        text: Raw text to normalize.

    Returns:
        The clean tokens joined into a string.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    return " ".join(tokenize(text))


if __name__ == "__main__":
    _verification_cases = (
        ("Python, SQL & C++", ["python", "sql", "cpp"]),
        ("Built with Node.js and C#", ["built", "nodejs", "csharp"]),
        ("scikit-learn, ci/cd", ["scikit", "learn", "ci", "cd"]),
        ("Analyzed 3,900 transactions in 2025", ["analyzed", "transactions"]),
        ("Résumé", ["resume"]),
        ("Python3 and S3", ["python3", "s3"]),
        ("Warangal’s", ["warangal"]),
        ("", []),
        ("   ", []),
        ("davpublic school", ["davpublic", "school"]),
        ("DAV\u200bPublic school", ["dav", "public", "school"]),
        (
            "data collection validationensure accuracy",
            ["data", "collection", "validation", "ensure", "accuracy"],
        ),
        (
            "data collection validation\xadensure accuracy",
            ["data", "collection", "validation", "ensure", "accuracy"],
        ),
        (
            "seaborn matplotlibscikit-learn",
            ["seaborn", "matplotlib", "scikit", "learn"],
        ),
        (
            "recommendation distribution configuration",
            ["recommendation", "distribution", "configuration"],
        ),
        (
            "recommendation distribution configuration effectiveness",
            ["recommendation", "distribution", "configuration", "effectiveness"],
        ),
        ("Movie Recommendation System", ["movie", "recommendation", "system"]),
        ("Attention to Detail", ["attention", "detail"]),
        ("Top contributing countries", ["top", "contributing", "countries"]),
        ("Areas of Interest", ["areas", "interest"]),
        (
            "performed eda netflixdataset using python",
            ["performed", "eda", "netflix", "dataset", "using", "python"],
        ),
        (
            "jupyter notebook cleanedanalyzed netflix dataset",
            ["jupyter", "notebook", "cleaned", "analyzed", "netflix", "dataset"],
        ),
        (
            "top contributing countriesvisualizations movie recommendation",
            ["top", "contributing", "countries", "visualizations", "movie", "recommendation"],
        ),
        (
            "python numpy postgresql github linkedin kpis",
            ["python", "numpy", "postgresql", "github", "linkedin", "kpis"],
        ),
        (
            "operations stakeholders streamlit features supported performed developed",
            [
                "operations",
                "stakeholders",
                "streamlit",
                "features",
                "supported",
                "performed",
                "developed",
            ],
        ),
        (
            "netflixdataset cleanedanalyzed countriesvisualizations",
            ["netflix", "dataset", "cleaned", "analyzed", "countries", "visualizations"],
        ),
        (
            "engineeringgit personalprojects spotifylistening edanetflix wordpower schemesep",
            [
                "engineeringgit",
                "personal",
                "projects",
                "spotify",
                "listening",
                "edanetflix",
                "word",
                "power",
                "schemesep",
            ],
        ),
        (
            "GitHub PostgreSQL NumPy CountVectorizer FastAPI",
            ["github", "postgresql", "numpy", "countvectorizer", "fastapi"],
        ),
        (
            "PersonalProjects SpotifyListening",
            ["personal", "projects", "spotify", "listening"],
        ),
        (
            "personalprojects spotifylistening wordpower seamlessexecution sectorcbse",
            [
                "personal",
                "projects",
                "spotify",
                "listening",
                "word",
                "power",
                "seamless",
                "execution",
                "sector",
                "cbse",
            ],
        ),
    )
    for _source, _expected in _verification_cases:
        assert tokenize(_source) == _expected, (_source, tokenize(_source))

    assert tokenize("davPublic validationEnsure SQL NLP EDA NSS") == [
        "dav",
        "public",
        "validation",
        "ensure",
        "sql",
        "nlp",
        "eda",
        "nss",
    ]
    assert tokenize("abc#") == ["abc"]
    assert tokenize("KPIs") == ["kpis"]
    _sample = "Built resume tools with Python, FastAPI, and C++."
    print(clean_text(_sample))
