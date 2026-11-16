# import re
# import string
# import json
# import nltk
# import wordninja
# from nltk.corpus import stopwords, words

# nltk.download('stopwords', quiet=True)
# nltk.download('words', quiet=True)

# english_vocab = set(words.words())
# stop_words = set(stopwords.words('english'))

# # Load your taxonomy-derived skill terms
# with open("pipeline/skillWords/skill_terms.json", "r", encoding="utf-8") as f:
#     KNOWN_SKILL_TERMS = set(json.load(f))

# # Manual fallback additions
# KNOWN_SKILL_TERMS |= {"nalco"}

# def dehyphenate(text: str) -> str:
#     """Rejoin line-wrap-hyphenated words, e.g. 'filter-\ning' -> 'filtering'."""
#     def try_join(match):
#         first, second = match.group(1), match.group(2)
#         joined = first + second
#         if joined.lower() in english_vocab:
#             return joined
#         return f"{first}-{second}"

#     return re.sub(r'(\w+)-\n(\w+)', try_join, text)


# def normalize_and_protect_skills(text: str) -> str:
#     """Dynamically checks for broken or spaced-out variations of known 
#     taxonomy skill terms and normalizes them into single unified tokens."""
    
#     # Common PDF artifact repairs for data science & backend tools
#     common_fixes = {
#         "sci kit learn": "scikit-learn",
#         "scikit learn": "scikit-learn",
#         "post gre sql": "postgresql",
#         "post gresql": "postgresql",
#         "num py": "numpy",
#         "mat plot lib": "matplotlib",
#         "seaborn": "seaborn",
#         "count vector ize r": "count vectorizer",
#         "cbseodisha": "cbse odisha",
#         "damanjodiodisha": "damanjodi odisha"
#     }
    
#     for wrong, right in common_fixes.items():
#         text = text.replace(wrong, right)
        
#     # Generalized dynamic check: if multi-word skills from our JSON 
#     # appear with spaces/missing hyphens, normalize them temporarily
#     for skill in KNOWN_SKILL_TERMS:
#         # If the skill contains a hyphen (e.g. "scikit-learn"), check if it's written as "scikit learn"
#         if "-" in skill:
#             spaced_version = skill.replace("-", " ")
#             if spaced_version in text:
#                 text = text.replace(spaced_version, skill)
                
#     return text


# def lowercase(text: str) -> str:
#     return text.lower()


# def remove_punctuation(text: str) -> str:
#     """Removes general punctuation while preserving essential separators 
#     found inside known technical terms (like hyphens)."""
#     # Allow hyphens and underscores for tech terms, wipe the rest
#     allowed_chars = set("-_")
#     translator = str.maketrans('', '', ''.join([c for c in string.punctuation if c not in allowed_chars]))
#     text = text.translate(translator)
#     return re.sub(r'\s+-\s+', ' ', text)


# def remove_stopwords(text: str) -> str:
#     tokens = text.split()
#     # Ensure we don't accidentally strip short single-letter or common tech stopwords if needed, 
#     # but strictly follow standard stopword removal otherwise.
#     filtered = [word for word in tokens if word not in stop_words]
#     return " ".join(filtered)


# def split_fused_words(text: str, min_length: int = 15) -> str:
#     """Split unusually long tokens that are likely multiple fused words,
#     skipping known skill/tech terms entirely."""
#     tokens = text.split()
#     result = []
#     for token in tokens:
#         # Check against normalized forms (with or without punctuation)
#         clean_token = re.sub(r'[^a-z0-9]', '', token)
#         if clean_token in KNOWN_SKILL_TERMS or token.lower() in KNOWN_SKILL_TERMS:
#             result.append(token)
#         elif len(token) >= min_length:
#             split_words = wordninja.split(token)
#             result.extend(split_words)
#         else:
#             result.append(token)
#     return " ".join(result)


# def clean_text(text: str) -> str:
#     """Runs the full cleaning pipeline in sequence while maintaining modularity."""
#     text = dehyphenate(text)
#     text = lowercase(text)
#     text = normalize_and_protect_skills(text)  # <-- Intercepts and fixes broken terms early
#     text = remove_punctuation(text)
#     text = remove_stopwords(text)
#     text = split_fused_words(text)
#     return text

import re
import string
import json
import nltk
import wordninja
from nltk.corpus import stopwords, words

nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True)

english_vocab = set(words.words())
stop_words = set(stopwords.words('english'))

# Load your taxonomy-derived skill terms
with open("pipeline/skillWords/skill_terms.json", "r", encoding="utf-8") as f:
    KNOWN_SKILL_TERMS = set(json.load(f))

def dehyphenate(text: str) -> str:
    """Rejoin line-wrap-hyphenated words, e.g. 'filter-\ning' -> 'filtering'."""
    try_join = lambda m: m.group(1) + m.group(2) if (m.group(1) + m.group(2)).lower() in english_vocab else f"{m.group(1)}-{m.group(2)}"
    return re.sub(r'(\w+)-\n(\w+)', try_join, text)


def normalize_string(s: str) -> str:
    """Removes all non-alphanumeric characters for clean key lookups."""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def dynamic_ngram_normalization(text: str) -> str:
    """Dynamically checks sliding n-gram windows (up to 3 words) against 
    the taxonomy database to fix spaced-out or broken skill terms automatically."""
    tokens = text.split()
    processed_tokens = []
    i = 0
    n = len(tokens)

    while i < n:
        matched = False
        # Try matching 3-word, 2-word, then 1-word sliding windows
        for window_size in [3, 2]:
            if i + window_size <= n:
                window_phrase = " ".join(tokens[i:i+window_size])
                normalized_phrase = normalize_string(window_phrase)
                
                if normalized_phrase in KNOWN_SKILL_TERMS:
                    # Reconstruct back into a clean canonical-like unified token format
                    processed_tokens.append(window_phrase.replace(" ", "-"))
                    i += window_size
                    matched = True
                    break
        if not matched:
            processed_tokens.append(tokens[i])
            i += 1
            
    return " ".join(processed_tokens)


def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    allowed_chars = set("-_")
    translator = str.maketrans('', '', ''.join([c for c in string.punctuation if c not in allowed_chars]))
    text = text.translate(translator)
    return re.sub(r'\s+-\s+', ' ', text)


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


def split_fused_words(text: str, min_length: int = 15) -> str:
    tokens = text.split()
    result = []
    for token in tokens:
        clean_token = normalize_string(token)
        if clean_token in KNOWN_SKILL_TERMS or token.lower() in KNOWN_SKILL_TERMS:
            result.append(token)
        elif len(token) >= min_length:
            result.extend(wordninja.split(token))
        else:
            result.append(token)
    return " ".join(result)


def clean_text(text: str) -> str:
    """Runs the full cleaning pipeline sequentially."""
    text = dehyphenate(text)
    text = lowercase(text)
    text = dynamic_ngram_normalization(text)  # <-- Replaces manual hardcoded dictionary completely
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = split_fused_words(text)
    return text