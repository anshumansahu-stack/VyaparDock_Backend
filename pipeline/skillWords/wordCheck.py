import json
import re

def normalize(term: str) -> str:
    return re.sub(r'[^a-z0-9]', '', term.lower())

with open("pipeline/skillWords/taxonomy.json", "r", encoding="utf-8") as f:
    taxonomy_data = json.load(f)

KNOWN_SKILL_TERMS = set()

for category in taxonomy_data["categories"].values():
    for subcategory in category["subcategories"].values():
        for skill in subcategory["skills"]:
            KNOWN_SKILL_TERMS.add(normalize(skill["canonical_name"]))
            for alias in skill.get("aliases", []):
                KNOWN_SKILL_TERMS.add(normalize(alias))

print(len(KNOWN_SKILL_TERMS))
print(sorted(KNOWN_SKILL_TERMS)[:20])  # sanity check a slice of it

with open("pipeline/skillWords/skill_terms.json", "w", encoding="utf-8") as f:
    json.dump(sorted(KNOWN_SKILL_TERMS), f)