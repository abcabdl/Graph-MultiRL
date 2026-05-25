from __future__ import annotations

import re

VERIFY_TAG_RE = re.compile(r"<verify>\s*(approve|reject)\s*</verify>", re.IGNORECASE)


def verify_tags(text: str | None) -> list[str]:
    return [match.group(1).lower() for match in VERIFY_TAG_RE.finditer(text or "")]


def last_verifier_verdict(text: str | None) -> str:
    tags = verify_tags(text)
    return tags[-1] if tags else "other"


def has_single_final_verifier_tag(text: str | None) -> bool:
    matches = list(VERIFY_TAG_RE.finditer(text or ""))
    if len(matches) != 1:
        return False
    suffix = (text or "")[matches[0].end():].strip()
    return suffix == ""
