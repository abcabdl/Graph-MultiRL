from __future__ import annotations

_MOJIBAKE_REPLACEMENTS = {
    "鈧€": "0",
    "鈥檚": "'s",
    "鈥檛": "n't",
    "鈥檙": "'r",
    "鈥檝": "'v",
    "鈥檒": "'l",
    "鈥": "'",
    "鉁?": "",
    "虏": "^2",
    "魔": "hbar",
    "蠅": "omega",
    "路": "*",
}


def sanitize_text(text: str | None) -> str:
    """Repair common UTF-8-as-GBK mojibake in stored diagnostics text.

    This is intentionally conservative and aimed at GraphCredit logs/debug
    records. It should not be used to alter token ids or policy inputs.
    """

    if text is None:
        return ""
    repaired = str(text)
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    return repaired
