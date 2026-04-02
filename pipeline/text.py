import re
from typing import Dict, Optional


class TextNormalizer:
    """Text normalization."""

    DEFAULT_REPLACEMENTS = {
        "\\[": "$$",
        "\\]": "$$",
        "\\(": "$",
        "\\)": "$",
        "\u2018": "'",
        "\u2019": "'",
        "\u0060": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2026": "...",
    }

    def __init__(self, custom_rules: Optional[Dict[str, str]] = None):
        self.replacements = {**self.DEFAULT_REPLACEMENTS, **(custom_rules or {})}
        self._pattern = re.compile("|".join(re.escape(k) for k in self.replacements))

    def normalize(self, text: str) -> str:
        return self._pattern.sub(lambda m: self.replacements[m.group()], text)
