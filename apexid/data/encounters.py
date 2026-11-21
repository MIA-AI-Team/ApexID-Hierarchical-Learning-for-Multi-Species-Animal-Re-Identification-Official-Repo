"""Filename-based observation grouping for external evaluation datasets."""

import re
from pathlib import Path


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)


def infer_bristol_encounter(identity: str, filename: str) -> tuple[str, bool]:
    """Group numbered sessions and trial sequences without splitting frames."""

    stem = Path(filename).stem
    escaped_identity = re.escape(identity)
    session = re.match(rf"^{escaped_identity}-(\d+)-", stem, flags=re.IGNORECASE)
    if session:
        return f"bristol_gorillas:{identity}:session_{session.group(1)}", True
    trial = re.match(
        rf"^{escaped_identity}-(?:(?:[A-Za-z]+)-)?trial(?:[-_]|$)",
        stem,
        flags=re.IGNORECASE,
    )
    if trial:
        return f"bristol_gorillas:{identity}:trial", True
    return f"bristol_gorillas:{identity}:unknown_sequence", False


def _normalise_elephant_date(stem: str) -> str | None:
    """Extract the rightmost explicit date token without inventing precision.

    ``YYYY-MM-DD``, ``YYYY-MM``, and ``YYYY`` preserve the precision actually
    present in filenames.  A year-only token is still a useful conservative
    encounter group: every image for that identity with that year remains
    together.
    """

    patterns = (
        re.compile(
            rf"(?i)(?P<day>\d{{1,2}})\s*(?P<month>{MONTH_PATTERN})\s*(?P<year>\d{{4}})"
        ),
        re.compile(
            rf"(?i)(?P<month>{MONTH_PATTERN})\s*(?P<day>\d{{1,2}})\s*,?\s*(?P<year>\d{{4}})"
        ),
        re.compile(rf"(?i)(?P<month>{MONTH_PATTERN})\s*(?P<year>\d{{4}})"),
        re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)"),
    )
    candidates: list[tuple[int, int, dict[str, str | None]]] = []
    for pattern in patterns:
        for match in pattern.finditer(stem):
            candidates.append((match.end(), match.start(), match.groupdict()))
    if not candidates:
        return None
    _end, _start, groups = max(candidates, key=lambda value: (value[0], value[0] - value[1]))
    year = groups["year"]
    month_text = groups.get("month")
    day_text = groups.get("day")
    fields = [year]
    if month_text:
        fields.append(f"{MONTHS[month_text[:3].lower()]:02d}")
    if day_text:
        fields.append(f"{int(day_text):02d}")
    return "-".join(fields)


def infer_elephant_encounter(identity: str, filename: str) -> tuple[str, bool]:
    date = _normalise_elephant_date(Path(filename).stem)
    if date is not None:
        return f"elephants:{identity}:{date}", True
    return f"elephants:{identity}:unknown_date", False
