"""Pure host WRF namelist parser, shared by offline tools and model IO."""
from __future__ import annotations
from pathlib import Path
import re
from typing import Any

def _namelist_tokens(text: str) -> list[str]:
    """Tokenize delimiters outside strings; Fortran doubles embedded quotes."""

    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace():
            pos += 1
        elif char == "!":
            end = text.find("\n", pos)
            pos = len(text) if end < 0 else end + 1
        elif char in {"'", '"'}:
            start, quote = pos, char
            pos += 1
            while pos < len(text):
                if text[pos] == quote:
                    if pos + 1 < len(text) and text[pos + 1] == quote:
                        pos += 2
                        continue
                    pos += 1
                    tokens.append(text[start:pos])
                    break
                pos += 1
            else:
                raise ValueError("unterminated namelist character value")
        elif char in "&/=,*():":
            tokens.append(char)
            pos += 1
        else:
            start = pos
            while pos < len(text) and not text[pos].isspace() and text[pos] not in "&/=,*():!'\"":
                pos += 1
            tokens.append(text[start:pos])
    return tokens


def _parse_scalar(text: str) -> Any:
    raw = text.strip()
    if len(raw) >= 2 and raw[0] in {"'", '"'} and raw[-1] == raw[0]:
        return raw[1:-1].replace(raw[0] * 2, raw[0])
    lowered = raw.lower()
    if lowered in {".true.", "true", ".t.", "t"}:
        return True
    if lowered in {".false.", "false", ".f.", "f"}:
        return False
    normalized = raw.replace("D", "E").replace("d", "e")
    try:
        if not any(char in normalized for char in ".eE"):
            return int(normalized)
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid namelist value {raw!r}; character values must be quoted") from exc


def parse_namelist(path: Path) -> dict[str, dict[str, Any]]:
    """Read WRF scalar/one-dimensional namelists with list-directed semantics.

    Null slots are ``None``: consumers apply their Registry/default value.
    Reassigning a variable leaves previously assigned null slots untouched.
    Repeat counts apply to every scalar type, including nulls and characters.
    Unsupported syntax or malformed values raise instead of becoming strings.
    """

    tokens = _namelist_tokens(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    pos = 0
    identifier = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
    def assignment(at: int) -> bool:
        return at + 1 < len(tokens) and bool(identifier.fullmatch(tokens[at])) and tokens[at + 1] == "="

    while pos < len(tokens):
        token = tokens[pos]
        if token == "&":
            pos += 1
            if pos >= len(tokens) or not identifier.fullmatch(tokens[pos]):
                raise ValueError(f"{path}: invalid namelist group")
            name = tokens[pos].lower()
            if name == "end":
                current = None
            else:
                if current is not None:
                    raise ValueError(f"{path}: previous namelist group was not closed")
                current = groups.setdefault(name, {})
            pos += 1
            continue
        if token == "/":
            if current is None:
                raise ValueError(f"{path}: unexpected group terminator")
            current = None
            pos += 1
            continue
        if current is None or not assignment(pos):
            raise ValueError(f"{path}: expected namelist assignment, got {token!r}")
        key = token.lower()
        pos += 2
        values: list[Any] = []
        expect_value = True
        while pos < len(tokens) and tokens[pos] not in {"/", "&"} and not assignment(pos):
            token = tokens[pos]
            if token == ",":
                if expect_value:
                    values.append(None)
                expect_value = True
                pos += 1
                continue
            count = 1
            if pos + 1 < len(tokens) and tokens[pos + 1] == "*":
                if not token.isdecimal() or int(token) < 1:
                    raise ValueError(f"{path}: invalid repeat count {token!r}")
                count = int(token)
                pos += 2
                if pos >= len(tokens) or tokens[pos] in {",", "/", "&"} or assignment(pos):
                    values.extend([None] * count)
                    expect_value = False
                    continue
                token = tokens[pos]
            values.extend([_parse_scalar(token)] * count)
            pos += 1
            expect_value = False
        prior = current.get(key, [])
        merged = list(prior) if isinstance(prior, list) else [prior]
        if len(merged) < len(values):
            merged.extend([None] * (len(values) - len(merged)))
        for index, value in enumerate(values):
            if value is not None:
                merged[index] = value
        if merged:
            current[key] = merged[0] if len(merged) == 1 else merged
    if current is not None:
        raise ValueError(f"{path}: unterminated namelist group")
    return groups


