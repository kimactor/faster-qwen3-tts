from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def default_pronunciation_lexicon_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "pronunciation_lexicon.txt"


@dataclass(frozen=True)
class PronunciationRule:
    phrase: str
    pinyin: tuple[str, ...] = ()
    spoken: str | None = None


_COMMENT_RE = re.compile(r"\s+#.*$")
_LEADING_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*)")
_SPACE_RE = re.compile(r"\s+")
_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
_PUNCT_RE = re.compile(r"\s*([,\.;:!?\u3001\u3002\uff01\uff1f\uff1b\uff1a\uff0c])\s*")
_DUP_PUNCT_RE = re.compile(r"([,\.;:!?\u3001\u3002\uff01\uff1f\uff1b\uff1a\uff0c]){2,}")
_BRACKET_RE = re.compile(r"[<>\[\]{}\\^~]+")


_BUILTIN_SILENT_RULES = tuple(
    PronunciationRule(symbol, spoken="")
    for symbol in (
        "**",
        "__",
        "*",
        "`",
        "#",
        '"',
        "\u201c",
        "\u201d",
        "'",
        "\u2018",
        "\u2019",
        "--",
        "\u2013",
        "\u2014",
        "\u2014\u2014",
        "-",
        "\u00b7",
        "|",
    )
)


def _normalize_basic_markdown(text: str) -> str:
    text = _FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _LEADING_BULLET_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    return text


def _load_rule_line(line: str) -> PronunciationRule | None:
    body = _COMMENT_RE.sub("", line).strip()
    if not body:
        return None

    spoken: str | None = None
    if "=>" in body:
        body, spoken = body.split("=>", 1)
        spoken = spoken.strip()

    pinyin: tuple[str, ...] = ()
    phrase = body.strip()
    bracket_index = body.find("[")
    if bracket_index != -1:
        phrase = body[:bracket_index].strip()
        payload = body[bracket_index:].strip()
        if payload:
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid pronunciation rule: {line.strip()}") from exc
            if isinstance(parsed, str):
                pinyin = (parsed,)
            else:
                pinyin = tuple(str(item).strip() for item in parsed if str(item).strip())

    if not phrase:
        return None
    return PronunciationRule(phrase=phrase, pinyin=pinyin, spoken=spoken)


def load_pronunciation_rules(path: str | Path | None) -> list[PronunciationRule]:
    rules = list(_BUILTIN_SILENT_RULES)
    if not path:
        return rules

    lexicon_path = Path(path)
    if not lexicon_path.exists():
        return rules

    for line in lexicon_path.read_text(encoding="utf-8").splitlines():
        rule = _load_rule_line(line)
        if rule is not None:
            rules.append(rule)
    return rules


class TextNormalizer:
    def __init__(self, lexicon_path: str | Path | None = None):
        self.lexicon_path = Path(lexicon_path) if lexicon_path else None
        self._mtime_ns: int | None = None
        self._rules: list[PronunciationRule] = []
        self._reload(force=True)

    def _reload(self, force: bool = False) -> None:
        path = self.lexicon_path
        if path is None:
            if force:
                self._rules = load_pronunciation_rules(None)
            return

        mtime_ns = path.stat().st_mtime_ns if path.exists() else None
        if not force and mtime_ns == self._mtime_ns:
            return

        self._mtime_ns = mtime_ns
        self._rules = load_pronunciation_rules(path)

    @property
    def rules(self) -> list[PronunciationRule]:
        self._reload()
        return list(self._rules)

    def normalize_for_tts(self, text: str) -> str:
        self._reload()

        normalized = _normalize_basic_markdown(text or "")
        for rule in sorted(self._rules, key=lambda item: len(item.phrase), reverse=True):
            replacement = rule.spoken
            if replacement is None or not rule.phrase:
                continue
            normalized = normalized.replace(rule.phrase, replacement)

        normalized = normalized.replace("\r", "\n")
        normalized = _BRACKET_RE.sub(" ", normalized)
        normalized = re.sub(r"\s*\n\s*", " ", normalized)
        normalized = _SPACE_RE.sub(" ", normalized)
        normalized = _CJK_SPACE_RE.sub("", normalized)
        normalized = _PUNCT_RE.sub(r"\1", normalized)
        normalized = _DUP_PUNCT_RE.sub(r"\1", normalized)
        return normalized.strip()

    def describe_rules(self) -> list[dict]:
        self._reload()
        return [
            {
                "phrase": rule.phrase,
                "pinyin": list(rule.pinyin),
                "spoken": rule.spoken,
            }
            for rule in self._rules
        ]


def normalize_tts_text(text: str, lexicon_path: str | Path | None = None) -> str:
    return TextNormalizer(lexicon_path).normalize_for_tts(text)


def iter_spoken_aliases(rules: Iterable[PronunciationRule]) -> Iterable[tuple[str, str]]:
    for rule in rules:
        if rule.spoken is not None:
            yield rule.phrase, rule.spoken
