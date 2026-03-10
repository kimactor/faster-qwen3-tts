import sys
import types

sys.modules.setdefault("soundfile", types.SimpleNamespace())

from faster_qwen3_tts.text_processing import TextNormalizer, load_pronunciation_rules


RULE_LINE = "\u8bf4\u670d ['shui4','fu2'] => \u7a0e\u670d\n"


def test_load_pronunciation_rules_parses_pinyin_and_spoken_alias(tmp_path):
    lexicon = tmp_path / "lexicon.txt"
    lexicon.write_text(RULE_LINE, encoding="utf-8")

    rules = load_pronunciation_rules(lexicon)
    match = [rule for rule in rules if rule.phrase == "\u8bf4\u670d"][-1]

    assert match.pinyin == ("shui4", "fu2")
    assert match.spoken == "\u7a0e\u670d"


def test_text_normalizer_strips_markdown_symbols_and_quotes(tmp_path):
    lexicon = tmp_path / "lexicon.txt"
    lexicon.write_text("", encoding="utf-8")
    normalizer = TextNormalizer(lexicon)

    text = '**"\u4f60\u597d"- \u8fd9\u662f`\u6d4b\u8bd5`**'
    assert normalizer.normalize_for_tts(text) == "\u4f60\u597d\u8fd9\u662f\u6d4b\u8bd5"


def test_text_normalizer_applies_custom_spoken_alias(tmp_path):
    lexicon = tmp_path / "lexicon.txt"
    lexicon.write_text(RULE_LINE, encoding="utf-8")
    normalizer = TextNormalizer(lexicon)

    assert normalizer.normalize_for_tts("\u8bf7\u8bf4\u670d\u5927\u5bb6\u3002") == "\u8bf7\u7a0e\u670d\u5927\u5bb6\u3002"
