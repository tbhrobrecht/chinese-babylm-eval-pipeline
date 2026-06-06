from __future__ import annotations

import re
from typing import Any


INPUT_REPRESENTATION_HANZI = "hanzi"
INPUT_REPRESENTATION_ENCODED_HANZI = "encoded_hanzi"
INPUT_REPRESENTATION_CHOICES = (
    INPUT_REPRESENTATION_HANZI,
    INPUT_REPRESENTATION_ENCODED_HANZI,
)

_CJK_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    "\U0002b740-\U0002b81f\U0002b820-\U0002ceaf"
    "\U0002ceb0-\U0002ebef\U00030000-\U0003134f]"
)
_TONE_RE = re.compile(r"([1-5])$")
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "，": ",",
        "。": ".",
        "、": ",",
        "；": ";",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "　": " ",
    }
)


def _is_chinese_char(char: str) -> bool:
    return bool(_CJK_RE.fullmatch(char))


def _require_encoding_deps():
    try:
        import jieba
        from pypinyin import Style, pinyin
    except ImportError as exc:
        raise ImportError(
            "encoded_hanzi input representation requires the 'jieba' and "
            "'pypinyin' packages. Install them from requirements.txt."
        ) from exc
    return jieba, pinyin, Style


def _syllable_to_initial_digit(syllable: str) -> str:
    match = _TONE_RE.search(syllable)
    tone = int(match.group(1)) if match else 5
    base = _TONE_RE.sub("", syllable)
    if not base:
        return syllable

    initial = base[0]
    if tone in (1, 3, 5):
        initial = initial.upper()
    else:
        initial = initial.lower()

    length = len(base)
    length_offset = 4 if length >= 5 else max(length - 1, 0)
    tone_offset = 0 if tone in (1, 2) else 5
    return f"{initial}{tone_offset + length_offset}"


def _encode_jieba_word(word: str, pinyin, style) -> str:
    syllables = pinyin(
        word,
        style=style.TONE3,
        heteronym=False,
        neutral_tone_with_five=True,
        errors=lambda chars: list(chars),
        strict=False,
    )

    encoded = []
    for char, item in zip(word, syllables):
        value = item[0] if item else char
        if _is_chinese_char(char):
            encoded.append(_syllable_to_initial_digit(value))
        else:
            encoded.append(char.translate(_PUNCTUATION_TRANSLATION))

    if len(encoded) < len(word):
        encoded.extend(word[len(encoded):].translate(_PUNCTUATION_TRANSLATION))
    return "".join(encoded)


def encode_hanzi_to_initial_digit(text: str) -> str:
    """Encode Chinese Hanzi text as Pinyin-initial-plus-digit tokens.

    Jieba word boundaries are represented with spaces. Existing whitespace is
    preserved so prompts with line breaks or few-shot formatting stay readable.
    Non-Chinese text, numbers, and punctuation are kept as literal text.
    """
    if not isinstance(text, str) or text == "":
        return text

    jieba, pinyin, style = _require_encoding_deps()
    pieces: list[str] = []

    for word in jieba.cut(text):
        if word == "":
            continue
        if word.isspace():
            pieces.append(word)
            continue

        encoded_word = _encode_jieba_word(word, pinyin, style)
        if pieces and not pieces[-1].isspace() and not pieces[-1].endswith((" ", "\n", "\t", "\r")):
            pieces.append(" ")
        pieces.append(encoded_word)

    return "".join(pieces)


def input_representation_from_args(args: Any) -> str:
    return getattr(args, "input_representation", INPUT_REPRESENTATION_HANZI) or INPUT_REPRESENTATION_HANZI


def should_encode_hanzi(input_representation: str | None) -> bool:
    return input_representation == INPUT_REPRESENTATION_ENCODED_HANZI


def convert_text_for_representation(text: str | None, input_representation: str | None) -> str | None:
    if text is None or not should_encode_hanzi(input_representation):
        return text
    return encode_hanzi_to_initial_digit(text)


def convert_text_sequence_for_representation(
    texts: list[str | None],
    input_representation: str | None,
) -> list[str | None]:
    return [convert_text_for_representation(text, input_representation) for text in texts]


def convert_completion_ranking_item(
    data_dict: dict[str, Any],
    input_representation: str | None,
) -> dict[str, Any]:
    if not should_encode_hanzi(input_representation):
        return data_dict

    converted = dict(data_dict)
    sentences = list(data_dict["sentences"])
    prefixes = list(data_dict["prefixes"])
    completions = list(data_dict["completions"])

    converted_completions = convert_text_sequence_for_representation(
        completions,
        input_representation,
    )
    converted_prefixes = convert_text_sequence_for_representation(
        prefixes,
        input_representation,
    )
    converted_sentences: list[str] = []

    for sentence, completion, converted_completion in zip(sentences, completions, converted_completions):
        if completion is not None and sentence.endswith(completion):
            prefix_text = sentence[: len(sentence) - len(completion)] if completion else sentence
            converted_prefix_text = convert_text_for_representation(prefix_text, input_representation)
            if completion:
                converted_sentences.append(_join_encoded_parts(converted_prefix_text, converted_completion))
            else:
                converted_sentences.append(converted_prefix_text or "")
        else:
            converted_sentences.append(convert_text_for_representation(sentence, input_representation) or "")

    converted["sentences"] = converted_sentences
    converted["prefixes"] = converted_prefixes
    converted["completions"] = converted_completions
    return converted


def _join_encoded_parts(prefix: str | None, completion: str | None) -> str:
    prefix = prefix or ""
    completion = completion or ""
    if not prefix:
        return completion
    if not completion:
        return prefix
    if prefix.endswith((" ", "\n", "\t", "\r")) or completion.startswith((" ", "\n", "\t", "\r")):
        return prefix + completion
    return prefix + " " + completion
