from __future__ import annotations

import importlib.util
import unittest

from evaluation_pipeline.text_encoding import encode_hanzi_to_initial_digit


HAS_ENCODING_DEPS = (
    importlib.util.find_spec("jieba") is not None
    and importlib.util.find_spec("pypinyin") is not None
)


@unittest.skipUnless(HAS_ENCODING_DEPS, "jieba and pypinyin are required")
class TextEncodingTests(unittest.TestCase):
    def test_wo_men_example(self):
        self.assertEqual(encode_hanzi_to_initial_digit("\u6211\u4eec"), "W6M7")

    def test_jieba_words_are_space_separated(self):
        encoded = encode_hanzi_to_initial_digit("\u5df2\u7ecf\u5f88\u665a\u4e86")
        self.assertGreaterEqual(len(encoded.split()), 2)

    def test_chinese_punctuation_is_normalized(self):
        encoded = encode_hanzi_to_initial_digit("\u201c\u6211\u4eec\u201d\uff0c\u597d\u3002")
        self.assertIn('"', encoded)
        self.assertIn(",", encoded)
        self.assertIn(".", encoded)
        self.assertNotIn("\u201c", encoded)
        self.assertNotIn("\uff0c", encoded)

    def test_mixed_text_does_not_crash(self):
        encoded = encode_hanzi_to_initial_digit("BabyLM\u6211\u4eec123, OK")
        self.assertIn("BabyLM", encoded)
        self.assertIn("123", encoded)
        self.assertIn(",", encoded)


if __name__ == "__main__":
    unittest.main()
