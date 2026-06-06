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
        self.assertEqual(encode_hanzi_to_initial_digit("我们"), "W6M7")

    def test_jieba_words_are_space_separated(self):
        encoded = encode_hanzi_to_initial_digit("已经很晚了")
        self.assertGreaterEqual(len(encoded.split()), 2)

    def test_punctuation_is_preserved(self):
        encoded = encode_hanzi_to_initial_digit("我们。")
        self.assertIn("。", encoded)

    def test_mixed_text_does_not_crash(self):
        encoded = encode_hanzi_to_initial_digit("BabyLM我们123, OK")
        self.assertIn("BabyLM", encoded)
        self.assertIn("123", encoded)
        self.assertIn(",", encoded)


if __name__ == "__main__":
    unittest.main()
