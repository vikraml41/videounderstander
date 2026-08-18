import unittest

from vu.artifact import fmt_ts, parse_ts
from vu.distill import (check_quotes, quote_in_transcript,
                        split_lines_by_time)

TRANSCRIPT = ("so if you look at this curve, the interesting part isn't the "
              "peak — it's how fast it decays after. We call this K-entropy, "
              "not to be confused with cage entropy.")


class TestQuoteCheck(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(quote_in_transcript(
            "the interesting part isn't the peak", TRANSCRIPT))

    def test_punctuation_and_case_insensitive(self):
        self.assertTrue(quote_in_transcript(
            "Its how fast it decays after", TRANSCRIPT))

    def test_fuzzy_asr_noise(self):
        # one word off within a long quote still passes
        self.assertTrue(quote_in_transcript(
            "if you look at these curve the interesting part isn't the peak",
            TRANSCRIPT))

    def test_paraphrase_fails(self):
        self.assertFalse(quote_in_transcript(
            "the decay rate matters more than the maximum value", TRANSCRIPT))

    def test_normalized_term_fails(self):
        # the novelty-collapse case: model rewrote "K-entropy" to a standard
        # term and invented a supporting quote
        self.assertFalse(quote_in_transcript(
            "we call this Kolmogorov entropy", TRANSCRIPT))

    def test_empty_quote_fails(self):
        self.assertFalse(quote_in_transcript("", TRANSCRIPT))

    def test_check_quotes_annotates(self):
        chunks = [{"claims": [
            {"claim": "a", "quote": "we call this K-entropy"},
            {"claim": "b", "quote": "totally invented sentence here"},
        ]}]
        failures = check_quotes(chunks, TRANSCRIPT)
        self.assertEqual(failures, 1)
        self.assertTrue(chunks[0]["claims"][0]["quote_verified"])
        self.assertFalse(chunks[0]["claims"][1]["quote_verified"])


class TestChunkSplit(unittest.TestCase):
    def test_split_on_boundaries(self):
        lines = [
            '[00:10] "intro"',
            '[04:00] "first argument"',
            '[09:30] "second argument"',
            '[12:00] "second argument continues"',
        ]
        # 9:30 sits exactly on the second boundary -> starts the next chunk;
        # the empty middle chunk (5:00–9:30 had no lines) is dropped
        chunks = split_lines_by_time(lines, [300.0, 570.0])
        self.assertEqual(chunks, [lines[:2], lines[2:]])

    def test_hour_timestamps(self):
        lines = ['[59:50] "a"', '[1:00:10] "b"']
        chunks = split_lines_by_time(lines, [3600.0])
        self.assertEqual(chunks, [[lines[0]], [lines[1]]])


class TestTimestamps(unittest.TestCase):
    def test_fmt(self):
        self.assertEqual(fmt_ts(252), "04:12")
        self.assertEqual(fmt_ts(3723), "1:02:03")
        self.assertEqual(fmt_ts(0), "00:00")

    def test_parse(self):
        self.assertEqual(parse_ts("04:12"), 252.0)
        self.assertEqual(parse_ts("1:02:03"), 3723.0)
        self.assertEqual(parse_ts("00:07,500"), 7.5)
        self.assertEqual(parse_ts("42"), 42.0)


if __name__ == "__main__":
    unittest.main()
