import unittest

from vu.transcribe import (full_text, parse_vtt_or_srt, parse_whisper_json,
                           transcript_window)

VTT = """WEBVTT

00:01.000 --> 00:04.000 line:0
so if you look at <b>this curve</b>

00:04.000 --> 00:07.500
the interesting part isn't the peak

3
00:07.500 --> 01:02.000
it's how fast it decays after
"""

SRT = """1
00:00:01,000 --> 00:00:04,000
so if you look at this curve

2
00:00:04,000 --> 00:00:07,500
the interesting part isn't the peak
"""


class TestSubtitleParsing(unittest.TestCase):
    def test_vtt(self):
        segs = parse_vtt_or_srt(VTT)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0]["start"], 1.0)
        self.assertEqual(segs[0]["text"], "so if you look at this curve")
        self.assertEqual(segs[2]["end"], 62.0)

    def test_srt(self):
        segs = parse_vtt_or_srt(SRT)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[1]["start"], 4.0)
        self.assertEqual(segs[1]["end"], 7.5)

    def test_autocaption_repeats_merged(self):
        vtt = ("WEBVTT\n\n00:01.000 --> 00:02.000\nhello world\n\n"
               "00:02.000 --> 00:03.000\nhello world\n\n"
               "00:03.000 --> 00:04.000\nnext line\n")
        segs = parse_vtt_or_srt(vtt)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["end"], 3.0)

    def test_whisper_json(self):
        obj = {"segments": [
            {"start": 0.0, "end": 2.0, "text": " hello  there ",
             "words": [{"start": 0.0, "end": 1.0, "word": " hello"},
                       {"start": 1.0, "end": 2.0, "word": " there"}]}]}
        segs = parse_whisper_json(obj)
        self.assertEqual(segs[0]["text"], "hello there")
        self.assertEqual(segs[0]["words"][1]["word"], "there")


class TestWindows(unittest.TestCase):
    def test_transcript_window(self):
        segs = parse_vtt_or_srt(VTT)
        # centered at t=5: all three segments overlap ±30s
        self.assertIn("decays", transcript_window(segs, 5.0))
        # centered at t=50 with tiny radius: only the long last segment
        window = transcript_window(segs, 50.0, radius=5)
        self.assertEqual(window, "it's how fast it decays after")

    def test_full_text(self):
        segs = parse_vtt_or_srt(SRT)
        self.assertEqual(
            full_text(segs),
            "so if you look at this curve the interesting part isn't the peak")


if __name__ == "__main__":
    unittest.main()
