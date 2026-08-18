import unittest

from vu.interleave import build_events, render_aligned

MANIFEST = [
    {"t": 10.0, "ts": "00:10", "interval_start": 4.0, "interval_end": 10.0,
     "file": "frames/frame_000010.jpg", "group": 0, "entered_by": "start"},
    {"t": 31.0, "ts": "00:31", "interval_start": 12.0, "interval_end": 31.0,
     "file": "frames/frame_000031.jpg", "group": 0, "entered_by": "build"},
]

SEGMENTS = [
    {"start": 4.0, "end": 11.0, "text": "so if you look at this curve"},
    {"start": 12.0, "end": 19.0, "text": "the interesting part isn't the peak"},
]


class TestInterleave(unittest.TestCase):
    def test_frame_precedes_speech_at_same_time(self):
        events = build_events(MANIFEST, SEGMENTS)
        # both start at t=4.0; the frame must come first so speech about the
        # visual follows the visual
        self.assertEqual(events[0]["kind"], "frame")
        self.assertEqual(events[1]["kind"], "speech")

    def test_render_order_and_markers(self):
        text = render_aligned(MANIFEST, SEGMENTS)
        lines = [l for l in text.splitlines() if l]
        frame1 = next(i for i, l in enumerate(lines) if "frame_000010" in l)
        speech1 = next(i for i, l in enumerate(lines) if "this curve" in l)
        frame2 = next(i for i, l in enumerate(lines) if "frame_000031" in l)
        speech2 = next(i for i, l in enumerate(lines) if "peak" in l)
        self.assertLess(frame1, speech1)
        self.assertLess(speech1, frame2)
        self.assertLess(frame2, speech2)
        # accretion marker on the build frame
        self.assertIn("accretion group 0", lines[frame2])
        # speech carries a time span
        self.assertTrue(lines[speech1].startswith("[00:04–00:11]"))

    def test_descriptions_injected(self):
        descs = {"frames/frame_000010.jpg": {
            "description": "exponential decay curve",
            "pointer_target": "the tail of the curve"}}
        text = render_aligned(MANIFEST, SEGMENTS, descs)
        self.assertIn("frame content: exponential decay curve", text)
        self.assertIn("pointer on: the tail of the curve", text)


if __name__ == "__main__":
    unittest.main()
