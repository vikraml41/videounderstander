import unittest

from vu.sampling import (classify_transition, parse_pgm, segment_frames,
                         select_keepers, tile_diffs)

W, H = 64, 32


def flat(value: int) -> bytes:
    return bytes([value]) * (W * H)


def with_patch(base: bytes, x0: int, y0: int, w: int, h: int,
               value: int) -> bytes:
    buf = bytearray(base)
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            buf[y * W + x] = value
    return bytes(buf)


class TestPgm(unittest.TestCase):
    def test_parse_roundtrip(self):
        pixels = bytes(range(256)) * (W * H // 256)
        data = f"P5\n# comment\n{W} {H}\n255\n".encode() + pixels
        w, h, px = parse_pgm(data)
        self.assertEqual((w, h), (W, H))
        self.assertEqual(px, pixels)

    def test_rejects_non_p5(self):
        with self.assertRaises(ValueError):
            parse_pgm(b"P2\n2 2\n255\n0 0 0 0")


class TestClassification(unittest.TestCase):
    def test_identical_is_stable(self):
        d = tile_diffs(flat(100), flat(100), W, H)
        self.assertEqual(classify_transition(d), "stable")

    def test_localized_change_is_build(self):
        a = flat(200)
        b = with_patch(a, 0, 0, 8, 8, 0)  # one dark patch = new bullet
        self.assertEqual(classify_transition(tile_diffs(a, b, W, H)), "build")

    def test_global_change_is_cut(self):
        self.assertEqual(
            classify_transition(tile_diffs(flat(200), flat(20), W, H)), "cut")

    def test_global_brightness_shift_is_stable(self):
        # every tile moves the same amount; median subtraction cancels it
        a = flat(100)
        b = flat(115)
        self.assertEqual(classify_transition(tile_diffs(a, b, W, H)), "stable")

    def test_wobble_plus_local_change_is_build(self):
        # small global shift everywhere plus one big local change
        a = flat(100)
        b = with_patch(flat(106), 0, 0, 8, 8, 250)
        self.assertEqual(classify_transition(tile_diffs(a, b, W, H)), "build")


class TestSegmentation(unittest.TestCase):
    def test_accretion_groups_and_final_frame(self):
        base = flat(220)
        step1 = with_patch(base, 0, 0, 8, 4, 0)      # bullet 1
        step2 = with_patch(step1, 0, 8, 8, 4, 0)     # bullet 2
        other = flat(30)                             # scene cut
        frames = [base, base, step1, step1, step1, step2, other, other]
        ivs = segment_frames(frames, W, H)
        self.assertEqual(len(ivs), 4)
        # builds stay in group 0; cut starts group 1
        self.assertEqual([iv.group for iv in ivs], [0, 0, 0, 1])
        self.assertEqual([iv.entered_by for iv in ivs],
                         ["start", "build", "build", "cut"])
        # final frame of each stable interval is the keeper
        self.assertEqual([iv.end_idx for iv in ivs], [1, 4, 5, 7])

    def test_transition_junk_dropped(self):
        a, junk, b = flat(220), flat(128), flat(30)
        frames = [a, a, junk, b, b]
        ivs = select_keepers(segment_frames(frames, W, H))
        # the single-frame cut->cut interval (the fade frame) is dropped
        self.assertEqual(len(ivs), 2)
        self.assertEqual([iv.end_idx for iv in ivs], [1, 4])

    def test_empty(self):
        self.assertEqual(segment_frames([], W, H), [])


if __name__ == "__main__":
    unittest.main()
