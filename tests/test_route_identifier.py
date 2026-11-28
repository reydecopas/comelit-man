"""Unit tests for entrance route identification (frame fingerprinting)."""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL")

from PIL import Image, ImageDraw  # noqa: E402

from custom_components.comelit_man.route_identifier import (  # noqa: E402
    LEARN_DISTANCE,
    MATCH_MARGIN,
    MATCH_THRESHOLD,
    fingerprint,
    hamming,
    identify,
    is_noise_hash,
)


def _jpeg(pattern: str, size: tuple[int, int] = (320, 240), quality: int = 85) -> bytes:
    """Render one of the two field views as JPEG bytes.

    The patterns are deliberately large-scale (horizontal bands vs a strong
    vertical edge): that is what a dHash sees, and it makes the fixture
    independent of resolution and JPEG quality, like the real scenes are.
    """
    width, height = size
    image = Image.new("RGB", size, (128, 128, 128))
    draw = ImageDraw.Draw(image)
    if pattern == "hedge":
        draw.rectangle([0, 0, width, int(height * 0.62)], fill=(30, 70, 30))
        draw.rectangle([0, int(height * 0.62), width, int(height * 0.86)], fill=(60, 110, 50))
        draw.rectangle([0, int(height * 0.86), width, height], fill=(200, 195, 180))
        # A trunk breaks the horizontal bands: a real hedge has depth, and a
        # frame made only of horizontal bands hashes to all-zero (which is not
        # a usable fingerprint).
        draw.rectangle(
            [int(width * 0.30), 0, int(width * 0.42), int(height * 0.86)],
            fill=(70, 50, 35),
        )
    else:
        draw.rectangle([0, 0, int(width * 0.48), height], fill=(220, 220, 220))
        draw.rectangle([int(width * 0.48), 0, int(width * 0.62), height], fill=(150, 150, 150))
        draw.rectangle([int(width * 0.62), 0, width, height], fill=(40, 60, 40))
        draw.rectangle([0, 0, int(width * 0.2), int(height * 0.2)], fill=(255, 255, 255))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality)
    return out.getvalue()


class TestFingerprint:
    def test_same_frame_is_stable(self):
        assert fingerprint(_jpeg("hedge")) == fingerprint(_jpeg("hedge"))

    def test_reencode_keeps_it_close(self):
        best = fingerprint(_jpeg("street", quality=90))
        other = fingerprint(_jpeg("street", quality=55))
        assert hamming(best, other) <= MATCH_THRESHOLD

    def test_resolution_does_not_matter(self):
        best = fingerprint(_jpeg("hedge", size=(640, 480)))
        other = fingerprint(_jpeg("hedge", size=(320, 240)))
        assert hamming(best, other) <= MATCH_THRESHOLD

    def test_distinct_scenes_are_far_apart(self):
        hedge = fingerprint(_jpeg("hedge"))
        street = fingerprint(_jpeg("street"))
        assert hamming(hedge, street) >= LEARN_DISTANCE

    def test_invalid_data_returns_none(self):
        assert fingerprint(b"not a jpeg") is None

    def test_blank_frame_is_not_a_fingerprint(self):
        # A black or uniform frame has no horizontal structure. Returning an
        # all-zero hash would let a dark stream match any other blank frame at
        # distance 0 and strike the wrong door.
        image = Image.new("RGB", (320, 240), (0, 0, 0))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85)
        assert fingerprint(out.getvalue()) is None

    def test_uniform_grey_frame_is_not_a_fingerprint(self):
        image = Image.new("RGB", (320, 240), (128, 128, 128))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85)
        assert fingerprint(out.getvalue()) is None


class TestNoise:
    """A dead/noisy input must never become a route reference."""

    def test_alternating_patterns_are_noise(self):
        assert is_noise_hash(0xAAAAAAAAAAAAAAAA)
        assert is_noise_hash(0x5555555555555555)
        assert is_noise_hash(0xAAAAAAAA8AAAAAAA)

    def test_real_views_are_not_noise(self):
        assert not is_noise_hash(fingerprint(_jpeg("hedge")))
        assert not is_noise_hash(fingerprint(_jpeg("street")))

    def test_noisy_frame_is_not_a_fingerprint(self):
        # 1px alternating columns are what the field saw from the far camera
        # while it warmed up (dHash 0xAAAA...). It must be skipped, not hashed
        # into a reference or counted as an inconclusive match.
        image = Image.new("L", (9, 8))
        pixels = image.load()
        for y in range(8):
            for x in range(9):
                pixels[x, y] = 255 if x % 2 == 0 else 0
        out = io.BytesIO()
        image.save(out, format="PNG")
        assert fingerprint(out.getvalue()) is None


class TestIdentify:
    def test_no_references_never_guesses(self):
        assert identify(0b1010, {}) == (None, None)
    def test_exact_match(self):
        slot, distance = identify(0x0000, {"cam1": 0x0000, "cam2": 0xFFFFFFFF})
        assert (slot, distance) == ("cam1", 0)

    def test_close_match_with_clear_margin(self):
        slot, distance = identify(0x0003, {"cam1": 0x0000, "cam2": 0xFFFFFFFF})
        assert slot == "cam1"
        assert distance == 2

    def test_beyond_threshold_is_unknown(self):
        at_threshold = (1 << MATCH_THRESHOLD) - 1  # exactly MATCH_THRESHOLD bits
        assert identify(at_threshold, {"cam1": 0x0000})[0] == "cam1"
        beyond = (1 << (MATCH_THRESHOLD + 1)) - 1  # one bit past the threshold
        assert hamming(beyond, 0x0000) == MATCH_THRESHOLD + 1
        assert identify(beyond, {"cam1": 0x0000}) == (None, None)

    def test_ambiguous_view_is_unknown(self):
        sample = 0x0F0F  # 8 bits from each 16-bit reference
        assert hamming(sample, 0x0000) == hamming(sample, 0xFFFF)
        assert identify(sample, {"cam1": 0x0000, "cam2": 0xFFFF}) == (None, None)

    def test_margin_is_configurable(self):
        sample = 0x0007  # 3 bits from cam1, 13 bits from cam2
        assert identify(sample, {"cam1": 0x0000, "cam2": 0xFFFF}, margin=MATCH_MARGIN) == (
            "cam1",
            3,
        )
        # Only one bit behind cam1: too close to call.
        assert identify(sample, {"cam1": 0x0000, "cam2": 0x000F}, margin=MATCH_MARGIN) == (
            None,
            None,
        )

    def test_multiple_references_per_slot(self):
        # cam1 keeps a day view and a night view; a night sample matches the
        # closest of them instead of falling outside the threshold.
        references = {"cam1": [0x0000, 0x00FF], "cam2": 0xFFFFFFFFFFFFFFFF}
        assert hamming(0x00FE, 0x00FF) == 1
        assert identify(0x00FE, references) == ("cam1", 1)

    def test_empty_slot_reference_is_ignored(self):
        assert identify(0x0000, {"cam1": [], "cam2": 0x3FFF}) == ("cam2", 14)
