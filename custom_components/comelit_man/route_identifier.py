"""Identify which entrance panel is streaming from a decoded video frame.

Field-observed 2026-11-20: the two physical panels (home entrance and
urbanización entrance) sit behind a single selector relay, so only one of
them is connected at a time. Neither the protocol nor the RTP stream says
which one: the panel re-arms its encoder on every call, so the SSRC is
fresh per session and cannot be mapped to a panel.

What is stable is the *view*. Each panel's camera looks at a different
scene, and a 64-bit difference hash (dHash) of one decoded frame is enough
to tell them apart while surviving the JPEG re-encode, the resolution and
ordinary exposure changes. The integration keeps one fingerprint per slot
and compares it with the first frame of every session, so the camera
selector reflects the panel that is really streaming instead of a guess.
"""

from __future__ import annotations

import io
import logging

_LOGGER = logging.getLogger(__name__)

# dHash grid: 9x8 luminance samples produce 8x8 horizontal gradients = 64 bits.
HASH_SIDE = 8
# Maximum Hamming distance still considered the same scene.
MATCH_THRESHOLD = 14
# The best candidate must beat the runner-up by this margin; otherwise the
# frame is ambiguous (night view, camera covered) and guessing would be
# worse than saying "unknown" — a wrong route strikes the wrong door.
MATCH_MARGIN = 8
# A view this far from the only known reference can only be the other
# entrance; it is learned as that slot's reference on first sight.
LEARN_DISTANCE = 24


def fingerprint(jpeg: bytes) -> int | None:
    """Return the 64-bit dHash of a JPEG frame, or None if it cannot decode."""
    try:
        from PIL import Image  # Pillow ships with Home Assistant
    except ImportError:  # pragma: no cover - HA always has Pillow
        _LOGGER.error("Pillow is required for entrance route identification")
        return None
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            small = image.convert("L").resize(
                (HASH_SIDE + 1, HASH_SIDE), _resampling()
            )
            pixels = small.tobytes()
    except Exception:
        _LOGGER.debug("Cannot fingerprint frame", exc_info=True)
        return None
    if len(pixels) < (HASH_SIDE + 1) * HASH_SIDE:
        return None
    bits = 0
    for row in range(HASH_SIDE):
        base = row * (HASH_SIDE + 1)
        for col in range(HASH_SIDE):
            bits <<= 1
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1
    return bits


def hamming(first: int, second: int) -> int:
    """Return the number of differing bits between two fingerprints."""
    return (first ^ second).bit_count()


def identify(
    sample: int,
    references: dict[str, int],
    threshold: int = MATCH_THRESHOLD,
    margin: int = MATCH_MARGIN,
) -> tuple[str | None, int | None]:
    """Return ``(slot, distance)`` for the reference that matches ``sample``.

    ``(None, None)`` means "do not guess": there are no references yet, the
    closest one is farther than ``threshold``, or the two references are
    closer to each other than ``margin`` (ambiguous view).
    """
    if not references:
        return None, None
    ranked = sorted(
        (hamming(sample, reference), slot)
        for slot, reference in references.items()
    )
    best_distance, best_slot = ranked[0]
    if best_distance > threshold:
        return None, None
    if len(ranked) > 1 and ranked[1][0] - best_distance < margin:
        return None, None
    return best_slot, best_distance


def _resampling() -> int:
    """Return the bilinear resampling filter constant for this Pillow."""
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")
