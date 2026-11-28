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
from collections.abc import Mapping, Sequence

_LOGGER = logging.getLogger(__name__)

# dHash grid: 9x8 luminance samples produce 8x8 horizontal gradients = 64 bits.
HASH_SIDE = 8
# Maximum Hamming distance still considered the same scene.
# 20 (of 64 bits) tolerates the exposure/glare swing of the same camera between
# morning and midday (measured: 16-18 bits on the hedge view) while the two
# entrances stay 23-29 bits apart from each other.  The margin below is what
# turns that tolerance into a safe decision: a view must still beat the other
# entrance by MATCH_MARGIN bits, otherwise it is ambiguous and refused.
MATCH_THRESHOLD = 20
# The best candidate must beat the runner-up by this margin; otherwise the
# frame is ambiguous (night view, camera covered) and guessing would be
# worse than saying "unknown" — a wrong route strikes the wrong door.
MATCH_MARGIN = 8
# A view this far from the only known reference can only be the other
# entrance; it is learned as that slot's reference on first sight.
LEARN_DISTANCE = 24
# A frame with almost no horizontal gradient (black screen, covered lens,
# uniform exposure) hashes to a degenerate value. Storing or matching it
# would let a blank frame impersonate an entrance and strike the wrong
# door, so such frames are not fingerprints at all.
MIN_HASH_BITS = 4
MAX_HASH_BITS = 60


def is_usable_hash(value: int) -> bool:
    """Return True when a dHash carries enough structure to identify a view."""
    return MIN_HASH_BITS <= int(value).bit_count() <= MAX_HASH_BITS


def fingerprint(jpeg: bytes) -> int | None:
    """Return the 64-bit dHash of a JPEG frame, or None if it cannot decode.

    ``None`` is also returned for degenerate frames (a black or uniform
    picture): an all-zero hash would otherwise match any other blank frame
    at distance 0.
    """
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
    if not is_usable_hash(bits):
        _LOGGER.debug(
            "Skipping uninformative frame (dHash 0x%016X, %d bits)",
            bits,
            bits.bit_count(),
        )
        return None
    return bits


def hamming(first: int, second: int) -> int:
    """Return the number of differing bits between two fingerprints."""
    return (first ^ second).bit_count()


def identify(
    sample: int,
    references: Mapping[str, int | Sequence[int]],
    threshold: int = MATCH_THRESHOLD,
    margin: int = MATCH_MARGIN,
) -> tuple[str | None, int | None]:
    """Return ``(slot, distance)`` for the reference that matches ``sample``.

    Each slot may carry one fingerprint (``int``) or several (day/night or
    different exposure views) as a sequence; the slot's distance is the
    closest of its references.

    ``(None, None)`` means "do not guess": there are no references yet, the
    closest one is farther than ``threshold``, or the two references are
    closer to each other than ``margin`` (ambiguous view).
    """
    distances = []
    for slot, value in references.items():
        hashes = (value,) if isinstance(value, int) else tuple(value)
        if not hashes:
            continue
        distances.append((min(hamming(sample, hashed) for hashed in hashes), slot))
    if not distances:
        return None, None
    ranked = sorted(distances)
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
