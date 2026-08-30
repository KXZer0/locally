"""Single source of truth for the `locally` mark.

Emits the SVG path data used by the inline sprite and the favicon, and
rasterises the PNG/ICO tiles from the same numbers. The geometry itself lives
in mark.py, fitted to the original artwork — read the module docstring there
before changing any of it.

The mark is a filled ring (a three-lobed outer edge with a three-lobed hole)
plus a free-floating circle. It is NOT stroked: an earlier version drew
concentric hairlines, which is a different mark, and thin strokes disintegrate
in a 16px taskbar icon anyway.

Sizing: mark.py's coordinates carry the original PNG's own padding, so the ink
spans only 79.6 of its 100-unit box. `normalise()` rescales the shape about its
bounding-box centre to span INK_SPAN and re-centres it on (50, 50), so the
sprite fills whatever box CSS gives it and the tile padding is decided here in
one place rather than baked into the curve.

Run:  python scripts/build_mark.py static/icons
"""
import io
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mark as M

INK_SPAN = 96.0      # how much of the 100-unit viewBox the widest ink spans

# Bezier segments per curve. Cubic Hermite error falls off roughly as h^4, and
# 12 segments left the outer edge 0.33 units off the true curve -- 1.7px once
# the icon is drawn at 512. The 6th harmonic is what costs the accuracy: it
# puts three extra inflections in a curve that 12 spans have to bridge. 24
# brings it under a fiftieth of a unit and costs about a kilobyte of path data.
SEGMENTS = 24


def normalise(ink_span=INK_SPAN):
    """Scale the measured curves to fill the viewBox, centred on the bbox.

    Returns (centre, outer, inner, core_r) in the rescaled space. The bounding
    box is what has to look centred in a square container — the polar centre
    sits 7 units below it, because the lobes are not symmetric top to bottom.
    """
    x0, y0, x1, y1 = M.bbox()
    s = ink_span / max(x1 - x0, y1 - y0)
    bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2

    def scale(spec):
        return {k: v * s for k, v in spec.items()}

    centre = (50.0 + (M.CENTRE[0] - bcx) * s, 50.0 + (M.CENTRE[1] - bcy) * s)
    return centre, scale(M.OUTER), scale(M.INNER), M.CORE_R * s


CENTRE, OUTER, INNER, CORE_R = normalise()


def ring_path(segments=SEGMENTS):
    return M.ring_path(centre=CENTRE, segments=segments, outer=OUTER, inner=INNER)


def core_attrs():
    return CENTRE[0], CENTRE[1], CORE_R


# --- rasteriser -------------------------------------------------------------
def mark_image(size, rgb=(255, 255, 255), ss=6):
    """Draw the mark at `size` px. Supersampled, since the lobes are curves."""
    S = size * ss
    k = S / 100.0
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    d.polygon([(x * k, y * k) for x, y in M.sample(OUTER, CENTRE, n=1440)], fill=255)
    d.polygon([(x * k, y * k) for x, y in M.sample(INNER, CENTRE, n=1440)], fill=0)
    cx, cy, r = core_attrs()
    d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], fill=255)
    alpha = img.resize((size, size), Image.LANCZOS)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = np.array(alpha)
    return Image.fromarray(out, "RGBA")


def tile(size, pad_frac=0.155, radius_frac=0.225, bg=(10, 10, 10)):
    """App-icon tile: the canvas colour as a rounded square, white mark on top."""
    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        [0, 0, S - 1, S - 1], radius=S * radius_frac, fill=bg + (255,))
    inner = int(S * (1 - 2 * pad_frac))
    img.alpha_composite(mark_image(inner, ss=3), ((S - inner) // 2,) * 2)
    return img.resize((size, size), Image.LANCZOS)


def write_ico(path, sizes=(16, 20, 24, 32, 40, 48, 64, 96, 128, 256)):
    """Write a multi-resolution .ico with per-size artwork.

    PIL's ICO writer downsamples everything from one image, and passing a small
    base with append_images silently writes a single 16x16 entry that Windows
    then upscales into a blur. Each entry is rendered at its own size here and
    the directory assembled by hand. PNG payloads throughout: supported since
    Vista and what every current toolchain emits.
    """
    blobs = []
    for s in sizes:
        buf = io.BytesIO()
        tile(s).save(buf, format="PNG", optimize=True)
        blobs.append(buf.getvalue())
    out = [struct.pack("<HHH", 0, 1, len(sizes))]
    offset = 6 + 16 * len(sizes)
    for s, blob in zip(sizes, blobs):
        d = 0 if s >= 256 else s          # 0 means 256 in the ICO directory
        out.append(struct.pack("<BBBBHHII", d, d, 0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    with open(path, "wb") as fh:
        fh.write(b"".join(out) + b"".join(blobs))


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path fill="#fff" d="{ring}"/>
  <circle fill="#fff" cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}"/>
</svg>
"""

# The core circle's (cx, cy) is the curves' shared origin, and normalise() puts
# it BELOW the box centre (it centres the bounding box, not the origin). Anything
# that rotates the mark must use that point: static/css/03-thread.css hard-codes
# it as `transform-origin: 50% 58.445%`, so if these numbers move, move that too.
SPRITE = """  <symbol id="i-mark" viewBox="0 0 100 100" fill="currentColor" stroke="none">
    <path d="{ring}"/>
    <circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}"/>
  </symbol>"""


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "static/icons"
    cx, cy, r = core_attrs()

    with open(f"{out}/locally.svg", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(SVG.format(ring=ring_path(), cx=cx, cy=cy, r=r))

    for s in (44, 64, 144, 192, 256, 512):
        tile(s).save(f"{out}/locally-{s}.png")
    write_ico(f"{out}/locally.ico")

    sheet = Image.new("RGBA", (620, 220), (18, 18, 18, 255))
    x = 16
    for s in (128, 64, 48, 32, 24, 16):
        sheet.alpha_composite(tile(s), (x, 16))
        sheet.alpha_composite(mark_image(s), (x, 170 + (32 - s) // 2))
        x += s + 20
    sheet.save(f"{out}/preview.png")

    print("--- paste into the sprite in templates/index.html ---")
    print(SPRITE.format(ring=ring_path(), cx=cx, cy=cy, r=r))
    print("\nassets written to", out)
