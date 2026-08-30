"""Geometry for the `locally` mark, measured from the original artwork.

The mark is NOT a rounded triangle. It is a ring whose two edges are both
three-lobed polar curves, plus a free-floating circle at the centre. That is
why the silhouette bulges at the corners and curves *inward* between them. An
earlier reconstruction used a rounded triangle — straight edges joined by
circular corner arcs — which read far more angular and technical than the real
thing. See TODONT.md.

Both edges fit

    r(theta) = a0 + a3 * sin(3*theta) + a6 * cos(6*theta)

Fitted by least squares to the radial profile of `locally logo.png`
(1000x1000, shape carried in the alpha channel, decomposed into components
with a connected-component pass). A 9th harmonic was measured and discarded:
its amplitude is 0.017 units, or 0.17px on the 1000px original.

    ring, outer edge   a0=35.06  a3=7.950  a6=-0.819    fit RMS 3.9 px
    ring, inner edge   a0=22.01  a3=1.772  a6=+0.185    fit RMS 1.0 px
    core               true circle, r=11.43, circularity 0.997
    shared centre      (49.95, 57.15)

Two relationships matter if the proportions are ever retuned. The inner edge
lobes far less than the outer one (a3/a0 of 0.08 against 0.227), so the band is
thick across the middle of each side and thin at the lobes — equal lobing turns
it into a plain ring. And a6 flips sign between the two edges, which is what
flattens the outer sides while rounding the inner ones.

Curves are emitted as cubic Beziers rather than sampled polylines: a smooth
polar curve converts exactly via Hermite tangents, so twelve segments hold the
shape to well under a pixel at any size this icon is ever drawn.
"""
import math

# --- measured from the original, normalised to a 100-unit box ---------------
CENTRE = (49.95, 57.15)
OUTER = dict(a0=35.0581, a3=7.9499, a6=-0.8186)
INNER = dict(a0=22.0097, a3=1.7715, a6=0.1850)
CORE_R = 11.433


def radius(t, a0, a3, a6):
    return a0 + a3 * math.sin(3 * t) + a6 * math.cos(6 * t)


def _dradius(t, a0, a3, a6):
    return 3 * a3 * math.cos(3 * t) - 6 * a6 * math.sin(6 * t)


def _point(t, spec, centre):
    r = radius(t, **spec)
    return (centre[0] + r * math.cos(t), centre[1] + r * math.sin(t))


def _tangent(t, spec):
    """dP/dtheta, used for the Bezier control points."""
    r, dr = radius(t, **spec), _dradius(t, **spec)
    return (dr * math.cos(t) - r * math.sin(t),
            dr * math.sin(t) + r * math.cos(t))


def lobe_path(spec, centre=CENTRE, segments=12, reverse=False):
    """SVG path data for one three-lobed curve.

    `reverse` walks it the other way round, which is how the inner edge cuts a
    hole under the default nonzero fill rule.
    """
    ts = [2 * math.pi * i / segments for i in range(segments + 1)]
    if reverse:
        ts = ts[::-1]
    p0 = _point(ts[0], spec, centre)
    d = [f"M{p0[0]:.3f} {p0[1]:.3f}"]
    for i in range(segments):
        t0, t1 = ts[i], ts[i + 1]
        h = (t1 - t0) / 3.0
        a, b = _point(t0, spec, centre), _point(t1, spec, centre)
        da, db = _tangent(t0, spec), _tangent(t1, spec)
        c1 = (a[0] + h * da[0], a[1] + h * da[1])
        c2 = (b[0] - h * db[0], b[1] - h * db[1])
        d.append(f"C{c1[0]:.3f} {c1[1]:.3f} {c2[0]:.3f} {c2[1]:.3f}"
                 f" {b[0]:.3f} {b[1]:.3f}")
    d.append("Z")
    return "".join(d)


def ring_path(centre=CENTRE, segments=12, outer=None, inner=None):
    """The ring: outer edge, then the inner edge wound the opposite way."""
    return (lobe_path(outer or OUTER, centre, segments) +
            lobe_path(inner or INNER, centre, segments, reverse=True))


def sample(spec, centre=CENTRE, n=720):
    """Dense polyline of one curve, for rasterising and for tests."""
    return [_point(2 * math.pi * i / n, spec, centre) for i in range(n)]


def bbox(spec=None, centre=CENTRE):
    pts = sample(spec or OUTER, centre)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


if __name__ == "__main__":
    x0, y0, x1, y1 = bbox()
    print(f"outer bbox  x {x0:.2f}..{x1:.2f}  y {y0:.2f}..{y1:.2f}"
          f"   ({x1-x0:.2f} x {y1-y0:.2f})")
    print(f"bbox centre ({(x0+x1)/2:.2f}, {(y0+y1)/2:.2f})")
