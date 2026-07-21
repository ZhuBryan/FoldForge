"""Flat-foldability theorems: can this pattern fold completely flat?

Two classic necessary conditions, both checked per interior vertex:

Kawasaki's theorem
    Sweep around an interior vertex and label the wedge angles a1, a2, a3, ...
    The pattern can fold flat *only if* the alternating sum is zero:
        a1 - a2 + a3 - a4 + ... = 0
    Because the wedges always sum to 360 degrees, that is the same as saying
    the odd-numbered wedges total 180 and the even-numbered ones total 180.
    (This also requires an even number of creases at the vertex.)

Maekawa's theorem
    At an interior vertex of a flat-foldable pattern, the number of mountain
    folds minus the number of valley folds is exactly +2 or -2:
        |M - V| = 2

These are *necessary* conditions, not sufficient ones: passing both does not
prove a pattern folds flat globally (layers can still collide). But failing
either proves it cannot. That makes them a fast, honest first screen.
"""

from __future__ import annotations

from dataclasses import dataclass

from foldforge.geometry.crease_graph import CreasePattern

# Default angular tolerance in degrees. Generators and hand-made files carry a
# little floating-point and rounding noise, so we don't demand exact equality.
DEFAULT_TOL = 1e-6


def check_kawasaki(
    pattern: CreasePattern, vertex: int, tol: float = DEFAULT_TOL
) -> bool | None:
    """Does ``vertex`` satisfy Kawasaki's theorem?

    Returns True/False for an interior vertex, or ``None`` if the theorem does
    not apply (a border vertex, or a vertex with no creases).
    """
    if pattern.is_boundary_vertex(vertex):
        return None
    sectors = pattern.sector_angles(vertex)
    if not sectors:
        return None
    if len(sectors) % 2 != 0:
        # An odd number of creases can never split into two equal alternating
        # groups, so flat-foldability is impossible here.
        return False
    odd_sum = sum(sectors[0::2])
    even_sum = sum(sectors[1::2])
    return abs(odd_sum - even_sum) < tol


def check_maekawa(pattern: CreasePattern, vertex: int) -> bool | None:
    """Does ``vertex`` satisfy Maekawa's theorem (|M - V| == 2)?

    Returns True/False for an interior vertex whose creases are all assigned
    M or V. Returns ``None`` if the theorem does not apply (border vertex) or
    cannot be judged yet (some incident crease is still F/U/unassigned).
    """
    if pattern.is_boundary_vertex(vertex):
        return None
    creases = pattern.vertex_creases(vertex)
    if not creases:
        return None
    kinds = [a for _, a, _ in creases]
    if any(k not in ("M", "V") for k in kinds):
        return None  # unknown/flat crease present -> can't decide
    mountains = kinds.count("M")
    valleys = kinds.count("V")
    return abs(mountains - valleys) == 2


@dataclass
class VertexReport:
    """Per-vertex foldability result. ``None`` means 'theorem does not apply'."""

    vertex: int
    kawasaki: bool | None
    maekawa: bool | None

    @property
    def ok(self) -> bool:
        """True if no applicable check failed (None counts as 'not failed')."""
        return self.kawasaki is not False and self.maekawa is not False


@dataclass
class FoldabilityReport:
    """Whole-pattern foldability summary, with the per-vertex detail kept."""

    vertices: list[VertexReport]

    @property
    def flat_foldable(self) -> bool:
        """True if every interior vertex passes the checks that apply to it.

        Necessary, not sufficient: see the module docstring.
        """
        return all(v.ok for v in self.vertices)

    @property
    def failures(self) -> list[VertexReport]:
        return [v for v in self.vertices if not v.ok]

    def summary(self) -> str:
        n_interior = len(self.vertices)
        n_fail = len(self.failures)
        verdict = "flat-foldable (passes both theorems)" if self.flat_foldable else (
            f"NOT flat-foldable: {n_fail} of {n_interior} interior vertices fail"
        )
        lines = [verdict]
        for v in self.failures:
            lines.append(
                f"  vertex {v.vertex}: kawasaki={v.kawasaki} maekawa={v.maekawa}"
            )
        return "\n".join(lines)


def foldability_report(
    pattern: CreasePattern, tol: float = DEFAULT_TOL
) -> FoldabilityReport:
    """Run Kawasaki + Maekawa on every interior vertex of ``pattern``."""
    reports = [
        VertexReport(
            vertex=v,
            kawasaki=check_kawasaki(pattern, v, tol),
            maekawa=check_maekawa(pattern, v),
        )
        for v in pattern.interior_vertices()
    ]
    return FoldabilityReport(vertices=reports)
