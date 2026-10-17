"""Global layer-ordering / self-intersection validation.

Three things are pinned here:

* a pattern that passes the *local* flat-foldability theorems at every vertex
  yet has **no** valid global layer order (the paper would self-intersect) is
  reported ``valid=False`` -- the whole reason this module exists;
* a genuinely orderable pattern returns ``valid=True`` with a concrete order;
* every order the solver hands back is re-checked, independently, to violate
  none of the taco-taco / taco-tortilla constraints.

All deterministic and fast (no randomness, no timing assumptions).
"""

import numpy as np
import pytest

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.geometry.foldability import foldability_report
from foldforge.geometry.layer_order import (
    layer_order, globally_flat_foldable, order_violations,
    self_intersections_in_order, _build_constraints,
)
from foldforge.design.classic import bird_base
from foldforge.design.treemaker import get_tree, design_base
from foldforge.design.lang_molecule import flat_fold_image
from foldforge.geometry.examples import single_vertex, miura


def pinwheel(d: float = 1.7, skew: float = 0.0) -> CreasePattern:
    """Triangular base with three flaps folded inward past the centre.

    For small ``d`` the flaps tuck neatly (orderable).  Once ``d`` is large
    enough that each folded flap reaches across the base and covers the *next*
    flap's hinge crease, the three flaps are cyclically trapped: flap j must lie
    outside the (base, flap i) taco for every pair, which no stack can satisfy.
    The sheet still folds flat geometrically (one crease per flap -> a tree, so
    closure is exact) and has no interior vertices, so the local theorems find
    nothing to reject.  That is the local-OK / global-FAIL gap.
    """
    ang = np.radians([90.0, 210.0, 330.0])
    base = np.array([[np.cos(a), np.sin(a)] for a in ang])
    verts = list(base)
    faces = [[0, 1, 2]]
    edges: list[tuple[int, int]] = []
    assign: list[str] = []

    def add(u, v, kind):
        edges.append((u, v))
        assign.append(kind)

    for i in range(3):
        j = (i + 1) % 3
        p0, p1 = base[i], base[j]
        mid = (p0 + p1) / 2
        out = mid / np.linalg.norm(mid)
        edge = (p1 - p0) / np.linalg.norm(p1 - p0)
        apex = mid + out * d + edge * skew
        verts.append(apex)
        ai = len(verts) - 1
        faces.append([j, i, ai])          # reversed winding -> shared edge (j,i)
        add(i, j, "V")                    # hinge crease base|flap
        add(j, ai, "B")
        add(ai, i, "B")

    return CreasePattern(
        vertices=np.array(verts, float), edges=np.array(edges),
        assignment=assign, faces=faces, metadata={"name": "pinwheel3"},
    )


def test_local_ok_but_globally_unorderable():
    """The headline: local theorems pass, global layer order does not exist."""
    p = pinwheel(d=1.7)
    # Local screen is satisfied (nothing to reject -> flat-foldable).
    assert foldability_report(p).flat_foldable is True
    # ...but no global stacking order exists.
    r = layer_order(p)
    assert r.valid is False
    assert r.order is None
    # It fails for the right reason: cyclic taco-tortilla constraints.
    assert "self-intersection" in r.note
    # The stronger verdict agrees.
    assert globally_flat_foldable(p) is False


def test_pinwheel_small_d_is_orderable():
    """The same construction with a modest fold is orderable (sanity floor)."""
    p = pinwheel(d=0.6)
    r = layer_order(p)
    assert r.valid is True
    assert order_violations(r.order, p) == []


def test_simple_pattern_is_orderable_and_order_is_consistent():
    """A plain flat-foldable pattern: valid order, and the order is clean."""
    for p in (single_vertex(), miura(3, 3), bird_base()):
        r = layer_order(p)
        assert r.valid is True, f"{p.metadata['name']} should be orderable"
        assert r.order is not None
        assert sorted(r.order) == list(range(len(p.faces)))   # a permutation
        # Independently re-check: the returned order violates no constraint.
        assert order_violations(r.order, p) == [], p.metadata["name"]
        # ...and the independent physical arbiter sees no self-intersection.
        assert self_intersections_in_order(r.order, p) == [], p.metadata["name"]


@pytest.mark.parametrize("name", ["three-flap", "four-flap", "five-flap",
                                  "river-caterpillar"])
def test_multiflap_bases_order_without_self_intersection(name):
    """The uniaxial bases get a valid order that is clean by the INDEPENDENT check.

    five-flap (36 faces) and river-caterpillar (14 faces) are the two patterns
    the sign bug used to silently certify with a self-intersecting stack.  The
    arbiter here is ``self_intersections_in_order`` -- a different primitive than
    the solver's constraint builder -- so a pass means the paper genuinely does
    not pass through itself.
    """
    _, pattern = design_base(get_tree(name))
    r = layer_order(pattern)
    assert r.valid is True, f"{name} should be orderable"
    assert r.order is not None
    assert sorted(r.order) == list(range(len(pattern.faces)))
    assert self_intersections_in_order(r.order, pattern) == [], name


def test_independent_check_detects_a_deliberately_wrong_order():
    """The arbiter must be ABLE to fail: a hand-wrecked stack is caught.

    Take a genuine taco-tortilla constraint from the five-flap base and build an
    order that wedges the tortilla face strictly between its two taco faces.
    ``self_intersections_in_order`` must report it (proving the validator is not
    trivially empty), while the solver's own clean order passes.
    """
    _, pattern = design_base(get_tree("five-flap"))
    img, _, _ = flat_fold_image(pattern)
    _, _, tortilla, _ = _build_constraints(pattern, img)
    assert tortilla, "five-flap should have taco-tortilla constraints to exercise"
    a, b, t = tortilla[0]
    rest = [f for f in range(len(pattern.faces)) if f not in (a, b, t)]
    wrecked = rest + [a, t, b]          # t strictly between a and b
    hits = self_intersections_in_order(wrecked, pattern)
    assert any(f"face {t} wedged between {a} and {b}" in h for h in hits)
    # The solver's own order, by contrast, is clean.
    assert self_intersections_in_order(layer_order(pattern).order, pattern) == []


def test_bird_base_is_globally_flat_foldable():
    """The crane's bird base clears local *and* global checks."""
    assert globally_flat_foldable(bird_base()) is True


def test_brute_force_agrees_no_order_exists():
    """Exhaustively confirm the pinwheel truly has zero valid orders (n=4)."""
    import itertools
    from foldforge.geometry.layer_order import _build_constraints
    from foldforge.design.lang_molecule import flat_fold_image

    p = pinwheel(d=1.7)
    img, _, _ = flat_fold_image(p)
    _, _, tortilla, taco = _build_constraints(p, img)

    def valid(order):
        pos = {f: i for i, f in enumerate(order)}
        for a, b, t in tortilla:
            lo, hi = sorted((pos[a], pos[b]))
            if lo < pos[t] < hi:
                return False
        return True

    assert not any(valid(o) for o in itertools.permutations(range(len(p.faces))))
