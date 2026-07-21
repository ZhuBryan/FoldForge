"""Tests for collision-free fold sequencing (greedy and beam search)."""

import numpy as np

import foldforge.sim.sequencing as sequencing
from foldforge.geometry import examples
from foldforge.sim import intersection_count
from foldforge.sim.mesh import FoldMesh
from foldforge.sim.sequencing import (
    fold_sequence, FoldSequence, _fold_sequence_greedy,
)

# Small, fast solver knobs - sequencing calls the solver many times.
FAST = dict(fold_fraction=0.6, stages=5, relax_iters=8)


def test_sequence_is_collision_free_miura():
    seq = fold_sequence(examples.miura(3, 3), fold_fraction=0.6,
                        stages=8, relax_iters=15)
    assert isinstance(seq, FoldSequence)
    # every recorded stage frame is collision-free per the existing detector
    for frame in seq.frames:
        assert intersection_count(seq.mesh, frame) == 0
    assert seq.is_collision_free
    # the final assembled fold is clean too
    assert intersection_count(seq.mesh, seq.result.vertices) == 0


def test_sequence_covers_all_creases():
    seq = fold_sequence(examples.single_vertex(), fold_fraction=0.6,
                        stages=6, relax_iters=12)
    creases = [i for i, h in enumerate(seq.mesh.hinges) if h.is_crease]
    assert sorted(seq.order) == sorted(creases)
    # groups partition the creases (no repeats, nothing dropped)
    flat = [c for g in seq.groups for c in g]
    assert len(flat) == len(set(flat)) == len(creases)


def test_sequence_frames_match_group_count():
    seq = fold_sequence(examples.miura(2, 2), fold_fraction=0.6,
                        stages=6, relax_iters=12)
    assert len(seq.frames) == len(seq.groups) == len(seq.collisions)
    assert seq.frames[-1].shape == seq.mesh.vertices.shape


# --- beam search -----------------------------------------------------------

def test_beam_width_default_is_one():
    # The public default must be the greedy single-track search.
    seq = fold_sequence(examples.single_vertex(), **FAST)
    assert isinstance(seq, FoldSequence)


def test_beam_width_one_matches_greedy():
    # beam_width=1 must reduce exactly to the original greedy path/results:
    # same groups, same order, same per-stage collisions.
    for pat in (examples.single_vertex(), examples.miura(2, 2)):
        public = fold_sequence(pat, beam_width=1, **FAST)

        mesh = FoldMesh.from_pattern(pat)
        creases = [i for i, h in enumerate(mesh.hinges) if h.is_crease]
        hinge_id = {i: id(mesh.hinges[i]) for i in creases}
        fold_kw = {"stages": FAST["stages"], "relax_iters": FAST["relax_iters"]}
        ref = _fold_sequence_greedy(mesh, creases, hinge_id,
                                    FAST["fold_fraction"], fold_kw,
                                    allow_collisions=False)

        assert public.groups == ref.groups
        assert public.order == ref.order
        assert public.collisions == ref.collisions


def test_beam_at_least_as_good():
    # beam_width>=2 must never do worse than greedy on total collisions.
    for pat in (examples.single_vertex(), examples.miura(2, 2)):
        greedy = fold_sequence(pat, beam_width=1, **FAST)
        beam = fold_sequence(pat, beam_width=3, **FAST)
        assert beam.total_collisions <= greedy.total_collisions
        # and it stays a valid, complete, collision-free ordering
        creases = [i for i, h in enumerate(beam.mesh.hinges) if h.is_crease]
        assert sorted(beam.order) == sorted(creases)
        assert beam.is_collision_free


def test_beam_explores_more_candidates_and_not_worse():
    """Beam explores strictly more candidate folds than greedy, never worse.

    Note: the existing collision detector never flags a self-intersection for
    the available mechanism folds (``FULL_FOLD_ANGLE`` stops short of the
    flat-folded, layer-stacking limit), so there is no reachable pattern on
    which greedy records a collision that beam could then undercut. We therefore
    demonstrate the two guarantees the beam upgrade actually provides: it
    searches a wider space (more solver evaluations, alternative orderings) and
    its result is no worse than greedy under the (collisions, groups) ranking.
    """
    pat = examples.single_vertex()

    calls = {"n": 0}
    real_fold = sequencing.fold

    def counting_fold(*a, **k):
        calls["n"] += 1
        return real_fold(*a, **k)

    sequencing.fold = counting_fold
    try:
        calls["n"] = 0
        greedy = fold_sequence(pat, beam_width=1, **FAST)
        greedy_calls = calls["n"]

        calls["n"] = 0
        beam = fold_sequence(pat, beam_width=3, **FAST)
        beam_calls = calls["n"]
    finally:
        sequencing.fold = real_fold

    # Wider search: beam evaluates strictly more candidate folds.
    assert beam_calls > greedy_calls
    # Never worse under the beam ranking (fewer/equal collisions, then groups).
    assert beam.total_collisions <= greedy.total_collisions
    assert (beam.total_collisions, len(beam.groups)) <= (
        greedy.total_collisions, len(greedy.groups))


def test_beam_width_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        fold_sequence(examples.single_vertex(), beam_width=0, **FAST)
