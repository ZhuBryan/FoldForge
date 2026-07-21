"""Fold sequencing: search a collision-free order to fold a pattern's creases.

Folding every crease at once can drive panels straight through each other. Real
folding is *ordered* - you make some creases first, which moves paper out of the
way before the next ones close. This module searches for such an order, leaning
on the existing collision detector to score trial folds.

Two search strategies share the same scoring machinery:

  * greedy (``beam_width=1``) - the original one-track search. Each round scores
    every not-yet-folded crease added on its own to the active set, folds to
    ``fold_fraction`` with the solver, and scores by
    :func:`~foldforge.sim.collision.intersection_count` (ties broken by the soft
    separation penalty). Every crease that stays at the round minimum is
    committed together as one "group", recovering the natural stages. A
    one-level backtrack undoes the previous group and retries it a crease at a
    time before conceding.

  * beam (``beam_width>=2``) - keeps the top ``beam_width`` partial sequences
    alive at once instead of committing to a single track. Each partial state is
    expanded both by the greedy group *and* by smaller alternative commits, so
    the search can defer a crease that greedy would have locked in early. States
    are scored by cumulative collision count (tie-break: fewer groups, then
    lower fold energy), duplicate states (same folded-crease set) are pruned, and
    only the best ``beam_width`` survive each round. The greedy track is always
    among the seeds, so beam search never returns a worse ordering than greedy.

``fold_sequence`` returns a :class:`FoldSequence`: the ordered crease groups, the
folded frame after each group (the trajectory), and the final
:class:`~foldforge.sim.solver.FoldResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foldforge.geometry.crease_graph import CreasePattern
from foldforge.sim.mesh import FoldMesh
from foldforge.sim.solver import fold, FoldResult
from foldforge.sim.collision import intersection_count, separation_penalty


@dataclass
class FoldSequence:
    """The result of :func:`fold_sequence`.

    Attributes:
        mesh:       the :class:`FoldMesh` that was folded.
        groups:     ordered list of crease groups; each group is a list of hinge
                    indices (into ``mesh.hinges``) folded together at that stage.
        frames:     folded vertices ``(V, 3)`` after each cumulative group - the
                    fold trajectory (``frames[-1]`` is the final shape).
        collisions: self-intersection count after each group (all 0 on success).
        result:     the final :class:`FoldResult` (full active set).
    """

    mesh: FoldMesh
    groups: list[list[int]]
    frames: list[np.ndarray]
    collisions: list[int]
    result: FoldResult

    @property
    def order(self) -> list[int]:
        """Flat crease-fold order (hinge indices), groups concatenated."""
        return [h for g in self.groups for h in g]

    @property
    def is_collision_free(self) -> bool:
        return all(c == 0 for c in self.collisions)

    @property
    def total_collisions(self) -> int:
        """Cumulative self-intersection count summed over every stage."""
        return int(sum(self.collisions))


def _score(mesh, active_ids, fold_fraction, fold_kw):
    """Fold the active crease set and score its collisions (lower is better)."""
    r = fold(mesh, fold_fraction,
             actuate=lambda h: id(h) in active_ids, **fold_kw)
    hits = intersection_count(mesh, r.vertices)
    # Tie-break by how hard the soft repulsion is pushing (smoother than the
    # integer count), so we prefer the least-penetrating order among ties.
    pen = float(np.linalg.norm(separation_penalty(mesh, r.vertices)))
    return hits, pen, r


# --- greedy search (beam_width == 1) ---------------------------------------

def _fold_sequence_greedy(mesh, creases, hinge_id, fold_fraction, fold_kw,
                          allow_collisions) -> FoldSequence:
    """Original greedy pass with a one-level backtrack (see module docstring)."""
    groups: list[list[int]] = []
    frames: list[np.ndarray] = []
    collisions: list[int] = []
    active: list[int] = []
    active_ids: set[int] = set()
    remaining = list(creases)
    tried_first: set[int] = set()   # creases already ruled out as a fresh group

    while remaining:
        # Score every remaining crease added on its own to the active set.
        scored = []
        for c in remaining:
            hits, pen, r = _score(mesh, active_ids | {hinge_id[c]},
                                  fold_fraction, fold_kw)
            scored.append((hits, pen, c, r))
        scored.sort(key=lambda s: (s[0], s[1]))
        best_hits = scored[0][0]

        if best_hits > 0 and not allow_collisions:
            # Dead end: undo the last committed group and re-fold its creases one
            # at a time (a light backtrack) before conceding.
            if groups and any(c not in tried_first for c in groups[-1]):
                undo = groups.pop()
                frames.pop(); collisions.pop()
                for c in undo:
                    active.remove(c); active_ids.discard(hinge_id[c])
                remaining = undo + remaining
                tried_first.update(undo)
                continue
            raise RuntimeError(
                "no collision-free fold order found "
                f"(best stage still has {best_hits} intersections); "
                "pass allow_collisions=True to return the best effort")

        # Commit every remaining crease that stays at the round-minimum score.
        group = [c for hits, pen, c, r in scored if hits == best_hits]
        # Fold the whole committed group together for the recorded frame.
        for c in group:
            active.append(c); active_ids.add(hinge_id[c])
            remaining.remove(c)
        hits, pen, r = _score(mesh, active_ids, fold_fraction, fold_kw)
        groups.append(group)
        frames.append(r.vertices.copy())
        collisions.append(hits)

    result = fold(mesh, fold_fraction,
                  actuate=lambda h: id(h) in active_ids, **fold_kw)
    if not frames:                                  # pattern had no creases
        frames.append(result.vertices.copy())
        collisions.append(intersection_count(mesh, result.vertices))
    return FoldSequence(mesh=mesh, groups=groups, frames=frames,
                        collisions=collisions, result=result)


# --- beam search (beam_width >= 2) -----------------------------------------

@dataclass
class _State:
    """One partial fold ordering carried in the beam."""
    active: tuple[int, ...]                 # committed creases, in commit order
    groups: list[list[int]] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)
    collisions: list[int] = field(default_factory=list)
    cum: int = 0                            # cumulative collision count
    energy: float = 0.0                     # last fold's max strain (tie-break)

    @property
    def key(self):
        # Two states with the same folded-crease *set* are duplicates - the
        # collision detector only sees the folded geometry, not the path.
        return frozenset(self.active)

    def rank(self):
        # Lower is better: fewest collisions, then fewest groups, then energy.
        return (self.cum, len(self.groups), self.energy)


def _candidate_groups(mesh, active_ids, remaining, hinge_id, fold_fraction,
                      fold_kw, branch):
    """Groups to try committing next from a state, with the greedy group first.

    Scores each remaining crease added on its own (exactly as greedy does), then
    offers: the greedy group (all round-minimum creases at once) and singleton
    commits of the best ``branch`` individual creases. The singletons are what
    let beam search defer a crease greedy would lock in immediately.
    """
    scored = []
    for c in remaining:
        hits, pen, _ = _score(mesh, active_ids | {hinge_id[c]},
                              fold_fraction, fold_kw)
        scored.append((hits, pen, c))
    scored.sort(key=lambda s: (s[0], s[1]))
    best_hits = scored[0][0]

    groups: list[list[int]] = []
    greedy_group = [c for hits, pen, c in scored if hits == best_hits]
    groups.append(greedy_group)
    # Singleton alternatives from the top of the ranking (skip if it would just
    # duplicate a size-1 greedy group).
    for hits, pen, c in scored[:max(1, branch)]:
        if [c] not in groups:
            groups.append([c])
    return groups, best_hits


def _fold_sequence_beam(mesh, creases, hinge_id, fold_fraction, fold_kw,
                        allow_collisions, beam_width) -> FoldSequence:
    """Beam search over fold orderings (see module docstring)."""
    branch = beam_width               # singleton alternatives explored per state
    beam: list[_State] = [_State(active=())]
    complete: list[_State] = []

    n_creases = len(creases)
    while beam:
        successors: list[_State] = []
        for st in beam:
            active_set = set(st.active)
            remaining = [c for c in creases if c not in active_set]
            active_ids = {hinge_id[c] for c in st.active}
            cand_groups, _ = _candidate_groups(
                mesh, active_ids, remaining, hinge_id, fold_fraction, fold_kw,
                branch)
            for grp in cand_groups:
                new_active = st.active + tuple(grp)
                new_ids = active_ids | {hinge_id[c] for c in grp}
                hits, pen, r = _score(mesh, new_ids, fold_fraction, fold_kw)
                if hits > 0 and not allow_collisions:
                    # A committed stage must be clean; drop paths that collide.
                    continue
                child = _State(
                    active=new_active,
                    groups=st.groups + [grp],
                    frames=st.frames + [r.vertices.copy()],
                    collisions=st.collisions + [hits],
                    cum=st.cum + hits,
                    energy=r.max_strain,
                )
                if len(new_active) == n_creases:
                    # A finished ordering: harvest it immediately so a scarce
                    # beam slot can never prune away a complete (possibly best)
                    # solution before it is recorded.
                    complete.append(child)
                else:
                    successors.append(child)

        if not successors:
            break

        # Prune duplicate states (same folded-crease set), keeping the best.
        dedup: dict = {}
        for s in successors:
            k = s.key
            if k not in dedup or s.rank() < dedup[k].rank():
                dedup[k] = s
        ranked = sorted(dedup.values(), key=_State.rank)
        beam = ranked[:beam_width]

    if not complete:
        raise RuntimeError(
            "no collision-free fold order found; "
            "pass allow_collisions=True to return the best effort")

    best = min(complete, key=_State.rank)

    result = fold(mesh, fold_fraction,
                  actuate=lambda h: id(h) in {hinge_id[c] for c in best.active},
                  **fold_kw)
    frames = list(best.frames)
    collisions = list(best.collisions)
    if not frames:                                  # pattern had no creases
        frames.append(result.vertices.copy())
        collisions.append(intersection_count(mesh, result.vertices))
    return FoldSequence(mesh=mesh, groups=list(best.groups), frames=frames,
                        collisions=collisions, result=result)


def fold_sequence(pattern: CreasePattern, *, fold_fraction: float = 0.7,
                  stages: int = 8, relax_iters: int = 15,
                  allow_collisions: bool = False, beam_width: int = 1,
                  **fold_kw) -> FoldSequence:
    """Search a collision-free order to fold ``pattern``'s creases.

    ``beam_width=1`` (the default) runs the original greedy search with a
    one-level backtrack; ``beam_width>=2`` runs a beam search that keeps the top
    ``beam_width`` partial orderings alive and can improve on greedy (and never
    does worse - the greedy track is always explored). See the module docstring.

    ``stages`` / ``relax_iters`` and any extra ``fold_kw`` are passed straight to
    :func:`~foldforge.sim.solver.fold` for each trial fold. If no zero-collision
    order is found and ``allow_collisions`` is False a ``RuntimeError`` is raised;
    set it True to return the best-effort ordering instead.

    Returns a :class:`FoldSequence`.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    mesh = FoldMesh.from_pattern(pattern) if isinstance(pattern, CreasePattern) \
        else pattern
    fold_kw = {"stages": stages, "relax_iters": relax_iters, **fold_kw}

    creases = [i for i, h in enumerate(mesh.hinges) if h.is_crease]
    hinge_id = {i: id(mesh.hinges[i]) for i in creases}

    if beam_width == 1:
        return _fold_sequence_greedy(
            mesh, creases, hinge_id, fold_fraction, fold_kw, allow_collisions)
    return _fold_sequence_beam(
        mesh, creases, hinge_id, fold_fraction, fold_kw, allow_collisions,
        beam_width)
