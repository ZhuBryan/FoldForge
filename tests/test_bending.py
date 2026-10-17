"""Tests for the sheet-metal bending-sequence planner.

The headline: a part (``coil``) whose naive bend-line-index order collides with
the simplified press-brake tool, while the planner finds a collision-free order.
Also checks the planner is deterministic, that reachability is correct on
hand-verified states, and that unconstrained parts (tray, u-channel) admit any
order.
"""

import numpy as np

from foldforge.fabricate import bending as B


# --- headline: planned order beats the naive one ---------------------------

def test_coil_planner_beats_naive():
    part = B.coil()
    naive = B.naive_plan(part)
    plan = B.plan_bend_order(part)

    # The naive inner-first order strands the outer bends (tool interference).
    assert naive.total_interference > 0
    assert not naive.feasible
    # The planner finds a fully collision-free order.
    assert plan.total_interference == 0
    assert plan.final_collisions == 0
    assert plan.feasible
    # headline inequality: planned collisions == 0 < naive collisions
    assert plan.total_interference == 0 < naive.total_interference
    # it is the classic outermost-first order
    assert plan.named_order() == ["L3", "L2", "L1"]
    # every bend reachable when it is made
    assert all(plan.reachable)


def test_planned_order_covers_all_bends():
    part = B.coil()
    plan = B.plan_bend_order(part)
    assert sorted(plan.order) == list(range(len(part.bends)))


def test_planner_is_deterministic():
    part = B.coil()
    a = B.plan_bend_order(part)
    b = B.plan_bend_order(part)
    assert a.order == b.order
    assert a.interference == b.interference


# --- reachability on hand-verified states -----------------------------------

def test_reachability_hand_verified_coil():
    part = B.coil()
    # On the flat blank, the outer bend L2 is reachable (nothing lifted it).
    assert B.bend_interference(part, 1, set()) == 0
    # After folding the inner bend L1, L2's line is lifted off the die.
    assert B.bend_interference(part, 1, {0}) > 0
    # The outermost bend L3 is always reachable from the flat blank.
    assert B.bend_interference(part, 2, set()) == 0


def test_reachability_hand_verified_overhang():
    # overhang never lifts a bend line; the only interference is punch clearance.
    part = B.overhang()  # bends: L1 = short wall at x=2, L2 = over-bent flap
    # Both bends are fine from the flat blank.
    assert B.bend_interference(part, 0, set()) == 0
    assert B.bend_interference(part, 1, set()) == 0
    # But forming the over-bent flap L2 first puts formed material directly
    # above L1's line, so the punch crashes when it comes down for L1.
    assert B.bend_interference(part, 0, {1}) > 0


def test_overhang_planner_avoids_the_crash():
    part = B.overhang()
    plan = B.plan_bend_order(part)
    # clean order exists and is found; final part is collision-free
    assert plan.feasible
    assert plan.final_collisions == 0
    # the reverse order (over-bent flap first) is the colliding one
    rev_inter, _, _ = B.evaluate_order(part, (1, 0))
    assert sum(rev_inter) > 0


# --- unconstrained parts: any order works -----------------------------------

def test_tray_any_order_reachable():
    part = B.tray()
    naive = B.naive_plan(part)
    plan = B.plan_bend_order(part)
    assert naive.feasible and naive.total_interference == 0
    assert plan.feasible


def test_u_channel_any_order_reachable():
    part = B.u_channel()
    naive = B.naive_plan(part)
    assert naive.feasible and naive.total_interference == 0


# --- the folded geometry is a real rigid fold -------------------------------

def test_fold_state_is_rigid_and_flat_start():
    part = B.coil()
    assert np.allclose(part.vertices[:, 2], 0.0)          # flat blank
    folded = part.fold_state(set(range(len(part.bends))))
    assert folded[:, 2].max() > 0.5                        # it actually folded up
