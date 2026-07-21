"""Studio shape-data sanity: the six baked shapes must actually fold to their
targets. Guards two regressions we fixed - the saddle folding to half its
target z-span, and the miura barely folding. Pure-numpy (no OpenCV / torch)."""

import os
import sys

import numpy as np
import pytest

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
sys.path.insert(0, _EX)
make_studio = pytest.importorskip("make_studio")


def _final(shape):
    return np.array(shape["frames"][-1])


def test_all_targeted_shapes_reach_their_target_zspan():
    data = make_studio.build_data()
    for name in ("dome", "image", "peak", "saddle", "ridge"):
        S = data["shapes"][name]
        fz = float(np.ptp(_final(S)[:, 2]))
        tz = float(np.ptp(np.array(S["target"]).reshape(-1, 3)[:, 2]))
        assert fz == pytest.approx(tz, rel=0.08), f"{name}: final z {fz:.2f} vs target {tz:.2f}"


def test_saddle_folds_to_full_depth_not_half():
    S = make_studio.build_data()["shapes"]["saddle"]
    fz = float(np.ptp(_final(S)[:, 2]))
    assert fz > 7.0, f"saddle only reached z-span {fz:.2f} (regression: half-fold)"


def test_miura_is_clearly_folded():
    S = make_studio.build_data()["shapes"]["miura"]
    V = _final(S)
    span = np.ptp(V, axis=0)
    assert span[2] > 2.0, f"miura relief too shallow: z-span {span[2]:.2f}"
    assert span[2] / max(span[0], span[1]) > 0.12   # reads as folded, not near-flat
