# FoldForge — next steps

My working notes on where the project stands and what's left. FoldForge is a
differentiable computational origami engine. It's feature-complete and tested;
what remains is publishing, not code.

## State (v0.8.0)

- All seven milestones done (M0 geometry/theory → M6 web studio), plus the
  Origamizer capstone and the true-2D `miura2d` origamizer (below).
- Suite: **149 passed, 1 skipped** (`pytest -q`). The skip is `test_jax.py`
  when JAX isn't installed.
- Core is pure numpy + matplotlib. Optional backends: JAX (autodiff/GPU),
  OpenCV + SciPy (animal segmentation, solver speedup), PyTorch (monocular
  depth, `pip install foldforge[depth]`).

### Package map
```
foldforge/
  geometry/    M0  FOLD I/O, CreasePattern, Kawasaki/Maekawa, 2D render
  sim/         M1  rigid-panel relaxation solver (sparse fast path);
                   collision.py = self-intersection detect + repulsion;
                   sequencing.py = greedy collision-free fold ordering
  diff/        M2+ fold-chain kinematics, exact 2D Miura, implicit.py
                   (general-solver gradients via implicit function theorem;
                   analytic SPARSE Hessian, O(V) assembly)
  design/      M3  gradient-descent inverse design + angles_from_curve
  generative/  M4  from-scratch numpy MLP, simulator-in-the-loop
  materials/   M5  Miura auxetic Poisson ratio + inverse design
  origamize/   ★   surface.py (profile/heightfield), io.py (image/fn/points),
                   shapes.py (text/terrain/OBJ), vision.py (auto-rect segment +
                   inflate), close_relief (watertight two-sheet solid),
                   depth.py (MiDaS monocular depth → true 3D relief),
                   miura_fit.py (true-2D warped-Miura origamization),
                   symmetry.py (mirror-symmetrize a subject)
  fabricate/       SVG/DXF (cutters), OBJ/STL (folded 3D), glTF (.glb) with
                   per-face M/V vertex colours
  jaxsim/          optional JAX autodiff backend
  cli.py           fold (photo→mesh, one command) | check | render | gen |
                   origamize [--silhouette|--depth] [--engine] [--symmetry]
                   | export
studio/index.html  Three.js viewer (single self-contained static file):
                   6 shapes + baked animal samples, drop-your-own-image
                   folding, blank-paper default + Photo-texture toggle,
                   pipeline strip, showcase gallery, OBJ download, .fold
                   loader, in-browser PBD fold solver, shadows
examples/make_studio.py   regenerates studio/index.html
examples/audit_shapes.py  per-shape final-vs-target audit + contact sheet
examples/output/          zebra real-photo proofs (panels + STLs)
.github/workflows/ci.yml  pytest on 3.10–3.12 + wheel build
```

### The 2D origamizer (`origamize/miura_fit.py`)

A genuine 2D warped-Miura tessellation fit to a height field. This breaks the
1D bound of the corrugation engine (whose single honest crease pattern is an
extrusion — a ridge, not a dome). A joint Adam optimiser co-adapts the flat
pattern and the folded surface so the folded mid-surface tracks the target
while edges stay isometric (near-rigid) and facets stay planar.

- Fidelity metric `surface_fit_error` (normalised mid-surface RMSE after
  scale/offset align) runs on both engines via `compare_engines`. `miura2d`
  wins 8–200×: hemisphere 0.276→0.017, saddle 0.166→0.001, zebra 0.322→0.035,
  cat 0.343→0.029, dog 0.278→0.015, elephant 0.290→0.011.
- Real M+V crease pattern, FOLD/SVG export, mean fold strain ≈0.2–0.5% (max at
  curvature peaks). Not flat-foldable by design (curved relief) — reported, not
  hidden.
- Reachable as `origamize_miura()` or `engine="miura2d"` on
  `origamize_image/_silhouette/_depth/_function/_points`; `--engine miura2d` on
  the `fold` and `origamize` CLI commands (corrugation stays default). A
  `miura2d` `.fold` loads and folds live in the studio. ~1s per photo on CPU.
  Tests: `tests/test_miura2d_fit.py`.

### Symmetry (`origamize/symmetry.py`)

Mirror-symmetrize a subject before folding so its two halves match. `--symmetry
off|auto|x|y` on the CLI (default off), `symmetry=` kwarg on the origamize API,
and a control in the studio. `auto` picks the stronger axis; `x`/`y` force
top-bottom / left-right. Used for the butterfly showcase (raw vs symmetrized).

## Remaining — publish only

- **PyPI**: `python -m build` (wheel builds clean) + `twine upload` so
  `pip install foldforge` works.
- **Blog**: publish `docs/blog_post.md` wherever it fits.
- **Optional**: serve `studio/index.html` on GitHub Pages — it's a single
  static file with no server calls, so it works as-is.

Pushing to GitHub triggers CI (pytest 3.10–3.12 + wheel build).

## How to run / test

```
pip install -e .[dev,vision]      # add ,depth for MiDaS
pytest -q
```

`test_implicit.py` and `test_jax.py` are the slow ones. `test_jax.py` skips
unless JAX is installed.

## Honest scope notes (keep accurate; don't over-claim)

- Default engine = corrugation approximation (exact for extruded profiles;
  approximate with reported error for full height fields), not Origamizer
  tuck-folding.
- `miura2d` = genuine 2D warped-Miura fit; folded mid-surface tracks the target
  8–200× better on curves and it returns a real 2D crease pattern, but it's
  relief/tessellation origami (not arbitrary-3D tuck-folding), is NOT
  flat-foldable (curved relief), and trades a small reported fold strain
  (~0.2–0.5% mean, higher at curvature peaks) for the fidelity — not an exact
  rigid fold. It approximates height fields / photo reliefs, not closed
  surfaces or overhangs.
- Animal folding: silhouette mode = inflation; depth mode = MiDaS *estimated*
  relief (monocular prediction, not ground-truth geometry).
- M5 mechanics are kinematic, not a validated stress analysis.
- Self-intersection is detected + softly penalisable, not globally guaranteed;
  `fold_sequence` is greedy, not exhaustive.
- Differentiability: exact closed-form families (chain, Miura) + the general
  solver via implicit diff (analytic sparse Hessian) + optional JAX autodiff.
  All FD-verified.

## Ideas beyond scope (nothing planned)

Depth-Anything as a third depth backend; RL fold sequencing; a stricter
layer-ordering collision model that fires near the flat-fold limit.
