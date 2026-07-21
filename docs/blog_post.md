# FoldForge: an integrated, from-scratch differentiable origami engine

I spent the last few months building FoldForge, a computational origami engine
that goes from a flat sheet's crease pattern all the way to a folded 3D shape,
and — the part I care about most — runs that whole path *differentiably*, so you
can ask it the inverse question: given a shape I want, what crease pattern folds
into it? This is a writeup of why the problem is interesting, what I think is
unusual about the engine, and three pieces of it I'm happy with, numbers
included. It is also honest about what it does not do.

## Why fold anything

Origami stopped being a paper craft and became an engineering tool once people
noticed that folding is a way to turn a flat sheet into a three-dimensional
structure with properties the flat sheet never had. A folded pattern can be
*deployable*: it packs flat for transport and springs open on site, which is why
the same Miura-ori fold shows up in proposed solar arrays, packable antennas, and
self-expanding stents. Folding is also a way to build *mechanical metamaterials*
— the geometry of the creases, not the material, sets the behavior. A Miura sheet
is auxetic: squeeze it in one direction and it contracts in the other, a negative
Poisson's ratio you get purely from the fold. Change the crease pattern and you
change the stiffness, the expansion, the whole response. That is a rare thing in
engineering: a structure whose properties are programmed by a pattern you can
draw on a flat sheet.

The catch is that designing those patterns is hard. Forward folding is already a
constrained mechanism problem, and inverse design — pattern *from* target — is
worse. That is the gap FoldForge tries to close.

## What makes the engine unusual

The whole thing is one NumPy codebase. It reads and writes the FOLD format (the
JSON standard for crease patterns), validates flat-foldability with Kawasaki's
and Maekawa's theorems, folds a pattern into 3D with a rigid-panel dynamic
relaxation simulator, differentiates through that fold three different ways, does
gradient-descent inverse design against a target, trains a small from-scratch MLP
that proposes folds with the simulator in the loop, computes metamaterial
mechanics, exports to laser-cutter vectors and to 3D-printable meshes, and ships
a single-file Three.js studio you can open in a browser. No deep-learning
framework in the core — the kinematics, the backprop in the generative model, the
implicit-function-theorem gradients, and the optimizers are all hand-written
NumPy, with every gradient checked against finite differences. There is an
optional JAX backend that reproduces the hand-derived gradients automatically and
matches them to about 1e-9, which is a nice cross-check on the hand math, but you
never need it. Eighty-eight tests cover the lot.

I wanted the design because it makes the seams legible. When the inverse designer
converges, I can point at exactly which analytic Jacobian carried the gradient.
There is nothing I cannot open up.

## Three pieces I'm happy with

**Differentiating the general solver.** The easy families — a single fold chain,
the 2D Miura tessellation — have closed-form fold maps I can differentiate by
hand. The hard case is an arbitrary crease graph, where the folded shape is the
minimizer of an energy: stiff bars keep the panels rigid while each hinge is
pulled toward its target fold angle. There is no closed form for where that
settles. The wrong way to get gradients is to backpropagate through the solver's
iterations, which is enormous in memory. The right way is the implicit function
theorem: at the equilibrium the energy gradient is zero, and differentiating that
condition gives a single linear system in the Hessian. One linear solve at the
fixed point, no unrolling.

The Hessian is the expensive part if you build it naively, so I assemble it
analytically and locally. Every bar and every hinge touches at most four
vertices, so each contributes a small dense block and the assembled matrix is
sparse — `O(V)` to build instead of the `O(V^2)` you pay to finite-difference the
global gradient. It matches the old dense numerical Hessian to about 1e-8, and it
runs 7 to 14 times faster in the 48-to-108 degrees-of-freedom range I tested, a
gap that widens as the pattern grows. If SciPy is around it comes back as a sparse
matrix; if not, the identical blocks assemble into a dense array.

**From a photo to a crease pattern.** The pipeline I had the most fun with folds a
real photograph. Take a zebra picture. First I segment the animal from the
background with GrabCut, because otherwise the background folds too. Then I need a
height field. The simple option inflates the silhouette into a rounded bas-relief
— it works, but the subject comes out a smooth blob. The better option estimates
genuine relief with a monocular depth network: MiDaS small by default (it needs
only PyTorch), or DPT_Hybrid if you install `timm`, masked to the subject and
handed to the origamizer. The origamizer itself is a corrugation solver: it models
the sheet as parallel rigid fold-chains and solves each chain's fold angles, with
the differentiable core, to match the target's cross-section. The output is a real
FOLD file plus an exact folded state.

```python
from foldforge.origamize import origamize_depth
result, relief = origamize_depth("zebra.jpg")   # segment, estimate depth, fold
result.pattern  # a real crease pattern; result.folded  # the folded 3D vertices
```

The numbers are honest about the tradeoff. Silhouette inflation (a rounded
spherical-cap balloon) folds the zebra to error 0.148; MiDaS depth folds to
0.239 — higher, because real relief carries
more high-frequency detail than a single corrugation can reproduce, but it
follows the animal's actual near-and-far structure instead of ballooning it. DPT
resolves depth edges about 19% sharper than MiDaS small at a similar fold error.
The command line does the same thing:

```bash
foldforge origamize zebra.jpg zebra.fold --depth --depth-model DPT_Hybrid
```

**Folding in the browser.** The studio is a single HTML file that loads Three.js
from a CDN — no build, no server. Beyond the six baked-in shapes, you can drop in
your own `.fold` file and it folds it live with a small position-based-dynamics
rigid-origami solver written in JavaScript. That solver has to reproduce, in the
browser, the rigidity the Python simulator guarantees. On a loaded Miura it folds
at 0.17% strain and recovers 22 of the 24 crease mountain/valley signs, with a
fold-extent ratio of 1.03 against the reference — close enough that the fold looks
and behaves right while you drag the slider. You can also drop in a photo and
watch it fold as blank paper (with the photo kept in a reference panel and a
Photo-texture toggle to wrap the fold in it), and download the current frame as
an OBJ.

## What it does not do

I would rather state the limits than have a reviewer find them. The origamizer is
a *corrugation* approximation — the simplest member of the surface-origami family,
not full Origamizer tuck-folding. A single extruded profile is exactly
developable; a full 2D height field is approximated as independent pleated strips,
and I report the approximation error rather than hide it. The photo modes estimate
shape rather than measure it: silhouette mode inflates a mask, and depth mode
folds a monocular *prediction* of relief, not ground-truth geometry. The
metamaterial mechanics are kinematic — they reproduce the right trends, like the
Miura's negative Poisson's ratio, but they are not a validated stress analysis.
Self-intersection is detected and can be softly penalized, and there is a fold
sequencer that searches for a collision-free order to close the creases (greedy,
or beam search that never does worse than greedy), but the search is heuristic:
neither guarantees a globally collision-free fold. And the flat-foldability
theorems are necessary, not sufficient — passing them is a strong signal, not a
proof.

## What's next

The engine is feature-complete for what I set out to do, so the next steps are
about reach rather than new subsystems. I would like the fold sequencer to use
something stronger than beam search — a learned or reinforcement-learning policy
that plans the order the way a person does. I would like the browser solver to
fold arbitrary loaded patterns, not just the image-grid ones the closed-form path
handles today. And the corrugation origamizer is the obvious place to push toward
real tuck-folding, which would trade the "independent strips" approximation for a
genuinely two-dimensional foldable surface. None of these change the shape of the
codebase; they deepen it. That was the point of building it one honest layer at a
time.
