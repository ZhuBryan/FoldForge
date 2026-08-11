"""Photo -> stick-figure metric tree: the missing bridge to figurative design.

The origamize side already turns a photo into a *silhouette mask* of the subject
(GrabCut, see :func:`foldforge.origamize.silhouette_mask`). The design side
(:mod:`foldforge.design.treemaker`) already turns a *metric tree* (a stick
figure) into a uniaxial base by circle packing. What was missing is the piece in
between: read the subject's limbs off its silhouette and hand them to the
designer.

The idea is the medial axis. Thin the mask to a one-pixel skeleton; its shape is
a stick figure of the subject - a starfish's five arms become five spokes, a
fish becomes a spine with a tail fork. We turn that pixel skeleton into a graph
(endpoints = leaves, branch points = internal nodes, the chains between them =
edges whose length is the arc length of the limb), prune the hair of tiny spurs
that thinning always leaves, and emit a :class:`~foldforge.design.treemaker.MetricTree`.

Honest scope: this recovers the *topology and relative limb lengths* a subject
"reads" as. It does not know a starfish has exactly five arms - it counts what
the silhouette gives after pruning, and reports that count so a caller can see
when segmentation merged two arms or invented one. And a clean tree is only half
the battle: ``treemaker.design_base`` folds 3-, 4- and 5+-flap star trees flat
end to end via the universal (gusset) molecule when the packed flap circles form
a tight tangential hull; a loose hull only partially collapses - so
``design_from_image`` reports the flap count and honest fold validity rather than
pretending every subject yields a finished figure.

``scikit-image`` is used for thinning when present (a fast, well-tested medial
axis); it is a lazy import, so importing this module - or ``foldforge`` - does
not require it. Without it we fall back to a scipy morphological thinning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foldforge.design.treemaker import MetricTree, star_tree


# --- skeletonization --------------------------------------------------------

def _skeletonize(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin a binary mask to a 1-pixel skeleton; also return a distance map.

    Returns ``(skeleton, dist)`` where ``skeleton`` is a boolean array (the
    medial axis) and ``dist`` is the Euclidean distance from every pixel to the
    background - the local "thickness", used to size the spur-pruning threshold.

    Uses scikit-image's ``skeletonize`` when available (lazy import); otherwise
    falls back to a scipy-only morphological thinning so the bridge still runs.
    """
    from scipy import ndimage
    m = np.asarray(mask) > 0
    dist = ndimage.distance_transform_edt(m)
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(m)
    except Exception:
        skel = _thin_fallback(m)
    return np.asarray(skel, dtype=bool), np.asarray(dist, dtype=float)


def _thin_fallback(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton via scipy (no scikit-image needed).

    The classic Lantuejoul union of erosion-minus-opening residues. Coarser than
    a true medial-axis thinning (it can leave 2-px stubs) but it is real - the
    downstream graph builder and pruner cope - and it keeps the module importable
    when scikit-image is absent.
    """
    from scipy import ndimage
    img = np.asarray(mask) > 0
    skel = np.zeros_like(img, dtype=bool)
    el = ndimage.generate_binary_structure(2, 1)
    cur = img.copy()
    while cur.any():
        opened = ndimage.binary_opening(cur, el)
        skel |= cur & ~opened
        cur = ndimage.binary_erosion(cur, el)
    return skel


# --- pixel skeleton -> node/edge graph --------------------------------------

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-connected skeleton neighbours at each skeleton pixel."""
    from scipy import ndimage
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    return ndimage.convolve(skel.astype(int), k, mode="constant") * skel


@dataclass
class _Graph:
    """A reduced skeleton graph: node coordinates + weighted edges."""
    coords: dict           # node id -> (row, col) representative pixel
    adj: dict              # node id -> list of (nbr id, length)


def _build_graph(skel: np.ndarray) -> _Graph:
    """Contract a 1-px skeleton into a graph of endpoints and branch points.

    Node pixels are those whose 8-neighbour count is not 2 (endpoints = 1,
    junctions >= 3, isolated = 0). Adjacent node pixels are merged into one node
    (a thick junction is a small blob). Every degree-2 chain between two nodes
    becomes one edge whose length is its pixel arc length (1 per orthogonal step,
    sqrt(2) per diagonal).
    """
    from scipy import ndimage
    nc = _neighbor_count(skel)
    node_px = skel & (nc != 2)
    # merge touching node pixels into single nodes (label 8-connected blobs)
    lbl, n = ndimage.label(node_px, structure=np.ones((3, 3), int))
    coords: dict = {}
    if n:
        centers = ndimage.center_of_mass(node_px, lbl, range(1, n + 1))
        for i, c in enumerate(centers, start=1):
            coords[i] = (float(c[0]), float(c[1]))
    node_of = lbl                                   # 0 = not a node pixel

    H, W = skel.shape
    adj: dict = {i: [] for i in coords}
    visited_step: set = set()

    def walk(r0, c0, r1, c1):
        """Follow a deg-2 chain starting p0->p1 until it hits a node pixel."""
        pr, pc = r0, c0
        cr, cc = r1, c1
        length = np.hypot(cr - pr, cc - pc)
        while node_of[cr, cc] == 0:                 # still on a mid-chain pixel
            nxt = None
            for dr, dc in _NEIGHBORS:
                nr, nc_ = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc_ < W and skel[nr, nc_] \
                        and (nr, nc_) != (pr, pc):
                    nxt = (nr, nc_)
                    break
            if nxt is None:
                return None
            length += np.hypot(nxt[0] - cr, nxt[1] - cc)
            pr, pc, cr, cc = cr, cc, nxt[0], nxt[1]
        return node_of[cr, cc], length

    node_pixels = np.argwhere(node_px)
    for r, c in node_pixels:
        a = node_of[r, c]
        for dr, dc in _NEIGHBORS:
            nr, nc_ = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc_ < W and skel[nr, nc_]):
                continue
            if (r, c, nr, nc_) in visited_step:
                continue
            visited_step.add((r, c, nr, nc_))
            res = walk(r, c, nr, nc_)
            if res is None:
                continue
            b, length = res
            # mark the reverse first step so we don't re-walk from the far node
            visited_step.add((nr, nc_, r, c)) if False else None
            if b == a:
                continue                            # self loop, drop
            adj[a].append((b, float(length)))
    return _Graph(coords=coords, adj=adj)


def _spanning_tree(graph: _Graph) -> tuple[dict, int]:
    """Largest-component spanning tree of the graph (drops cycle edges).

    Returns ``(tree_adj, dropped)`` where ``tree_adj`` is ``node -> [(nbr, len)]``
    for the biggest connected component made acyclic by BFS, and ``dropped`` is
    how many non-tree (cycle) edges were discarded - reported honestly, since a
    silhouette with a hole (a ring) is not a tree.
    """
    # dedup undirected multiedges, keep the shortest between a pair
    best: dict = {}
    for a, nbrs in graph.adj.items():
        for b, L in nbrs:
            key = (min(a, b), max(a, b))
            if key not in best or L < best[key]:
                best[key] = L
    und: dict = {i: [] for i in graph.adj}
    for (a, b), L in best.items():
        und[a].append((b, L))
        und[b].append((a, L))

    # find connected components, pick the largest
    seen: set = set()
    comps: list = []
    for start in und:
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in und[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    if not comps:
        return {}, 0
    comp = max(comps, key=len)
    compset = set(comp)

    # BFS spanning tree over the chosen component
    tree_adj: dict = {u: [] for u in comp}
    used: set = set()
    visited: set = set()
    order = [comp[0]]
    visited.add(comp[0])
    dropped = 0
    while order:
        u = order.pop()
        for v, L in und[u]:
            if v not in compset:
                continue
            key = (min(u, v), max(u, v))
            if v not in visited:
                visited.add(v)
                used.add(key)
                tree_adj[u].append((v, L))
                tree_adj[v].append((u, L))
                order.append(v)
    # count edges within component not used by the tree
    for (a, b) in best:
        if a in compset and b in compset and (min(a, b), max(a, b)) not in used:
            dropped += 1
    return tree_adj, dropped


# --- prune to a clean metric tree -------------------------------------------

def _prune(tree_adj: dict, prune_frac: float, min_len: float) -> dict:
    """Remove tiny spur leaves and collapse degree-2 chains.

    A spur is a leaf edge shorter than ``max(prune_frac * longest_edge,
    min_len)`` - the hair thinning leaves at limb roots and along thick bodies.
    After removing spurs, degree-2 nodes (a bend in a single limb) are merged so
    each surviving edge spans a whole limb. Leaves are never pruned below 2 nodes
    total, so a blob still yields a (degenerate) tree instead of nothing.
    """
    adj = {u: list(v) for u, v in tree_adj.items()}

    def degree(u):
        return len(adj[u])

    def merge_deg2():
        changed = True
        while changed:
            changed = False
            for u in list(adj):
                if u not in adj:
                    continue
                if degree(u) == 2:
                    (a, la), (b, lb) = adj[u]
                    if a == b:
                        # both edges to same node: drop u, keep one edge? -> loop
                        adj[a] = [(x, L) for x, L in adj[a] if x != u]
                        del adj[u]
                        changed = True
                        continue
                    adj[a] = [(x, L) for x, L in adj[a] if x != u] + [(b, la + lb)]
                    adj[b] = [(x, L) for x, L in adj[b] if x != u] + [(a, la + lb)]
                    del adj[u]
                    changed = True

    merge_deg2()
    all_lens = [L for u in adj for _, L in adj[u]]
    if not all_lens:
        return adj
    thresh = max(prune_frac * max(all_lens), min_len)

    changed = True
    while changed:
        changed = False
        # count remaining leaves so we never dissolve below a minimal tree
        leaves = [u for u in adj if degree(u) == 1]
        if len(adj) <= 2:
            break
        for u in list(leaves):
            if u not in adj or degree(u) != 1:
                continue
            (nbr, L), = adj[u]
            if L < thresh and len(adj) > 2:
                adj[nbr] = [(x, w) for x, w in adj[nbr] if x != u]
                del adj[u]
                changed = True
        merge_deg2()
    return adj


def _pick_root(adj: dict) -> int:
    """Tree centre: the node whose farthest-leaf distance is smallest.

    Rooting at the centre makes each leaf's flap length (path to root) read as
    that limb's length from the body, which is what the packer wants.
    """
    if not adj:
        return 0
    nodes = list(adj)

    def eccentricity(s):
        # BFS farthest distance (weighted) from s
        best = 0.0
        stack = [(s, -1, 0.0)]
        while stack:
            u, p, d = stack.pop()
            best = max(best, d)
            for v, L in adj[u]:
                if v != p:
                    stack.append((v, u, d + L))
        return best

    return min(nodes, key=eccentricity)


@dataclass
class TreeExtraction:
    """A metric tree read off a silhouette, plus honesty metadata."""
    tree: MetricTree
    n_leaves: int
    edge_lengths: list
    dropped_cycle_edges: int
    degenerate: bool
    note: str
    skeleton: np.ndarray = field(default=None, repr=False)
    dist: np.ndarray = field(default=None, repr=False)
    node_coords: dict = field(default=None, repr=False)
    tree_adj: dict = field(default=None, repr=False)


def tree_from_mask(mask: np.ndarray, prune_frac: float = 0.35,
                   min_len_px: float = 6.0, smooth_frac: float = 0.012,
                   expected_leaves: int | None = None) -> TreeExtraction:
    """Extract a :class:`MetricTree` (stick figure) from a binary subject mask.

    ``prune_frac`` sets the spur threshold as a fraction of the longest edge;
    ``min_len_px`` is a floor in pixels. ``smooth_frac`` blurs the mask boundary
    (Gaussian sigma = ``smooth_frac * max(H, W)``, re-thresholded) before
    thinning: a raw GrabCut edge is ragged, and every wiggle sprouts a medial-axis
    spur, so a little smoothing turns dozens of hair branches into a handful of
    real limbs (set 0 to disable). Lengths in the returned tree are arc lengths
    normalised by the mask's larger dimension (so a full-frame limb is ~1.0).
    ``expected_leaves`` (if known, e.g. 5 for a starfish) is only used to fill the
    honesty ``note`` - it does not change the extraction.
    """
    m = np.asarray(mask) > 0
    from scipy import ndimage
    # keep the largest connected component so a stray fleck doesn't split things
    lbl, n = ndimage.label(m, structure=np.ones((3, 3), int))
    if n > 1:
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)

    if smooth_frac:                                 # de-fringe the ragged boundary
        sig = smooth_frac * max(m.shape)
        m = ndimage.binary_fill_holes(
            ndimage.gaussian_filter(m.astype(float), sig) > 0.5)
        if not m.any():                             # smoothing erased everything
            m = np.asarray(mask) > 0

    skel, dist = _skeletonize(m)
    graph = _build_graph(skel)
    tree_adj, dropped = _spanning_tree(graph)

    scale = float(max(m.shape))
    pruned = _prune(tree_adj, prune_frac=prune_frac, min_len=min_len_px)

    # relabel to 0..k-1 and build edge list with normalised lengths
    ids = sorted(pruned)
    remap = {old: i for i, old in enumerate(ids)}
    seen: set = set()
    edges = []
    for u in ids:
        for v, L in pruned[u]:
            key = (min(u, v), max(u, v))
            if key in seen or u == v:
                continue
            seen.add(key)
            edges.append((remap[u], remap[v], L / scale))

    degenerate = len(edges) < 2 or len([u for u in pruned if len(pruned[u]) == 1]) < 2
    if not edges:
        # totally degenerate (blob with no medial branches): a two-node stub
        edges = [(0, 1, max(dist.max(), 1.0) / scale)]
        pruned = {0: [(1, 0.0)], 1: [(0, 0.0)]}
        remap = {0: 0, 1: 1}
        degenerate = True

    root_old = _pick_root(pruned) if pruned else ids[0]
    root = remap.get(root_old, 0)
    tree = MetricTree(edges=edges, root=root, name="photo")
    leaves = tree.leaves()
    n_leaves = len(leaves)
    edge_lengths = [round(L, 4) for _, _, L in edges]

    if degenerate:
        note = (f"degenerate/near-blob silhouette: {n_leaves} leaf(s) - "
                "not enough limbs to design a base")
    elif n_leaves < 3:
        note = (f"only {n_leaves} leaves found; need >=3 to pack a base "
                "(subject too smooth or over-pruned)")
    else:
        note = f"{n_leaves} leaves (limbs) found"
    if expected_leaves is not None:
        if n_leaves == expected_leaves:
            note += f"; matches expected {expected_leaves}"
        else:
            note += f"; expected {expected_leaves} (off by {n_leaves - expected_leaves})"

    return TreeExtraction(
        tree=tree, n_leaves=n_leaves, edge_lengths=edge_lengths,
        dropped_cycle_edges=dropped, degenerate=degenerate, note=note,
        skeleton=skel, dist=dist, node_coords=graph.coords, tree_adj=pruned,
    )


def tree_from_image(path: str, prune_frac: float = 0.35,
                    smooth_frac: float = 0.012,
                    expected_leaves: int | None = None,
                    rect=None) -> TreeExtraction:
    """Photo -> subject mask (GrabCut) -> :class:`MetricTree`. See ``tree_from_mask``.

    ``rect=(x, y, w, h)`` (original pixel coords) overrides GrabCut's auto-seeded
    box; useful when the auto edge-density seed latches onto a busy background
    (e.g. a starfish on wood grain) instead of the subject.
    """
    from foldforge.origamize import silhouette_mask
    mask = silhouette_mask(path, rect=rect)
    ext = tree_from_mask(mask, prune_frac=prune_frac, smooth_frac=smooth_frac,
                         expected_leaves=expected_leaves)
    ext.mask = mask
    return ext


# --- end-to-end: photo -> tree -> uniaxial base -----------------------------

@dataclass
class PhotoDesign:
    """The whole bridge's output: the tree it read and the base it designed."""
    extraction: TreeExtraction
    packing: object = None
    pattern: object = None
    max_flap_error: float = float("nan")
    kawasaki_pass: int = 0
    maekawa_pass: int = 0
    n_interior: int = 0
    fold_flat: bool = False
    star_tree: object = None
    note: str = ""


def design_from_image(path: str, prune_frac: float = 0.35,
                      smooth_frac: float = 0.012,
                      expected_leaves: int | None = None, rect=None) -> PhotoDesign:
    """Full bridge: photo -> mask -> stick tree -> packed uniaxial base.

    Returns a :class:`PhotoDesign` carrying the extracted tree *and*, when the
    tree has >=3 leaves, the packed base and its honest fold-validity: max flap
    length error, and how many interior vertices pass Kawasaki/Maekawa. A base
    "folds flat" only when every interior vertex passes both - true for the 3/4
    flap tangential cases treemaker guarantees, and reported as False (partial)
    otherwise, never faked.
    """
    ext = tree_from_image(path, prune_frac=prune_frac, smooth_frac=smooth_frac,
                          expected_leaves=expected_leaves, rect=rect)
    out = PhotoDesign(extraction=ext)
    if ext.n_leaves < 3:
        out.note = ("tree has <3 leaves; treemaker needs >=3 to triangulate a "
                    "base. Extraction succeeded, base skipped.")
        return out
    from foldforge.design.treemaker import design_base, flap_length_errors
    from foldforge.geometry.foldability import foldability_report
    # Collapse the raw stick tree (which may carry internal spine edges - a
    # river-like tree treemaker cannot fold flat) into the single-hub STAR tree
    # of the same limbs: each flap length is the leaf's distance to the body
    # (its path to the root). This is the classic TreeMaker reduction and is what
    # design_base folds flat for 3/4 flaps. The raw tree is kept in ``extraction``
    # for honest topology reporting.
    flap_lengths = [ext.tree.flap_length(lf) for lf in ext.tree.leaves()]
    star = star_tree(flap_lengths, name="photo-star")
    out.star_tree = star
    try:
        packing, pattern = design_base(star)
    except Exception as exc:
        out.note = f"packing failed: {exc}"
        return out
    out.packing = packing
    out.pattern = pattern
    err = flap_length_errors(packing, pattern)
    out.max_flap_error = float(err.max()) if len(err) else float("nan")
    rep = foldability_report(pattern)
    out.n_interior = len(rep.vertices)
    out.kawasaki_pass = sum(1 for v in rep.vertices if v.kawasaki)
    out.maekawa_pass = sum(1 for v in rep.vertices if v.maekawa)
    out.fold_flat = (out.n_interior > 0
                     and out.kawasaki_pass == out.n_interior
                     and out.maekawa_pass == out.n_interior)
    if out.fold_flat:
        out.note = (f"{ext.n_leaves}-flap base folds flat end to end "
                    f"(max flap error {out.max_flap_error * 100:.1f}%)")
    else:
        out.note = (f"{ext.n_leaves}-flap base is PARTIAL: "
                    f"{out.kawasaki_pass}/{out.n_interior} Kawasaki, "
                    f"{out.maekawa_pass}/{out.n_interior} Maekawa pass. The "
                    "universal (gusset) molecule folds a tight tangential hull "
                    "flat; when the packed flap circles don't form one (arm "
                    "lengths give a loose, non-tangential hull) some interior "
                    "vertices don't collapse, so this packs but is not fully flat.")
    return out
