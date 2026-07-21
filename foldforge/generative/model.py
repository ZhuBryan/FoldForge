"""Generative model with the simulator in the loop (Milestone 4).

Instead of optimising one fold at a time (M3), we train a small network to
*propose* a fold for any target in one shot - amortised inverse design. The
differentiable simulator is literally in the training loop: we fold the
network's proposal and backpropagate the shape error through the analytic fold
gradient and then through the network. A noise input lets it propose several
distinct folds for the same target.

Everything here is from-scratch numpy (forward + manual backprop + Adam); no
deep-learning framework. Small by design, but it is a real generative model
trained against a physics verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldforge.diff.kinematics import fold_chain  # noqa: F401 (handy for callers)


def batch_fold(angles: np.ndarray, seg: float = 1.0):
    """Fold a batch of angle vectors at once. Returns (spine (B,n+1,2), headings)."""
    headings = np.cumsum(angles, axis=1)
    steps = seg * np.stack([np.cos(headings), np.sin(headings)], axis=-1)
    B, n, _ = steps.shape
    spine = np.zeros((B, n + 1, 2))
    spine[:, 1:] = np.cumsum(steps, axis=1)
    return spine, headings


def batch_recon_grad(angles: np.ndarray, target: np.ndarray, seg: float = 1.0):
    """Batch-mean reconstruction loss and the *per-sample* gradients.

    Returns ``(mean_loss, grad)`` where ``mean_loss`` is the loss averaged over
    the batch, and ``grad[i]`` is the gradient of sample ``i``'s own loss (its
    mean squared point residual) with respect to that sample's angles. The two
    are deliberately on different conventions: ``grad[i]`` is per-sample so the
    MLP backprop can sum them (descending the total batch loss), while
    ``mean_loss`` is reported per sample for readability. ``grad`` is verified
    against an independent finite-difference of the per-sample loss in the tests.

    Vectorised closed form: with residuals r_k and segment heading-derivatives
    w_m the gradient telescopes into two reverse cumulative sums, so a whole
    batch costs a few numpy calls.
    """
    B, n = angles.shape
    spine, headings = batch_fold(angles, seg)
    resid = spine - target
    mean_loss = np.mean(np.sum(resid ** 2, axis=2), axis=1).mean()
    revcs = np.cumsum(resid[:, ::-1, :], axis=1)[:, ::-1, :]   # sum_{k>=i} r_k
    R = revcs[:, 1:, :]                                        # R_m, m=0..n-1
    w = seg * np.stack([-np.sin(headings), np.cos(headings)], axis=-1)
    g = np.sum(w * R, axis=2)                                  # (B, n)
    grad = (2.0 / (n + 1)) * np.cumsum(g[:, ::-1], axis=1)[:, ::-1]
    return mean_loss, grad


def smooth_angles(rng: np.random.Generator, n: int, scale: float = 0.35) -> np.ndarray:
    """A smooth low-frequency fold profile -> arch/wave-like reachable targets."""
    k = rng.integers(1, 4)
    t = np.linspace(0, np.pi, n)
    a = sum(rng.uniform(-1, 1) * np.sin((i + 1) * t + rng.uniform(0, np.pi))
            for i in range(k))
    return scale * a / (np.abs(a).max() + 1e-9)


def make_dataset(m: int, n: int, seg: float, rng: np.random.Generator) -> np.ndarray:
    """``m`` reachable target spines (flattened), from random smooth folds."""
    angles = np.array([smooth_angles(rng, n) for _ in range(m)])
    spine, _ = batch_fold(angles, seg)
    return spine.reshape(m, -1)


@dataclass
class FoldGenerator:
    """An MLP that maps a target spine (+ optional noise) to fold angles."""

    n: int                       # number of fold angles to output
    seg: float = 1.0
    hidden: int = 96
    noise_dim: int = 4

    def __post_init__(self):
        rng = np.random.default_rng(0)
        din = 2 * (self.n + 1) + self.noise_dim
        he = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2 / a)
        self.W1, self.b1 = he(din, self.hidden), np.zeros(self.hidden)
        self.W2, self.b2 = he(self.hidden, self.hidden), np.zeros(self.hidden)
        self.W3, self.b3 = he(self.hidden, self.n), np.zeros(self.n)
        self._params = ["W1", "b1", "W2", "b2", "W3", "b3"]

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1; h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2; h2 = np.tanh(z2)
        y = h2 @ self.W3 + self.b3
        return y, (X, h1, h2)

    def generate(self, target_spine: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        """Propose fold angles for a target spine (shape ``(n+1, 2)`` or flat)."""
        flat = np.asarray(target_spine).reshape(-1)
        if noise is None:
            noise = np.zeros(self.noise_dim)
        X = np.concatenate([flat, noise])[None, :]
        return self._forward(X)[0][0]

    def train(self, targets: np.ndarray, *, epochs: int = 80, batch: int = 64,
              lr: float = 2e-3, seed: int = 0) -> list[float]:
        """Train against the simulator-in-the-loop reconstruction loss."""
        rng = np.random.default_rng(seed)
        mom = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        vel = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        history = []
        for _ in range(epochs):
            idx = rng.permutation(len(targets))
            epoch_loss, seen = 0.0, 0
            for s in range(0, len(targets), batch):
                bi = idx[s:s + batch]
                tgt = targets[bi]
                noise = rng.standard_normal((len(bi), self.noise_dim))
                X = np.concatenate([tgt, noise], axis=1)
                y, (X_, h1, h2) = self._forward(X)
                loss, dy = batch_recon_grad(y, tgt.reshape(len(bi), -1, 2), self.seg)
                epoch_loss += loss * len(bi); seen += len(bi)
                grads = {}
                grads["W3"] = h2.T @ dy; grads["b3"] = dy.sum(0)
                dh2 = dy @ self.W3.T; dz2 = dh2 * (1 - h2 ** 2)
                grads["W2"] = h1.T @ dz2; grads["b2"] = dz2.sum(0)
                dh1 = dz2 @ self.W2.T; dz1 = dh1 * (1 - h1 ** 2)
                grads["W1"] = X_.T @ dz1; grads["b1"] = dz1.sum(0)
                t += 1
                for p in self._params:
                    mom[p] = b1 * mom[p] + (1 - b1) * grads[p]
                    vel[p] = b2 * vel[p] + (1 - b2) * grads[p] ** 2
                    upd = lr * (mom[p] / (1 - b1 ** t)) / (np.sqrt(vel[p] / (1 - b2 ** t)) + eps)
                    setattr(self, p, getattr(self, p) - upd)
            history.append(epoch_loss / seen)        # true epoch-mean loss
        return history
