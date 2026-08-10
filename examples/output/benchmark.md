# FoldForge engine benchmark

Normalised mid-surface fit error (lower is better) of the two origamize engines on a fixed target set, via `compare_engines` (`surface_fit_error`, the same metric both engines are scored with). `ratio` = corrugation / miura2d. Regenerate with `python examples/benchmark.py`.

| target | corrugation (1D) | miura2d (2D) | ratio | miura2d strain (mean / max) |
|---|---|---|---|---|
| hemisphere | 0.276 | **0.017** | 16.3x | 0.23% / 1.8% |
| saddle | 0.166 | **0.001** | 217.6x | 0.15% / 0.9% |
| radial ripple | 0.202 | **0.023** | 8.8x | 0.43% / 5.4% |
| butterfly | 0.344 | **0.024** | 14.2x | 0.42% / 4.2% |
| starfish | 0.329 | **0.026** | 12.5x | 0.33% / 2.7% |
| fish | 0.312 | **0.023** | 13.7x | 0.43% / 4.8% |

Synthetic rows (hemisphere / saddle / radial ripple) reproduce the README's engine-comparison table exactly. Photo rows use the no-torch silhouette-inflation relief (cv2 distance transform), downsampled to a small grid - honest, but distinct from the README's depth-based photo rows.
