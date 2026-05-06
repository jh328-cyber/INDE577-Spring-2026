# 08 — DBSCAN vs. K-Means on Non-Convex Clusters

**Notebook:** [`dbscan_moons.ipynb`](dbscan_moons.ipynb)

Side-by-side demonstration on the canonical two-moons dataset showing the *one
specific structural failure* of k-means (it cannot recover non-convex clusters
because its decision boundaries are straight) and how DBSCAN fixes it.

## Why this dataset?

Synthetic two-moons is the cleanest possible illustration of the convex/non-convex
distinction, and it removes the noise of real-world feature engineering so the
algorithmic difference is what you see. Real datasets where this matters: gene
expression, road networks, social-network communities.

## Key takeaways

- K-means: ~75% purity. It bisects the picture along a straight line.
- DBSCAN with eps=0.18, min_samples=5: ≥ 99% purity, recovers both crescents.
- DBSCAN finds k automatically and labels noise as -1; that's a feature, not a bug.
- Choosing eps is the main practical challenge; the k-distance elbow plot is the
  standard heuristic.
