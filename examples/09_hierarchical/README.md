# 09 — Hierarchical Clustering on Iris

**Notebook:** [`hierarchical_iris.ipynb`](hierarchical_iris.ipynb)

Agglomerative hierarchical clustering on Iris with three linkage methods compared
side-by-side. The point is to show the dendrogram as the *primary output* of the
algorithm — the cluster assignments are derived from it, not the other way around —
and to demonstrate the well-known sensitivity of single linkage to chaining.

## Why this dataset?

Iris is small enough to plot a full dendrogram readably, and we know what 'right' looks
like (three species) so we can score the result. The single/complete/average linkage
disagreement is sharp on this data, which makes the comparison concrete.

## Key takeaways

- Single and average linkages put setosa alone on the top split (50 vs 100); complete linkage's top split is 73 vs 77 instead (it groups setosa with ~20 versicolor flowers), but all three still recover setosa as essentially a clean cluster at k=3.
- Single linkage chains, complete linkage gives compact balanced clusters, average linkage sits in between. Complete is best on Iris by best-match accuracy (0.787 vs single 0.660 vs average 0.687).
- The dendrogram is the output, not just a visualization. Tall vertical lines indicate
  stable cluster counts.
