# 11 — Bonus: Relational / Graph Data Analysis

**Notebook:** [`graph_iris_community.ipynb`](graph_iris_community.ipynb)

A bonus notebook that bridges the vector-data ML in the rest of the repo to the
relational/graph data introduced in lecture (the food-truck example, Instagram
follow-graph). The pipeline:

1. Take a feature-vector dataset (Iris).
2. Build a **similarity graph** by connecting points within distance threshold τ.
3. Recover community structure using *only* the graph (BFS-based connected components).
4. Sweep τ and observe how the recovery quality changes.

## Why this matters

This was directly motivated by the lecture remark that vector data and graph data are
two different views of the world, and that some structure (the food-truck story) is
only visible as a graph. Here we show the converse direction: even data that started
as feature vectors can be usefully re-expressed as a graph, and graph-native
algorithms then become applicable.

## Key takeaways

- The connected-component method recovers the same clustering structure as k-means
  and hierarchical clustering on Iris — strong cross-validation that the structure is
  real.
- τ is the only knob, and the τ-sweep plot exposes a quantitative picture of where the
  versicolor/virginica boundary is fuzzy.
- This is the conceptual precursor to spectral clustering, community detection
  algorithms (Louvain, Leiden), and graph neural networks.
