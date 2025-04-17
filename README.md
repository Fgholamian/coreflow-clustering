# CoreFlow Clustering

**CoreFlow Clustering** is a novel unsupervised clustering algorithm that works by iteratively shifting points toward the average of their local neighborhood — simulating data flow into density cores.

## How it works

1. Each point finds its k nearest neighbors.
2. It shifts toward the mean of these neighbors.
3. After several iterations, points settle into density cores.
4. A secondary clustering method (like DBSCAN or Agglomerative Clustering) assigns final labels.

##Files

- `coreflow.py`: Main algorithm implementation.
- `demo_data.py`: Synthetic 4-cluster dataset demo using DBSCAN.
- `requirements.txt`: Dependencies.

## Example Result

The code will visualize the original data and the clustered result side by side.

##Why CoreFlow?

- Does not require specifying `k` (k-means)
- Resistant to noise and shape variations
- Flexible with different post-clustering methods

---

##  Install & Run

```bash
pip install -r requirements.txt
python demo_data.py