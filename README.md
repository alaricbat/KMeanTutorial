# Custom K-Means Clustering

A from-scratch implementation of the **K-Means clustering algorithm** using Python and NumPy.

This project is intended for learning and experimentation with the internal mechanism of K-Means, including centroid initialization, cluster assignment, centroid update, convergence, visualization, and future optimization techniques.

---

## 1. Project Overview

K-Means is an unsupervised machine learning algorithm that partitions a dataset into `K` clusters.

The algorithm iteratively performs three main steps:

1. Initialize `K` cluster centroids.
2. Assign each data point to the nearest centroid.
3. Recalculate each centroid using the mean of the points assigned to that cluster.

The process continues until the centroids converge or a maximum number of iterations is reached.

### Current implementation

The current `CustomKMeans` implementation follows the basic K-Means workflow:

```text
Input Dataset
     │
     ▼
Initialize Centroids
     │
     ▼
Assign Data Points
to Nearest Centroid
     │
     ▼
Update Centroids
using Cluster Mean
     │
     ▼
Check Convergence
     │
   ┌─┴─┐
   │   │
 No    Yes
   │   │
   └───┘
     │
     ▼
Return Model
```

---

## 2. Repository Structure

```text
.
├── CustomKMeans.py
├── DisplayUtil.py
│
├── KMeansCluseringMNIST.ipynb
├── KMeansClusteringObjSegmentation.ipynb
└── KMeansClusteringVisualize.ipynb
```

### Files

| File                                    | Description                                         |
| --------------------------------------- | --------------------------------------------------- |
| `CustomKMeans.py`                       | Main from-scratch K-Means implementation            |
| `DisplayUtil.py`                        | Utility functions for visualizing image/filter data |
| `KMeansCluseringMNIST.ipynb`            | K-Means experiments on MNIST                        |
| `KMeansClusteringObjSegmentation.ipynb` | K-Means application for object segmentation         |
| `KMeansClusteringVisualize.ipynb`       | Visualization and exploration of K-Means            |

---

## 3. Current Implementation

The current implementation contains the fundamental K-Means operations.

### 3.1 Model Initialization

```python
class CustomKMeans:
    def __init__(self, n_clusters=5):
        self.k = n_clusters
        self.labels = []
```

The number of clusters is controlled through `n_clusters`.

---

### 3.2 Random Centroid Initialization

The current implementation randomly selects `K` observations from the training data as initial centroids.

```python
self.centroids = self.X_train[
    np.random.choice(
        a=self.X_train.shape[0],
        size=self.k,
        replace=False
    )
]
```

This corresponds to the basic **random initialization** strategy.

### Current status

* `random` initialization: **Implemented**
* `k-means++`: **Not implemented**
* Custom initial centroids: **Not implemented**
* Callable initialization: **Not implemented**

---

## 4. Cluster Assignment

For each observation, the implementation calculates the squared Euclidean distance to each centroid.

```python
distances = np.sum(
    (self.X_train[:, np.newaxis, :] - self.centroids) ** 2,
    axis=2
)
```

The observation is assigned to the centroid with the smallest distance:

```python
np.argmin(distances, axis=1)
```

Therefore, the current implementation uses the nearest-centroid rule based on squared Euclidean distance.

---

## 5. Centroid Update

After assigning observations to clusters, the centroid of each cluster is recalculated as the mean of the observations belonging to that cluster.

```python
for i in range(self.k):
    Xi = self.X_train[self.labels == i, :]

    if len(Xi) == 0:
        centroids[i, :] = self.centroids[i, :]
    else:
        centroids[i, :] = np.mean(Xi, axis=0)
```

An empty cluster currently keeps its previous centroid.

---

## 6. Convergence

The current implementation checks whether the old and new centroid positions are sufficiently close.

```python
np.allclose(
    self.centroids,
    new_centroids,
    atol=1e-4
)
```

The current tolerance is therefore hard-coded to:

```text
atol = 1e-4
```

A future version will expose this value through a configurable `tol` parameter.

---

## 7. Prediction

After fitting the model, new observations can be assigned to their nearest centroid using:

```python
model.predict(X)
```

The prediction process uses the same nearest-centroid assignment mechanism as training.

---

# 8. K-Means Parameters and Future Implementation Plan

The long-term goal of this project is to extend the custom implementation with functionality commonly associated with production K-Means implementations.

## 8.1 Initialization

### `init`

Planned supported initialization methods:

```text
init = "k-means++"
init = "random"
init = array-like
init = callable
```

| Method       | Description                                                              | Status        |
| ------------ | ------------------------------------------------------------------------ | ------------- |
| `random`     | Randomly select `K` observations as initial centroids                    | ✅ Implemented |
| `k-means++`  | Select initial centroids using a distance-based probability strategy     | ⬜ Planned     |
| `array-like` | Allow the user to provide initial centroids with shape `(K, n_features)` | ⬜ Planned     |
| `callable`   | Allow a user-defined initialization function                             | ⬜ Planned     |

### Implementation plan

```text
Initialization
      │
      ├── random
      │
      ├── k-means++
      │
      ├── user-provided centroids
      │
      └── custom callable
```

---

# 9. Random State

## `random_state`

The current implementation uses NumPy's random generator directly.

Because no random seed is currently exposed, repeated executions may produce different initial centroids and therefore different clustering results.

### Planned API

```python
CustomKMeans(
    n_clusters=5,
    random_state=42
)
```

### Goal

Make experiments reproducible:

```text
Same dataset
     +
Same parameters
     +
Same random_state
     ↓
Same initialization
     ↓
Same clustering result
```

---

# 10. Maximum Number of Iterations

## `max_iter`

The current implementation uses:

```python
while True:
```

and stops when the centroids converge.

A future implementation will add a maximum iteration limit.

### Planned API

```python
CustomKMeans(
    n_clusters=5,
    max_iter=300
)
```

### Motivation

A maximum iteration limit prevents the optimization loop from continuing indefinitely when convergence is slow or does not occur.

---

# 11. Convergence Tolerance

## `tol`

The current implementation uses a fixed convergence threshold:

```python
atol=1e-4
```

This will be changed to a configurable parameter.

### Planned API

```python
CustomKMeans(
    n_clusters=5,
    tol=1e-4
)
```

This allows experiments with different convergence criteria.

---

# 12. Inertia

## `inertia_`

A future version will calculate the K-Means objective function, commonly referred to as **inertia**.

The objective is based on the sum of squared distances between each observation and the centroid of its assigned cluster.

Planned model attribute:

```python
model.inertia_
```

### Why implement inertia?

Inertia can be used to:

* evaluate the compactness of clusters;
* compare different K-Means runs;
* select the best result among multiple initializations;
* visualize the effect of different values of `K`.

---

# 13. Multiple Initializations

## `n_init`

K-Means can produce different results depending on the initial centroid positions.

A future implementation will support running K-Means multiple times with different initializations.

```python
CustomKMeans(
    n_clusters=5,
    n_init=10
)
```

Conceptually:

```text
                 ┌── Run 1 ── inertia = ...
                 │
Initializations ─┼── Run 2 ── inertia = ...
                 │
                 ├── Run 3 ── inertia = ...
                 │
                 └── Run N ── inertia = ...
                              │
                              ▼
                       Select best result
```

The implementation will retain the result with the lowest inertia.

---

# 14. K-Means Optimization Algorithms

The project will eventually support different algorithms for the optimization stage.

## 14.1 Lloyd

The classical K-Means procedure can be represented as:

```text
Initialize centroids
        ↓
Assign observations
        ↓
Update centroids
        ↓
Check convergence
        ↓
Repeat
```

The current implementation follows this general approach.

### Status

**Partially implemented / current algorithm**

---

## 14.2 Elkan

A future implementation may support an Elkan-style optimization.

The main idea is to reduce unnecessary distance calculations by using the triangle inequality.

Conceptually:

```text
Standard approach
-----------------
Every point
    ↓
Calculate distance
to every centroid
    ↓
Select nearest centroid


Elkan approach
--------------
Every point
    ↓
Use distance bounds
    ↓
Avoid unnecessary
distance calculations
    ↓
Calculate only
when necessary
```

### Advantages

Potentially fewer distance calculations for datasets with well-separated clusters.

### Trade-off

Additional memory is required to maintain distance-related bounds.

### Status

**Planned — Advanced**

---

# 15. Empty Cluster Handling

The current implementation already handles an empty cluster.

If no observations are assigned to a cluster, the implementation keeps the previous centroid:

```python
if len(Xi) == 0:
    centroids[i, :] = self.centroids[i, :]
```

### Current status

**Implemented**

### Future work

Investigate alternative strategies for empty clusters and compare their effects on convergence and final clustering results.

---

# 16. Input Validation

Input validation is not currently implemented as a dedicated component.

Future versions should validate:

* `n_clusters > 0`;
* number of clusters does not exceed the number of observations;
* input is two-dimensional;
* number of features is consistent;
* input does not contain unsupported values;
* initial centroid shape is `(n_clusters, n_features)`;
* initialization parameters are valid.

### Status

**Planned**

---

# 17. Data Handling

## `copy_x`

A future implementation may provide control over whether input data is copied during preprocessing.

Planned parameter:

```python
CustomKMeans(
    n_clusters=5,
    copy_x=True
)
```

This is a lower-priority feature compared with the core clustering algorithm.

### Status

**Planned — Optional**

---

# 18. Visualization

Visualization is an important part of this project because the notebooks are also used to understand how K-Means works.

Future visualization experiments may include:

### 18.1 Cluster assignment

```text
Data points
    ↓
Cluster labels
    ↓
Visualize clusters
```

### 18.2 Centroid movement

Track the centroid positions at every iteration.

```text
Iteration 0
     ↓
Iteration 1
     ↓
Iteration 2
     ↓
...
     ↓
Convergence
```

### 18.3 Inertia versus K

Run K-Means using different values of `K` and visualize the corresponding inertia.

```text
K = 2  → inertia
K = 3  → inertia
K = 4  → inertia
K = 5  → inertia
...
```

This can be used for experimentation with the relationship between the number of clusters and clustering compactness.

---

# 19. Applications

The current project includes experiments involving different types of data.

## MNIST

K-Means can be applied to image data by representing each image as a feature vector.

```text
Image
  ↓
Flatten pixels
  ↓
Feature vector
  ↓
K-Means
  ↓
Cluster assignment
```

Notebook:

```text
KMeansCluseringMNIST.ipynb
```

---

## Object Segmentation

K-Means can also be used for image segmentation.

Conceptually:

```text
Original Image
      ↓
Pixel Features
      ↓
K-Means
      ↓
Cluster Labels
      ↓
Segmented Image
```

Notebook:

```text
KMeansClusteringObjSegmentation.ipynb
```

---

## Visualization

The project also contains visualization experiments for understanding how the clustering process behaves.

Notebook:

```text
KMeansClusteringVisualize.ipynb
```

---

# 20. Development Roadmap

The implementation will be developed incrementally.

## Phase 1 — Core Stability

* [x] Random centroid initialization
* [x] Cluster assignment
* [x] Centroid update
* [x] Empty cluster handling
* [x] Convergence check
* [x] Prediction
* [ ] `max_iter`
* [ ] Configurable `tol`
* [ ] `random_state`
* [ ] Input validation
* [ ] Inertia

---

## Phase 2 — Initialization Improvements

* [ ] `random` initialization as an explicit option
* [ ] `k-means++`
* [ ] Array-like initial centroids
* [ ] Callable initialization
* [ ] Initialization validation

---

## Phase 3 — Multiple Initializations

* [ ] `n_init`
* [ ] Multiple K-Means runs
* [ ] Compare inertia values
* [ ] Select the best clustering result
* [ ] Reproducibility tests

---

## Phase 4 — Algorithm Improvements

* [ ] Explicit Lloyd implementation
* [ ] Separate assignment/update components
* [ ] Elkan optimization
* [ ] Performance comparison between Lloyd and Elkan

---

## Phase 5 — Analysis and Visualization

* [ ] Centroid movement visualization
* [ ] Inertia per iteration
* [ ] Inertia versus `K`
* [ ] Cluster visualization
* [ ] Convergence analysis
* [ ] Runtime comparison

---

## Phase 6 — Benchmarking

Compare the custom implementation against a reference K-Means implementation.

Potential comparison metrics:

| Metric               | Custom K-Means | Reference implementation |
| -------------------- | -------------: | -----------------------: |
| Runtime              |            TBD |                      TBD |
| Inertia              |            TBD |                      TBD |
| Number of iterations |            TBD |                      TBD |
| Initialization       |            TBD |                      TBD |
| Clustering result    |            TBD |                      TBD |

The purpose of the benchmark is to understand the differences between the from-scratch implementation and a mature implementation rather than to assume that the implementations are equivalent.

---

# 21. Planned API

The eventual API may look like:

```python
model = CustomKMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=42,
    algorithm="lloyd"
)

model.fit(X)

labels = model.predict(X)

print(model.centroids)
print(model.inertia_)
```

The parameters will be introduced incrementally as the corresponding implementations are completed.

---

# 22. Implementation Priority

The planned features are prioritized as follows:

| Priority    | Feature                   | Reason                              |
| ----------- | ------------------------- | ----------------------------------- |
| 🔴 High     | `k-means++`               | Important initialization technique  |
| 🔴 High     | `random_state`            | Reproducible experiments            |
| 🔴 High     | `max_iter`                | Robust termination condition        |
| 🔴 High     | `tol`                     | Configurable convergence            |
| 🔴 High     | `inertia`                 | Core K-Means objective              |
| 🔴 High     | `n_init`                  | Reduce dependence on initialization |
| 🟡 Medium   | Input validation          | Improve robustness                  |
| 🟡 Medium   | Array-like initialization | API flexibility                     |
| 🟡 Medium   | Callable initialization   | Custom experimentation              |
| 🟡 Medium   | Visualization             | Better algorithm analysis           |
| 🟠 Advanced | Elkan                     | Algorithmic optimization            |
| 🟢 Low      | `copy_x`                  | Data-handling compatibility         |

---

# 23. Current vs. Target Implementation

```text
CURRENT
=======

CustomKMeans
    │
    ├── Random Initialization
    │
    ├── Euclidean Distance
    │
    ├── Nearest Centroid Assignment
    │
    ├── Mean-based Centroid Update
    │
    ├── Empty Cluster Handling
    │
    └── Convergence Check


TARGET
======

CustomKMeans
    │
    ├── Initialization
    │   ├── random
    │   ├── k-means++
    │   ├── array-like
    │   └── callable
    │
    ├── Reproducibility
    │   └── random_state
    │
    ├── Multiple Initialization
    │   └── n_init
    │
    ├── Optimization
    │   ├── Lloyd
    │   └── Elkan
    │
    ├── Convergence
    │   ├── max_iter
    │   └── tol
    │
    ├── Evaluation
    │   └── inertia
    │
    ├── Robustness
    │   ├── input validation
    │   └── empty-cluster strategies
    │
    └── Analysis
        ├── visualization
        └── benchmarking
```

---

# 24. Learning Goals

This project is primarily focused on understanding the implementation of K-Means rather than simply using an existing machine learning library.

The main learning goals are:

1. Understand how centroid initialization affects K-Means.
2. Implement different initialization strategies.
3. Understand the assignment and update steps.
4. Understand convergence criteria.
5. Understand the K-Means objective function.
6. Investigate the effect of multiple initializations.
7. Understand the difference between Lloyd and Elkan approaches.
8. Analyze clustering behavior through visualization.
9. Compare a from-scratch implementation with a reference implementation.
10. Study how implementation choices affect clustering quality and runtime.

---

# 25. Status

**Project status: In Development**

Current implementation:

```text
Core K-Means
    ├── Random initialization      ✅
    ├── Assignment step            ✅
    ├── Centroid update             ✅
    ├── Empty cluster handling      ✅
    ├── Convergence check           ✅
    └── Prediction                  ✅

Advanced features
    ├── K-Means++                   ⬜
    ├── random_state                ⬜
    ├── max_iter                    ⬜
    ├── configurable tol            ⬜
    ├── inertia                     ⬜
    ├── n_init                      ⬜
    ├── custom initialization       ⬜
    ├── Lloyd option                ⬜
    ├── Elkan                       ⬜
    └── benchmarking                ⬜
```

---

## 26. Next Step

The next implementation step is:

```text
1. Add max_iter
        ↓
2. Add configurable tol
        ↓
3. Add random_state
        ↓
4. Implement inertia
        ↓
5. Implement k-means++
        ↓
6. Implement n_init
        ↓
7. Add custom initialization
        ↓
8. Add Lloyd / Elkan options
        ↓
9. Benchmark and visualize
```

This order keeps the implementation incremental: first make the existing algorithm robust, then improve initialization and reproducibility, and finally work on advanced optimization.