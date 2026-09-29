# Note

According to my test, K-means is roughly balanced if the `n_cluster` is small. Thus a possible solution is to use the hierarchical K-means.

`faiss` has a good implementation.

- [`faiss` wiki](https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster#k-means-clustering)
- [`faiss` code](https://github.com/facebookresearch/faiss/blob/2e6551ffa3f6fbdb1ba814c2c531fb399b00d4e3/faiss/python/extra_wrappers.py#L443)

## Parameters

### Progressive Dim Clustering Parameters

- progressive_dim_steps = 10
- apply_pca = true
- niter = 10

### Clustering Parameters

- niter = 25
- nredo = 1

- spherical = false (normalize centroids after each iteration for inner production)
  - true: use IVF flat IP
  - false: use IVF flat L2
- int_centroids = false (round centroids after each iteration)
- update_index = false (re-train index after each iteration)

- frozen_centroids = false (use provided centroids)
- min_points_per_centroid = 39
- max_points_per_centroid = 256 (subsample if exceeds)
- seed = 1234

- decode_block_size = 32768 (batch size of the codec decoder when the training set is encoded)

- check_input_data_for_NaNs = true
- use_faster_subsampling = false (splitmix64-based RNG, faster but could pick duplicates)

## Training

### Clustering

- clustering
  - assert (num >= k)
  - cast float to u8 (for convenient indexing, line_size is sizeof(float) * d)
  - subsample (if num > max_points_per_centroid * k)
    - shuffle and pick the top-(k * max_points_per_centroid)
  - random select k points as the initial centroids
  - post process centroids
    - normalize (if spherical)
    - round (if int_centroids)
  - index (IVF flat) add centroids
  - niter
    - search and assign
    - compute centroids
      - new centroids = sum(points) / count(points)
    - split clusters
      - for clusters that is empty
        - iter from cluster 0..
          - if (rand() < (cluster_size - 1) / (n - k))  // here n is the sampled points number
          - copy the centroid to the empty cluster and apply symmetric permutation
            - d_i%2 == 0: centroid[i][d_i] *= 1 + EPS, centroid[i][d_i+1] *= 1 - EPS
            - d_i%2 == 1: centroid[i][d_i] *= 1 - EPS, centroid[i][d_i-1] *= 1 + EPS
            - EPS = 1 / 1024.0
          - even split these two clusters
    - post process centroids
    - update index

### Progressive Dim Clustering

- train PCA for (n, x)
- iteration: di = int(pow(d, (1.0 + i) / progressive_dim_steps))
  - copy the centroids from the previous iteration
  - clustering for (n, xsub(di))
- revert PCA transform on centroids
