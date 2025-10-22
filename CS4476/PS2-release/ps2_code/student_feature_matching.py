import numpy as np
import sklearn.neighbors

def compute_feature_distances(features1, features2):
    """
    This function computes a list of euclidean distances from every feature in one array to every feature in another.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of features
    - features2: A numpy array of shape (m,feat_dim) representing the second set features

    Note: n, m is the number of feature (m not necessarily equal to n);
          feat_dim denotes the feature dimensionality;

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2. That is, dists[i][j]
      represents the distance from the i’th feature in features1 to the j’th
      feature in features2.

    Note: If your approach involves vectorizing the entire operation, you might run out of memory.
    One for-loop works well for this approach.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    f1 = np.asarray(features1, dtype=np.float32)
    f2 = np.asarray(features2, dtype=np.float32)

    n, d = f1.shape
    m = f2.shape[0]
    dists = np.empty((n, m), dtype=np.float32)
    f2_norm2 = np.sum(f2 * f2, axis=1)

    for i in range(n):
        a = f1[i]
        a_norm2 = np.sum(a * a).astype(np.float32)
        dots = f2 @ a
        sq = a_norm2 + f2_norm2 - 2.0 * dots
        sq = np.maximum(sq, 0.0, dtype=np.float32)
        dists[i, :] = np.sqrt(sq, dtype=np.float32)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).
    To start with, simply implement the NNDR, "ratio test", which is the equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper). Experiment with different ratios to achieve the
    best performance you can - we suggest starting with 0.5 and moving
    up from there.

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Step:
    1. Use `compute_feature_distances()` to find out the distance
    2. Implement the NNDR equation to find out the match
    3. Record the match indices ('matches') and distance of the match ('Confidences')

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of features
    - features2: A numpy array of shape (m,feat_dim) representing the second set features

    Note: n, m is the number of feature (m not necessarily equal to n);
          feat_dim denotes the feature dimensionality;

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match, which is the distance between matched pair.

    E.g. The first feature in 'features1' matches to the third feature in 'features2'.
         Then the output value for 'matches' should be [0,2] and 'confidences' [0.9]

    Note: 'matches' and 'confidences' can be empty which has shape (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    f1 = np.asarray(features1, dtype=np.float32)
    f2 = np.asarray(features2, dtype=np.float32)

    n = f1.shape[0]
    m = f2.shape[0]
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, ), dtype=np.float32)

    dists = compute_feature_distances(f1, f2)

    matches_list = []
    conf_list = []
    ratio_threshold = 0.9

    for i in range(n):
        row = dists[i]
        if m < 2:
            continue

        nn2_idx = np.argpartition(row, 2)[:2]
        nn2_idx = nn2_idx[np.argsort(row[nn2_idx])]
        j1, j2 = int(nn2_idx[0]), int(nn2_idx[1])
        d1, d2 = float(row[j1]), float(row[j2])

        if d2 <= 1e-12:
            continue
                
        ratio = d1 / d2
        if ratio < ratio_threshold:
            matches_list.append([i, j1])
            conf_list.append(1.0 - ratio)

    if len(matches_list) == 0:
        matches = np.zeros((0, 2), dtype=np.int32)
        confidences = np.zeros((0, ), dtype=np.float32)
    else:
        matches = np.asarray(matches_list, dtype=np.int32)
        confidences = np.asarray(conf_list, dtype=np.float32)
        order = np.argsort(-confidences)
        matches = matches[order]
        confidences = confidences[order]


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences

def pca(fvs1, fvs2, n_components= 24):
    """
    Perform PCA to reduce the number of dimensions in each feature vector which results in a speed up.
    You will want to perform PCA on all the data together to obtain the same principle components.
    You will then resplit the data back into image1 and image2 features.

    Helpful functions: np.linalg.svd, np.mean, np.cov

    Args:
    -   fvs1: numpy nd-array of feature vectors with shape (n,128) for number of interest points
        and feature vector dimension of image1
    -   fvs1: numpy nd-array of feature vectors with shape (m,128) for number of interest points
        and feature vector dimension of image2
    -   n_components: m desired dimension of feature vector

    Return:
    -   reduced_fvs1: numpy nd-array of feature vectors with shape (n, m) with m being the desired dimension for image1
    -   reduced_fvs2: numpy nd-array of feature vectors with shape (n, m) with m being the desired dimension for image2
    """

    reduced_fvs1, reduced_fvs2 = None, None
    #############################################################################
    # TODO: YOUR PCA CODE HERE                                                  #
    #############################################################################

    combined = np.concatenate((fvs1, fvs2)).astype(np.float64)
    mean_vector = combined.mean(axis=0)
    centered = combined - mean_vector

    cov = np.cov(centered.T)
    eigen_val, eigen_vec = np.linalg.eig(cov)

    order = np.argsort(-eigen_val.real)
    top_vec = eigen_vec[:, order[:n_components]].real

    projected = centered @ top_vec
    n1 = fvs1.shape[0]
    reduced_fvs1 = projected[:n1]
    reduced_fvs2 = projected[n1:]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return reduced_fvs1, reduced_fvs2

def accelerated_matching(features1, features2):
    """
    This method should operate in the same way as the match_features function you already coded.
    Try to make any improvements to the matching algorithm that would speed it up.
    One suggestion is to use a space partitioning data structure like a kd-tree or some
    third party approximate nearest neighbor package to accelerate matching.

    Note: Doing PCA here does not count. This implementation MUST be faster than PCA
    to get credit.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                  #
    #############################################################################

    threshold = 0.82
    A = np.asarray(features1, dtype=np.float64)
    B = np.asarray(features2, dtype=np.float64)
    na = np.linalg.norm(A, axis=1, keepdims=True); na[na == 0.0] = 1.0
    nb = np.linalg.norm(B, axis=1, keepdims=True); nb[nb == 0.0] = 1.0
    A_norm = A / na
    B_norm = B / nb
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean')
    nn.fit(B)

    dists, neighbors = nn.kneighbors(A_norm, n_neighbors=2, return_distance=True)

    eps = 1e-12
    d1 = dists[:, 0]
    d2 = dists[:, 1]
    valid = (d2 > eps) & np.isfinite(d1) & np.isfinite(d2) & ((d1 / d2) < threshold)

    i_idx = np.nonzero(valid)[0]
    j_idx = neighbors[valid, 0].astype(np.int64)
    confidence = d1[valid].astype(np.float64)
    order = np.argsort(confidence)
    matches = np.stack((i_idx[order], j_idx[order]), axis=1)
    confidences = confidence[order]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return matches, confidences
