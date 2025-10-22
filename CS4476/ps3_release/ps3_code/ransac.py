import math
from typing import Tuple

import numpy as np

from ps3_code import fundamental_matrix, two_view_data
from ps3_code.least_squares_fundamental_matrix import solve_F


def calculate_num_ransac_iterations(prob_success: float, 
                                    sample_size: int, 
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int representing the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    # TODO: Student code goes here
    all_correct = ind_prob_correct ** sample_size
    num_samples = np.log(1 - prob_success) / np.log(1 - all_correct)
    ##############################

    return num_samples


def find_inliers(x_0s: np.ndarray, 
                 F: np.ndarray, 
                 x_1s: np.ndarray, 
                 threshold: float) -> np.ndarray:
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix of shape (3, 3)
    -   x_1s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: float representing the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """
    inliers = None

    ##############################
    # TODO: Student code goes here
    Fx1 = (F @ x_1s.T).T
    FTx0 = (F.T @ x_0s.T).T

    num0 = np.abs(np.sum(Fx1 * x_0s, axis=1))
    num1 = np.abs(np.sum(FTx0 * x_1s, axis=1))

    denom0 = np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2) + 1e-12
    denom1 = np.sqrt(FTx0[:, 0]**2 + FTx0[:, 1]**2) + 1e-12

    error = np.maximum((num0 / denom0), (num1 / denom1))
    inliers = np.where(error < threshold)[0]


    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s: int, 
                              x_1s: int) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.
        4. You can use the `preprocess_data` function in `two_view_data` to make
           x_0s, and x_1s homogeneous.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliers_x_0 = None
    inliers_x_1 = None

    ##############################
    # TODO: Student code goes here
    k = 9
    p = 0.9
    P = 0.999
    threshold = 1.0

    x0 = np.asarray(x_0s)
    x1 = np.asarray(x_1s)
    N = x0.shape[0]

    x0h = np.hstack([x0, np.ones((N, 1))])
    x1h = np.hstack([x1, np.ones((N, 1))])

    S = int(np.ceil(calculate_num_ransac_iterations(P, k, p)))
    best_inlier_idx = None
    best_count = -1

    for _ in range(S):
        idx = np.random.choice(N, size=k, replace=False)
        candidate = solve_F(x0[idx], x1[idx])
        inlier_idx = find_inliers(x0h, candidate, x1h, threshold)
        count = inlier_idx.size
        if count > best_count:
            best_count = count
            best_inlier_idx = inlier_idx
            best_F = candidate
    
    best_F = solve_F(x0[best_inlier_idx], x1[best_inlier_idx])
    inliers_x_0 = x0[best_inlier_idx]
    inliers_x_1 = x1[best_inlier_idx]
    ##############################

    return best_F, inliers_x_0, inliers_x_1
