from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def recover_E_from_F(f_matrix: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    '''
    Recover the essential matrix from the fundamental matrix

    Args:
    -   f_matrix: fundamental matrix as a numpy array (shape=(3,3))
    -   k_matrix: the intrinsic matrix shared between the two cameras (shape=(3,3))
    Returns:
    -   e_matrix: the essential matrix as a numpy array (shape=(3,3))
    '''

    e_matrix = None

    ##############################
    # TODO: Student code goes here
    K = k_matrix.astype(float)
    F = f_matrix.astype(float)
    e_matrix = K.T @ F @ K
    ##############################

    return e_matrix

def recover_rot_translation_from_E(e_matrix: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    '''
    Decompose the essential matrix to get rotation and translation (upto a scale)

    Ref: Section 9.6.2 of Hartley and Zisserman's Multiple View Geometry in Computer Vision (Second Edition)

    Hint:
    - Refer to the docs for `Rotation.from_matrix` and `Rotation.as_rotvec` in scipy.spatial.transform module
    
    Args:
    -   e_matrix: the essential matrix as a numpy array (3 x 3 ndarray)
    Returns:
    -   R1: the (3,) array containing the rotation angles in radians; one of the two possible
    -   R2: the (3,) array containing the rotation angles in radians; other of the two possible
    -   t: a (3,) translation matrix with unit norm and +ve x-coordinate; if x-coordinate is zero then y should be positive, and so on.


    '''

    R1 = None
    R2 = None
    t = None

    ##############################
    # TODO: Student code goes here
    U, S, Vt = np.linalg.svd(e_matrix)

    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    R1m = U @ W @ Vt
    R2m = U @ W.T @ Vt

    if np.linalg.det(R1m) < 0:
        R1m = -R1m

    if np.linalg.det(R2m) < 0:
        R2m = -R2m
    
    # https://scipy.github.io/devdocs/reference/generated/scipy.spatial.transform.Rotation.as_rotvec.html
    R1 = Rotation.from_matrix(R1m).as_rotvec()
    R2 = Rotation.from_matrix(R2m).as_rotvec()

    t = U[:, 2]
    t = t / np.linalg.norm(t)

    epsilon = 1e-12

    if abs(t[0]) > epsilon:
        if t[0] < 0:
            t = -t
    elif abs(t[1]) > epsilon:
        if t[1] < 0:
            t = -t
    else:
        if t[2] < 0:
            t = -t
    ##############################

    return R1, R2, t
