from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt


class Eigenfaces(object):

    def __init__(self):
        pass

    def svd(self, X: np.ndarray) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """		
		First, 0-center the face dataset by subtracting the mean of each image (do not loop).
		Then, perform Singular Value Decomposition (SVD) on the given face dataset to compute
		the eigenfaces. You should use numpy.linalg.svd,
		unless you really want a linear algebra exercise.
		
		Args:
		    X: (N, D) numpy array where each row represents a flattened grayscale face image.
		Returns:
		    U: (N, min(N, D)) numpy array of left singular vectors
		    S: (min(N, D), ) numpy array of singular values
		    V: (min(N, D), D) numpy array of transposed right singular vectors
		Hints:
		    Set full_matrices=False when computing SVD!
		    The rows of our matrix are of size D, which is H*W of the images, which will be massive for any reasonably sized images.
		    Thus, there will only be N singular values.
		    Therefore, there's no reason to allocate a bunch of memory for invalid right singular vectors which won't have a corresponding singular value.
		    This is necessary, since an array of (H*W)^2 will take up too much memory.
		"""
        centered = X - np.mean(X, axis=0)
        # Referred to https://numpy.org/doc/2.2/reference/generated/numpy.linalg.svd.html
        U, S, V = np.linalg.svd(centered, full_matrices=False)
        
        return U, S, V

    def compress(self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
        ) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """		
		Compress the SVD factorization by keeping only the first k components.
		
		Args:
		    U (np.ndarray): (N, min(N, D)) numpy array
		    S (np.ndarray): (min(N,D), ) numpy array
		    V (np.ndarray): (min(N, D), D) numpy array
		    k (int): int corresponding to number of components to keep
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray]:
		        U_compressed: (N, k) numpy array
		        S_compressed: (k, ) numpy array
		        V_compressed: (k, D) numpy array
		"""
        U_compressed = U[:, :k]
        S_compressed = S[:k]
        V_compressed = V[:k, :]
        
        return U_compressed, S_compressed, V_compressed

    def rebuild_svd(self, U_compressed: np.ndarray, S_compressed: np.
        ndarray, V_compressed: np.ndarray) ->np.ndarray:
        """		
		Rebuild original matrix X from U, S, and V which have been compressed to k componments.
		
		Args:
		    U_compressed: (N,k) numpy array
		    S_compressed: (k, ) numpy array
		    V_compressed: (k,D) numpy array
		Returns:
		    Xrebuild: (N,D) numpy array of reconstructed matrix
		Hints:
		    Recall the definition of Singular Value Decomposition for guidance in reconstructing the image.
		"""
        Xrebuild = U_compressed @ np.diag(S_compressed) @ V_compressed
        
        return Xrebuild

    def compute_eigenfaces(self, X: np.ndarray, k: int) ->np.ndarray:
        """		
		Compute the top k "eigenfaces" (here, simply the right singular vectors) from the given face dataset using SVD.
		
		Args:
		    X: (N, D) numpy array where each row is a flattened face image.
		    k: Number of eigenfaces to retain.
		Returns:
		    Eigenfaces: (k, D) numpy array where each row represents an eigenface.
		"""
        centered = X - np.mean(X, axis=0)
        U, S, V = np.linalg.svd(centered, full_matrices=False)
        eigenfaces = V[:k, :]
        
        return eigenfaces

    def compression_ratio(self, X: np.ndarray, k: int) ->float:
        """		
		Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)
		Refer to https://timbaumann.info/svd-image-compression-demo/
		
		Args:
		    X: (N,D) numpy array corresponding to a single image
		    k: int corresponding to number of components
		Returns:
		    compression_ratio: float of proportion of storage used by compressed image
		"""
        og_size = np.prod(X.shape)
        compressed_size = X.shape[0] * k + k + k * X.shape[1]
        compression_ratio = compressed_size / og_size
        
        return compression_ratio

    def recovered_variance_proportion(self, S: np.ndarray, k: int) ->float:
        """		
		Compute the proportion of the variance in the original matrix recovered by a rank-k approximation
		
		Args:
		   S: (min(N,D), ) numpy array
		   k: int, rank of approximation
		Returns:
		   recovered_var: float corresponding to proportion of recovered variance
		"""
        og_variance = np.sum(S ** 2)
        recovered_variance = np.sum(S[:k] ** 2)
        variance_proportion = recovered_variance / og_variance
        
        return variance_proportion
