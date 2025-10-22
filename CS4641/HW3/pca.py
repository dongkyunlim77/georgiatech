import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):

	def __init__(self):
		self.U = None
		self.S = None
		self.V = None

	def fit(self, X: np.ndarray) ->None:
		"""		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
			  Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
			X: (N,D) numpy array corresponding to a dataset
		
		Return:
			None
		
		Set:
			self.U: (N, min(N,D)) numpy array
			self.S: (min(N,D), ) numpy array
			self.V: (min(N,D), D) numpy array
		"""
		centered = X - np.mean(X, axis=0)
		self.U, self.S, self.V = np.linalg.svd(centered, full_matrices=False)

	def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
		"""		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		by projecting onto the principal components.
		Utilize class members that were previously set in fit() method.
		
		Args:
			data: (N,D) numpy array corresponding to a dataset
			K: int value for number of columns to be kept
		
		Return:
			X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
		centered = data - np.mean(data, axis=0)
		project_v = self.V[:K, :]
		X_new = centered @ project_v.T
		
		return X_new

	def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
		) ->np.ndarray:
		"""		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
			data: (N,D) numpy array corresponding to a dataset
			retained_variance: float value for amount of variance to be retained
		
		Return:
			X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
				   to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
		# Referred to https://www.geeksforgeeks.org/numpy-cumsum-in-python/ to get cumulative sum of eigenvalues
		variance_ratio = np.cumsum(self.S ** 2) / np.sum(self.S ** 2)
		# Referred to https://www.geeksforgeeks.org/numpy-searchsorted-in-python/ 
		K = np.searchsorted(variance_ratio, retained_variance) + 1
		
		return self.transform(data, K)

	def get_V(self) ->np.ndarray:
		"""		
		Getter function for value of V
		"""
		return self.V

	def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
		"""		
		You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
		For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
		You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using matplotlib.
		
		Args:
			xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
			ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
		self.fit(X)
		two_d = self.transform(X, K=2)

		plt.figure()
		plt.scatter(two_d[:, 0], two_d[:, 1], c=y, cmap='viridis')
		plt.title(f"{fig_title} : PCA 2-Dimensional")
		plt.xlabel("PC1")
		plt.ylabel("PC2")
		plt.grid(True)
		plt.show()

		three_d = self.transform(X, K=3)
		fig = plt.figure()
		# Referred to https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html on converting to 3D
		plot_3d = fig.add_subplot(projection='3d')
		scatter = plot_3d.scatter(three_d[:, 0], three_d[:, 1], three_d[:, 2], c=y, cmap='viridis')
		plot_3d.set_title(f"{fig_title} : PCA 3D")
		plot_3d.set_xlabel("PC1")
		plot_3d.set_ylabel("PC2")
		plot_3d.set_zlabel("PC3")
		plt.show()

		plt.figure()
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
		plt.title(f"{fig_title} : Original Data")
		plt.xlabel("Feature 0")
		plt.ylabel("Feature 1")
		plt.grid(True)
		plt.show()

