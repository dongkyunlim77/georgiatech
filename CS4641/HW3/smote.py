from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def euclid_pairwise_dist(x: np.ndarray, y: np.ndarray) ->np.ndarray:
    """
    You implemented this in project 2! We'll give it to you here to save you the copypaste.
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
    """
    x_norm = np.sum(x ** 2, axis=1, keepdims=True)
    yt = y.T
    y_norm = np.sum(yt ** 2, axis=0, keepdims=True)
    dist2 = np.abs(x_norm + y_norm - 2.0 * (x @ yt))
    return np.sqrt(dist2)


def confusion_matrix_vis(conf_matrix: np.ndarray):
    """
    Fancy print of confusion matrix. Just encapsulating some code out of the notebook.
    """
    _, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix)
    ax.set_xlabel('Predicted Labels', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual Labels', fontsize=16)
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, str(val), ha='center', va='center', bbox=dict(
            boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.show()
    return


class SMOTE(object):

    def __init__(self):
        pass

    @staticmethod
    def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray
        ) ->np.ndarray:
        """		
		Generate the confusion matrix for the predicted labels of a classification task.
		This function should be able to process any number of unique labels, not just a binary task.
		
		The choice to put "true" and "predicted" on the left and top respectively is arbitrary.
		In other sources, you may see this format transposed.
		
		Args:
		    y_true: (N,) array of true integer labels for the training points
		    y_pred: (N,) array of predicted integer labels for the training points
		    These vectors correspond along axis 0. y_pred[i] is the prediction for point i, whose true label is y_true[i].
		Returns:
		    conf_matrix: (u, u) array of ints containing instance counts, where u is the number of unique labels present
		        conf_matrix[i,j] = the number of instances where a sample from the true class i was predicted to be in class j
		Hints:
		    You can assume that the labels will be ints of the form [0, u).
		    Thus, you can use these labels as valid indices.
		"""
        raise NotImplementedError

    @staticmethod
    def compute_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray, m: int=100):
        """		
		Calculate the True Positive Rate and False Positive Rate for the classification task.
		
		The threshold for the ROC curve is the value of the prediction at which we consider a point to be positive.
		For each threshold, we can calculate the true positive rate and false positive rate.
		The ROC curve is a plot of the true positive rate against the false positive rate for all thresholds.
		
		The length of thresholds should be m, and the first element should be 0, and the last element should be 1.
		m thresholds ensures that we fully capture model behavior across all the decision points.
		Higher the number of m, more accurate is the value of AUC, but the program may run slow with very high m values.
		
		Thresholds need to be even placed float values which is m sized array to capture all the possible values for
		TPR and FPR and get the value of AUC as accurate as possible.
		
		Since we want to model the Receiver Operating Characteristic (ROC) curve using the output of this function,
		FPR must be sorted in the ascending order. However, in the case of multiple points having the same FPR,
		we must further sort them by TPR in ascending order. This is important because the ROC curve should
		always move upwards when FPR remains constant.
		
		If the FPR values are not sorted correctly, the ROC will be misaligned and the curve will be having an incorrect shape
		which will lead to incorrect AUC value. Additionally, the corresponding TPR and threshold values must be aligned index-wise with
		their respective FPR values.
		
		This will ensure the correct plotting of the ROC curve and the accurate value of AUC.
		
		Args:
		    y_true: (N,) array of true integer labels for the training points
		    y_pred: (N,) array of predicted integer labels for the training points
		    These vectors correspond along axis 0. y_pred[i] is the prediction for point i, whose true label is y_true[i].
		    You can assume that the labels will be ints of the form [0, 1).
		Return:
		    tpr: (m,) array of the true positive rate for each threshold
		    fpr: (m,) array of the false positive rate for each threshold
		    thresholds: (m,) array of the thresholds used to calculate the TPR and FPR
		"""
        raise NotImplementedError

    @staticmethod
    def compute_roc_auc(tpr: np.ndarray, fpr: np.ndarray):
        """		
		Calculate the Area under the Receiver Operation Curve using the TPR and FPR values
		The ROC AUC is a measure of how well the classifier can separate the true positive rate from the false positive rate.
		Args:
		    tpr: (M,) array of the true positive rate
		    fpr: (M,) array of the false positive rate
		Return:
		    auc: float value of the Area under the Receiver Operation Curve
		"""
        raise NotImplementedError

    @staticmethod
    def plot_roc_auc(roc_auc: float, tpr, fpr, thresholds):
        """
        Plots the ROC curve and the operation curve for the given data.
        Args:
            roc_auc: (float) the area under the ROC curve
            tpr: (M,) array of the true positive rate for each threshold
            fpr: (M,) array of the false positive rate for each threshold
            thresholds: (M,) array of the thresholds used to calculate the TPR
        Return:
            Display the ROC and Operation Curve plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(fpr, tpr, marker='o', linestyle='-', color='blue', label
            =f'ROC Curve (AUC={roc_auc:.2f})')
        ax[0].fill_between(fpr, tpr, color='blue', alpha=0.2)
        ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax[0].set_xlabel('False Positive Rate (FPR)')
        ax[0].set_ylabel('True Positive Rate (TPR)')
        ax[0].set_title('Receiver Operating Characteristic (ROC) Curve')
        ax[0].legend()
        ax[1].scatter(thresholds, tpr, marker='o', color='green', label=
            'TPR (Sensitivity)')
        ax[1].scatter(thresholds, fpr, marker='o', color='red', label=
            'FPR (False Positive Rate)')
        ax[1].set_xlabel('Decision Threshold')
        ax[1].set_ylabel('Rate')
        ax[1].set_title('Operation Curve (TPR & FPR vs. Threshold)')
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def interpolate(start: np.ndarray, end: np.ndarray, inter_coeff: float
        ) ->np.ndarray:
        """		
		Return an interpolated point along the line segment between start and end.
		
		Hint:
		    if inter_coeff==0.0, this should return start;
		    if inter_coeff==1.0, this should return end;
		    if inter_coeff==0.5, this should return the midpoint between them;
		    to generalize this behavior, try writing this out in terms of vector addition and subtraction
		Args:
		    start: (D,) float array containing the start point
		    end: (D,) float array containing the end point
		    inter_coeff: (float) in [0,1] determining how far along the line segment the synthetic point should lie
		Return:
		    interpolated: (D,) float array containing the new synthetic point along the line segment
		"""
        raise NotImplementedError

    @staticmethod
    def k_nearest_neighbors(points: np.ndarray, k: int) ->np.ndarray:
        """		
		For each point, retrieve the indices of the k other points which are closest to that point.
		
		Hints:
		    Find the pairwise distances using the provided function: euclid_pairwise_dist.
		    For execution time, try to avoid looping over N, and use numpy vectorization to sort through the distances and find the relevant indices.
		Args:
		    points: (N, D) float array of points
		    k: (int) describing the number of neighbor indices to return
		Return:
		    neighborhoods: (N, k) int array containing the indices of the nearest neighbors for each point
		        neighborhoods[i, :] should be a k long 1darray containing the neighborhood of points[i]
		        neighborhoods[i, 0] = j, such that points[j] is the closest point to points[i]
		"""
        raise NotImplementedError

    @staticmethod
    def smote(X: np.ndarray, y: np.ndarray, k: int, inter_coeff_range:
        Tuple[float]) ->np.ndarray:
        """		
		Perform SMOTE on the binary classification problem (X, y), generating synthetic minority points from the minority set.
		In 6.1, we did work for an arbitrary number of classes. Here, you can assume that our problem is binary, that y will only contain 0 or 1.
		
		Outline:
		    # 1. Determine how many synthetic points are needed from which class label.
		    # 2. Get the subset of the minority points.
		    # 3. For each minority point, determine its neighborhoods. (call k_nearest_neighbors)
		    # 4. Generate |maj|-|min| synthetic data points from that subset.
		        # a. uniformly pick a random point as the start point
		        # b. uniformly pick a random neighbor as the endpoint
		        # c. uniformly pick a random interpolation coefficient from the provided range: `inter_coeff_range`
		        # d. interpolate and add to output (call interpolate)
		    # 5. Generate the class labels for these new points.
		Args:
		    X: (|maj|+|min|, D) float array of points, containing both majority and minority points; corresponds index-wise to y
		    y: (|maj|+|min|,) int array of class labels, such that y[i] is the class of X[i, :]
		    k: (int) determines the size of the neighborhood around the sampled point from which to sample the second point
		    inter_coeff_range: (a, b) determines the range from which to uniformly sample the interpolation coefficient
		        Sample U[a, b)
		        You can assume that 0 <= a < b <= 1
		Return:
		    A tuple containing:
		        - synthetic_X: (|maj|-|min|, D) float array of new, synthetic points
		        - synthetic_y: (|maj|-|min|,) array of the labels of the new synthetic points
		"""
        raise NotImplementedError
