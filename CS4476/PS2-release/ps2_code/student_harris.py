import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used later (in get_interest_points for calculating
    image gradients and a second moment matrix).
    You can call this function to get the 2D gaussian filter.

    Hints:
    1) Make sure the value sum to 1
    2) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """

    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################

    one_d = cv2.getGaussianKernel(ksize, sigma)
    kernel = one_d @ one_d.T
    kernel = kernel / kernel.sum()
    kernel = kernel.astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return kernel

def my_filter2D(image, filter, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Hints:
        Padding width should be half of the filter's shape (correspondingly)
        The conv_image shape should be same as the input image
        Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c),
                depending if image is grayscale or colored
    -   filter: filter that will be used in the convolution with shape (a,b)
    -   bias: An bias value added to every output

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################

    img = image.astype(np.float64)
    k = np.flip(filter.astype(np.float64), (0, 1))
    H, W = img.shape[:2]
    kh, kw = k.shape
    pad_h = kh // 2
    pad_w = kw // 2

    if img.ndim == 2:
        # Referred to https://www.geeksforgeeks.org/python/python-opencv-cv2-copymakeborder-method/
        pad_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
        conv_image = np.zeros((H, W), dtype=np.float64)
        for i in range(H):
            for j in range(W):
                window = pad_img[i:i+kh, j:j+kw]
                conv_image[i, j] = (window * k).sum() + bias

    else:
        C = img.shape[2]
        pad_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
        conv_image = np.zeros((H, W, C), dtype=np.float64)
        for i in range(H):
            for j in range(W):
                window = pad_img[i:i+kh, j:j+kw, :]
                for c in range(C):
                    conv_image[i, j, c] = (window[:, :, c] * k).sum() + bias
    
    conv_image = conv_image.astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter,
    which is of shape (3, 3). Sobel filters can be used to approximate the image
    gradient, and it will be a different filter for the x and y directions.

    Helpful functions: my_filter2D from above

    Args:
    -   image: A numpy array of shape (m,n) containing the image

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction

    Note: Remember that the image gradient in the x-direction corresponds to vertical edge detection and vice versa for y.
    """

    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    img = image.astype(np.float64)
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float64)
    
    ix = my_filter2D(img, sobel_x)
    iy = my_filter2D(img, sobel_y)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy array of shape (N,)
    -   y: numpy array of shape (N,)
    -   c: numpy array of shape (N,)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        Set this to 16 for unit testing. Treat the center point of this window as the bottom right
        of the center-most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-#removed vals,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-#removed vals,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N-#removed vals,) containing the strength
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    H, W = image.shape[:2]
    half = window_size // 2

    y_ok = (y - half >= 0) & (y + half - 1 < H)
    x_ok = (x - half >= 0) & (x + half - 1 < W)
    keep = y_ok & x_ok

    x = x[keep]
    y = y[keep]
    if c is not None:
        c = c[keep]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.
    Second moments, AKA the variance, provide a measure of how spread out the values are in a distribution.
    These moments are computed by convolving the image gradients with a Gaussian filter.

    Helpful functions: my_filter2D, get_gaussian_kernel

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    ix = ix.astype(np.float64)
    iy = iy.astype(np.float64)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    gaussian = get_gaussian_kernel(ksize, sigma)
    
    sx2 = my_filter2D(ix2, gaussian).astype(np.float32)
    sy2 = my_filter2D(iy2, gaussian).astype(np.float32)
    sxsy = my_filter2D(ixy, gaussian).astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments function below, calculate corner resposne.

    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    sx2 = sx2.astype(np.float64)
    sy2 = sy2.astype(np.float64)
    sxsy = sxsy.astype(np.float64)

    detM = sx2 * sy2 - (sxsy ** 2)
    traceM = sx2 + sy2
    R = detM - alpha * (traceM ** 2)
    R = R.astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression.
    Take a matrix and return a matrix of the same size but only the max values in a neighborhood that are not zero.
    We also do not want very small local maxima so remove all values that are below the median for the original R matrix.

    The input to this function is corner response matrix and the output is a filtered version of this
    matrix, where some of the responses have been set to 0.

    Helpful functions: scipy.ndimage.filters.maximum_filter

    Args:
    -   R: numpy nd-array of shape (m, n)
    -   neighborhood_size: int, the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero
    """

    R_local_pts = None

    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    R = R.astype(np.float64)
    # Referred to https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html
    R_max = maximum_filter(R, size=neighborhood_size, mode='nearest')
    local_max_mask = (R == R_max)
    threshold = np.median(R)
    strong_mask = (R >= threshold)
    keep = local_max_mask & strong_mask & (R != 0)
    R_local_pts = np.zeros_like(R, dtype=np.float32)
    R_local_pts[keep] = R[keep].astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts


def get_interest_points(image, n_pts = 1500):
    """
    Using your helper functions above, implement the Harris corner detector
    (See Szeliski 4.1.1). You will calculate the image gradients and second moments,
    use these to determine pixels with high corner response, and filter them via
    non maximum suppression and removing border values. You should return the
    top n_pts based on confidence score.

    Helpful functions:
        get_gradients, second_moments, corner_response, non_max_suppression, remove_border_vals

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer, number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """

    x, y, R_local_pts, confidences = None, None, None, None

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################

    if image.ndim == 3:
        img = image.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        gray = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    ix, iy = get_gradients(gray)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R = corner_response(sx2, sy2, sxsy, alpha=0.05)
    R_local_pts = non_max_suppression(R, neighborhood_size=7)
    ys, xs = np.nonzero(R_local_pts)
    confidences = R_local_pts[ys, xs]
    xs, ys, confidences = remove_border_vals(gray, xs, ys, confidences, window_size=16)
    if confidences.size == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                R_local_pts,
                np.array([], dtype=np.float32))

    order = np.argsort(-confidences)
    k = min(n_pts, confidences.size)
    sel = order[:k]
    x = xs[sel].astype(int)
    y = ys[sel].astype(int)
    confidences = confidences[sel].astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return x,y, R_local_pts, confidences
