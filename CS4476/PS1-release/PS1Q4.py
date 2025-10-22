import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        self.indoor = None
        self.outdoor = None
        ###### START CODE HERE ######
        self.indoor = io.imread('indoor.png').astype(np.uint8)
        self.outdoor = io.imread('outdoor.png').astype(np.uint8)
        ###### END CODE HERE ######

    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately for both the indoor and outdoor image.
           Use the "gray" colormap options for plotting each channel."""
        
        ###### START CODE HERE ######
        images = [(self.indoor, "Indoor"), (self.outdoor, "Outdoor")]
        fig, axes = plt.subplots(2, 6, figsize=(14, 6))
        for row, (image, label) in enumerate(images):
            if image.ndim == 3 and image.shape[2] == 4:
                rgb_float = color.rgba2rgb(image.astype(np.float32) / 255.0)
                image = (rgb_float * 255.0).astype(np.uint8)
            for i, channel in enumerate(["R", "G", "B"]):
                axes[row, i].imshow(image[:, :, i], cmap='gray')
                axes[row, i].set_title(f"{label} {channel}")
                axes[row, i].axis("off")
            lab_image = color.rgb2lab(image)
            lab_channels = ["L", "A", "B"]
            for i, channel in enumerate(lab_channels, start=3):
                channel_data = lab_image[:, :, i - 3]
                normalized = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-12)
                axes[row, i].imshow(normalized, cmap='gray')
                axes[row, i].set_title(f"{label} {channel}")
                axes[row, i].axis("off")
        plt.tight_layout()
        plt.show()
        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Plot the HSV image.
        Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image of size H x W x 3 with floating point values lying between 0 - 1 in each channel)
        """
        
        HSV = None
        ###### START CODE HERE ######
        rgb = io.imread('inputPS1Q4.jpg')
        rgb = rgb.astype(np.float64) / 255.0
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # Referred to https://www.w3resource.com/python-exercises/numpy/use-np-dot-maximum-dot-reduce-to-find-maximum-element-along-an-axis-in-numpy.php
        V = np.maximum.reduce([R, G, B])
        m = np.minimum.reduce([R, G, B])
        C = V - m
        # Referred to w3resource.com/numpy/array-creation/zeros_like.php
        S = np.zeros_like(V)
        nonzero_v = V > 1e-12
        S[nonzero_v] = C[nonzero_v] / V[nonzero_v]

        H = np.zeros_like(V)
        nonzero_c = C > 1e-12
        r_max = (V == R) & nonzero_c
        g_max = (V == G) & nonzero_c
        b_max = (V == B) & nonzero_c

        H_prime = np.zeros_like(V)
        H_prime[r_max] = (G[r_max] - B[r_max]) / C[r_max]
        H_prime[g_max] = (B[g_max] - R[g_max]) / C[g_max] + 2.0
        H_prime[b_max] = (R[b_max] - G[b_max]) / C[b_max] + 4.0
        H = (H_prime / 6.0) % 1.0
        HSV = np.dstack([H, S, V])

        plt.figure()
        plt.imshow(HSV)
        plt.show()
        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()





