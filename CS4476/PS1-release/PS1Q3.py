import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = None
        ###### START CODE HERE ######
        # Referred to https://scikit-image.org/docs/0.25.x/api/skimage.io.html
        self.img = io.imread('inputPS1Q3.jpg')
        self.img = self.img.astype(np.uint8)
        ###### END CODE HERE ######
        
    def prob_3_1(self):
        """
        Swap red and green color channels here. Plot and return swapImg
        Returns:
            swapImg: RGB image with R and G channels swapped (3 channel image of size H x W x 3 with integer values lying in 0 - 255)
        """

        swapImg = None
        ###### START CODE HERE ######
        # Referred to https://numpy.org/doc/2.2/reference/generated/numpy.copy.html
        swapImg = self.img.copy()
        swapImg[:, :, [0, 1]] = swapImg[:, :, [1, 0]]
        plt.figure()
        plt.imshow(swapImg)
        plt.show()
        ###### END CODE HERE ######
        return swapImg


    def rgb2gray(self, rgb):
        """
        Converts and RGB image to a grayscale image. Input is the RGB image (rgb) and you must return the grayscale image as gray.
        Returns:
            gray: grayscale image (single channel image of size H x W)
        """
        gray = None
        ###### START CODE HERE ######
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        gray = gray.astype(np.uint8)    
        ###### END CODE HERE ######
        return gray

    
    def prob_3_2(self):
        """
        This function should call your rgb2gray function to convert the input image to grayscale. Plot and return grayImg.
        Returns:
            grayImg: grayscale image (single channel image of size H x W with integer values lying between 0 - 255)
        """
        grayImg = None
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img)
        plt.figure()
        plt.imshow(grayImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert the grayscale image to its negative. Plot and return negativeImg.
        
        Returns:
            negativeImg: negative image (single channel image of size H x W with integer values lying between 0 - 255)
        """
        negativeImg = None
        ###### START CODE HERE ######
        gray = self.rgb2gray(self.img)
        negativeImg = 255.0 - gray.astype(np.float32)
        # Referred to https://numpy.org/doc/2.2/reference/generated/numpy.clip.html
        negativeImg = np.clip(negativeImg, 0, 255).astype(np.uint8)
        plt.figure()
        plt.imshow(negativeImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of grayscale image here. Plot and return mirrorImg.
        
        Returns:
            mirrorImg: mirror image (single channel image of size H x W with integer values lying between 0 - 255)
        """
        mirrorImg = None
        ###### START CODE HERE ######
        gray = self.rgb2gray(self.img)
        # Referred to https://numpy.org/doc/2.2/reference/generated/numpy.flip.html
        mirrorImg = np.flip(gray, axis=1)
        plt.figure()
        plt.imshow(mirrorImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here. Plot and return avgImg.
        
        Returns:
            avgImg: average of grayscale and mirror image (single channel image of size H x W with integer values lying between 0 - 255)
        """
        avgImg = None
        ###### START CODE HERE ######
        gray = self.rgb2gray(self.img)
        mirror = np.flip(gray, axis=1)
        avgImg = (gray.astype(np.float32) + mirror.astype(np.float32)) / 2.0
        avgImg = np.clip(avgImg, 0, 255).astype(np.uint8)
        plt.figure()
        plt.imshow(avgImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix with the same size as the grayscale image. Add the noise to the grayscale image, and clip to ensure that max value is 255. 
        Plot this noisy image, and return the noisy image and the noise matrix.
        
        Returns:
            noisyImg: grayscale image after adding noise (single channel image of size H x W with integer values lying between 0 - 255)
            noise: random noise matrix of size H x W
        """
        noisyImg, noise = [None]*2
        ###### START CODE HERE ######
        gray = self.rgb2gray(self.img)
        height, width = gray.shape
        noise = (255 * np.random.rand(height, width)).astype(np.float32)
        noisy = gray.astype(np.float32) + noise
        noisyImg = np.clip(noisy, 0, 255).astype(np.uint8)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        plt.figure()
        plt.imshow(noisyImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()

    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()