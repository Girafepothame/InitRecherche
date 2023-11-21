import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import os
import glob


def load_image(file):
    res = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return res

def invert_image(img):
    return 255-img

def skeletonize_image(img):
    # Invert the image to work with white letters on black and not white areas creating a black letter
    img = invert_image(img)
    size = np.size(img)
    # Create
    skel = np.zeros(img.shape, np.uint8)
    
    # Already binarised images but still needed for other cases
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Cross-pattern kernel 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    done = False
    while not(done):
        # We don't use the openin method because we need the eroded image later
        eroded = cv2.erode(img,kernel)
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
        
    return skel
    
    

# Return all files from "path" directory (default all png files)
def char_paths(path = "dataset/dataset_caracters"):
    return glob.glob(path + "/**/*.png", recursive=True)


def img_tab(tab, car):
    return [load_image(image) for image in tab[car]]
    
    