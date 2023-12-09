import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import os
import glob

from skimage.color import gray2rgb
from skimage.io import imread, imshow, imsave
from skimage.util import invert
from skimage.transform import resize, rotate
from skimage.morphology import erosion, dilation, opening, closing, skeletonize, square
from skimage.filters import threshold_isodata, threshold_li, threshold_mean, threshold_minimum, threshold_otsu, threshold_triangle, threshold_yen


def load_image(file):
    res = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return res

def invert_image(img):
    return 255-img  
    
# Return all files from "path" directory (default all png files)
def char_paths(path = "dataset/dataset_caracters"):
    return glob.glob(path + "/**/*.png", recursive=True)


def img_tab(tab, car):
    return [load_image(image) for image in tab[car]]

def freeman_encode(minutia, img):
    directions = []
    car = []
    
    
            
def minutia_extraction(im_skeleton):
    minutia = []
    h = im_skeleton.shape[0]
    w = im_skeleton.shape[1]
        
    for i in range(1, h-1):
        for j in range(1, w-1):
            # Browsing through the pixels of the skeleton
            if im_skeleton[i][j] !=0:
                # Get every neighbor
                P = [im_skeleton[i][j+1], im_skeleton[i-1][j+1], im_skeleton[i-1][j], im_skeleton[i-1][j-1], im_skeleton[i][j-1], im_skeleton[i+1][j-1], im_skeleton[i+1][j], im_skeleton[i+1][j+1], im_skeleton[i][j+1]]
                CN = 0
                # Pitié ALED
                for k in range(8):
                    CN += abs(P[k]/255 - P[k+1]/255)
                CN = CN/2
                    
                # 0 : Isolated point
                # 1 : Ending point
                # 2 : Connective point
                # 3 : Bifurcation point
                # 4 : Crossing point
                # only consider 0,1,3,4 CN values
                
                if CN==0:
                    minutia.append((i,j,0))
                elif CN == 1:
                    minutia.append((i,j,1))
                elif CN == 3:
                    minutia.append((i,j,3))
                elif CN == 4:
                    minutia.append((i,j,4))
  
    return minutia

# Coloring minutia pixels in red for visualization
def draw_minutia(minutia, im_skeleton):
    im_skeleton_color = gray2rgb(im_skeleton)
    # Looping through the minutias collected
    for m in minutia:
        # Colorize the pixel in red
        im_skeleton_color[m[0]][m[1]] = (255, 0, 0)
    return im_skeleton_color

# Distance between two points of an image
def euclidean_distance_minutia(m1, m2):
    return math.sqrt((m1[0] - m2[0])*(m1[0] - m2[0]) + (m1[1] - m2[1])*(m1[1] - m2[1]))

# delete serif (small parts of character) which size is inferior to the threshold - return the table of the minutia where the code will be generated between
def smoothing(minutia, threshold):
    smooth_minutia = []
    ending_points = []
    smooth_ending_points = []
    pb = []

    # Add all ending points to the right array
    for m in minutia:
        if m[2] != 1:
            smooth_minutia.append(m)
        else:
            ending_points.append(m)

    # Case where only ending points
    if smooth_minutia == []:
        return minutia
    # Else
    else:
        for m in ending_points:
            i = 0
            # Test the length of the *serif* we are analysing (between the current ending point and the next non-ending minutia)
            while (i < len(smooth_minutia)) and (euclidean_distance_minutia(m, smooth_minutia[i]) > threshold):
                i += 1
            if (i == len(smooth_minutia)):
                smooth_ending_points.append(m)
            else:
                pb.append(smooth_minutia[i])
                
    # removing duplicates
    pb = list(set(pb))

    for m in pb:
        smooth_minutia.remove(m)

    return smooth_minutia + smooth_ending_points