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

def freeman_encode(img):
    temp = img.copy()
    code = ""
    car = []
    
    car.append(findFirst(img))
    
                
def findFirst(img):
    min = minutia_extraction(img)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i][j] == 255:
                for m in min:
                    if (i, j) != (m[0], m[1]):
                        print(i, j, m[0], m[1])
                        return i, j
            
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
        im_skeleton_color[m[0]][m[1]] = (255, 0, 0)
    return im_skeleton_color

