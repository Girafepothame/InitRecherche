import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import os
import glob


def load(file):
    res = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return res

def erosion_image(image, structural_elem = 'None'):   
  if structural_elem != 'None':
    return cv2.erosion(image, structural_elem)
  else:
    return cv2.erosion(image)

def dilation_image(image, structural_elem = 'None'):  
  if structural_elem != 'None':
    return cv2.dilation(image, structural_elem)
  else:
    return cv2.dilation(image)

def opening_image(image, structural_elem = 'None'):
  if structural_elem != 'None':
    return cv2.opening(image, structural_elem)
  else:
    return cv2.opening(image)

def closing_image(image, structural_elem = 'None'):
  if structural_elem != 'None':
    return cv2.closing(image, structural_elem)
  else:
    return cv2.closing(image)

