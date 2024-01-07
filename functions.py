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

PATH = "dataset/dataset_caracters/04_2PS600_police12"

def resize(img, scale):
    """resize an image

    Args:
        img (np array/image): the image you want to resize
        scale (integer): the scale of the image you want

    Returns:
        np array/image: the image resized
    """
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def pre_treatment(img, path):
    """ Edit an image to get a better sample to treat afterward

    Args:
        img (np array/image): the image you want to change
        path (string): Where the image is collected, in this case it's needed to know if we have to scale the image

    Returns:
        np array/mage: the image edited
    """
    if path == "dataset/dataset_caracters/02_PS300_police12":
        img = resize(img, 2)
    
    img = cv2.blur(img,(5,5))
    
    ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
        
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        
    
    # blur = cv2.blur(img,(3,3))
    # ret, img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    
    res = circling_img(img)
    
    return res

def load_image(file):
    """ read an image from a file path

    Args:
        file (string): the path were we will get the file

    Returns:
        np array/image: an image
    """
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = pre_treatment(img, PATH)
    
    img = invert_image(img)
    return img

# invert a binarized image
def invert_image(img):
    return 255-img  
    

def char_paths(path = "dataset/dataset_caracters"):
    """Return all files from "path" directory (default all png files)

    Args:
        path (str, optional): the path were to find the images. Defaults to "dataset/dataset_caracters".

    Returns:
        list: list of paths(str)
    """
    return glob.glob(path + "/**/*.png", recursive=True)


# return an array of images
def img_tab(tab, car):
    return [load_image(image) for image in tab[car]]



# Returnsthe next white pixel neighbouring p, starting with direct neighbours
def get_next(img, p, cache):
    """Iterate in the neighbours of a point to find the right path to foloow

    Args:
        img (np array): the image we are working with
        p (tuple): the current point we are treating
        cache (array/list): a list of the point we already checked -> don't go back and ignore the points we smoothed earlier

    Returns:
        tuple: return the next point we will be treating, or None if there is no neighbours that are not in the cache
    """
    x = p[0]
    y = p[1]
    
    # checking neighbours starting with nearest (cross pattern) then in the corners
    neighbors = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]
    
    for coord in neighbors:
        pt = (coord[0], coord[1])

        if pt is None:
            continue
        # White pixel and not already checked
        if img[pt[0]][pt[1]] == 255 and not pt in cache:
            return pt

    return None



def freeman_travel(img, minutia, pt, cache):
    """Recusrive function to travel through the white pixels of the character

    Args:
        img (np array): the current image
        minutia (array/list): a list of the 'key' points in the shape (ending point, bifucating ...)
        pt (tuple): the current point
        cache (array/list): a list of the point we already checked -> don't go back and ignore the points we smoothed earlier

    Returns:
        _type_: _description_
    """
    # print(minutia)
    point = get_next(img, pt, cache)
    # print(point)
    
    if point is None:
        return []

    cache.append(point)

    # Check if next point is connective (between 2 minutias)
    for minu in minutia:
        # print(minu, point)
        if minu[:2] == point:
            return [point]
        
    elem = freeman_travel(img, minutia, point, cache)
    
    return (elem + [point])


# In case there is no "starting point"
def first_position_value(img, value, size):
    """finding the first point of the shape in case there is no minutias detected

    Args:
        img (np array): the image flattened (transformed into a 1-dim array)
        value (int): value of the pixel
        size (tuple): shape of the image

    Returns:
        tuple: a point referred as 'starting point'
    """
    indexes = np.where(img == value)[0]
    if not np.any(indexes):
        exit

    first = indexes[0]

    x = first // size[1]
    y = first % size[1]
    
    return (x, y, 1)


def encode(code):
    
    """ Encoding an array into Freeman code

    Args:
        code (array or list): An array of tuples that are pixels coordinates

    Returns:
        array or list: An array of "direction" integers -> Freeman encoding
    """
    
    res = ""
    dir = [0, 1, 2,
           7,-1, 3,
           6, 5, 4]
    
    prev_index = -1
    
    # We start with a minutia -> encode 42
    res += ">"
    # print(code)
    
    # Start loop at the second item
    for i in range(1, len(code)):
        prev = code[i-1]
        curr = code[i]
        
        # Retrieving direction vector in the table dir
        x = curr[1] - prev[1]
        y = curr[0] - prev[0]
        
        # If we change starting point (next minutia)
        if abs(x) > 1 or abs(y) > 1:
            res += ">"
            prev_index = -1
            
        else:
            # calculating index from a 2-dim array to a 1-dim array
            index = (y + 1)*3 + x + 1
            
            # if prev_index != index:
            res += str(dir[index])
                
            prev_index = index
                    
    return res



def freeman_encode(skel_img, cache):
    
    """Encode the image by iterating on the remaning minutias after smoothing

    Args:
        skel_img (array): the current image
        cache (list): a list of the point we already checked

    Returns:
        string: The freeman code generated and treated
    """
    
    code = []
    
    minutia = minutia_extraction(skel_img)
    remove, smooth_minutia = smoothing(skel_img, minutia, 15)
    cache += remove
    
    anchor = (0, 0)  # Top-Left Corner
    
    # Create a list of distance between the anchor point and all the minutias
    dist = list(map(lambda x: euclidean_distance_minutia(anchor, x[:2]), smooth_minutia))
    sorted_dist = sorted(dist)
    smooth_minutia = [x[0] for x in sorted(list(zip(smooth_minutia, dist)), key=lambda tupl: tupl[-1])]
    
    
    # Begin from a random point of the letter if there is no minutia (letter 'o' for example)
    if not smooth_minutia:
        flat = skel_img.flatten()
        first = first_position_value(flat, 255, skel_img.shape)
        smooth_minutia.append(first)
    
    for point in smooth_minutia:
        # print("\ncache : " + str(cache))
        # print("new : " + str(point))
        curr_p = point
        cache.append((point[0], point[1]))
        code += freeman_travel(skel_img, smooth_minutia, curr_p, cache)
        
    
    return encode(code)
    
    
            
def minutia_extraction(im_skeleton):
    """Itetrate through an image and retrieve the 'minutias' among the pixels

    Args:
        im_skeleton (np array): the image we are working with

    Returns:
        array/list: List of tuples corresponding to the minutias (x, y, CN)
    """
    minutia = []
    h = im_skeleton.shape[0]
    w = im_skeleton.shape[1]
        
    for i in range(1, h-1):
        for j in range(1, w-1):
            # Browsing through the pixels of the skeleton
            if im_skeleton[i][j] != 0:
                # Get every neighbor
                P = [im_skeleton[i][j+1], im_skeleton[i-1][j+1], im_skeleton[i-1][j],
                     im_skeleton[i-1][j-1], im_skeleton[i][j-1], im_skeleton[i+1][j-1],
                     im_skeleton[i+1][j], im_skeleton[i+1][j+1], im_skeleton[i][j+1]]
                CN = 0
                
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
def draw_minutia(minutia, im_skeleton, color):
    """colorize the wanted pixels of an image

    Args:
        minutia (list/array): the list of pixels we want to colorize
        im_skeleton (np array): current image
        color (tuple): BGR color (B, G, R)

    Returns:
        _type_: _description_
    """
    im_skeleton_color = gray2rgb(im_skeleton)
    ## Looping through the minutias collected
    for m in minutia:
        
        im_skeleton_color[m[0]][m[1]] = color
    return im_skeleton_color



# Distance between two points of an image (pythagore)
def euclidean_distance_minutia(m1, m2):
    return math.sqrt((m1[0] - m2[0])*(m1[0] - m2[0]) + (m1[1] - m2[1])*(m1[1] - m2[1]))



# delete serif (small parts of character) which size is inferior to the threshold - return the table of the minutia where the code will be generated between
def smoothing(skel_img, minutia, threshold):
    """create a table of pixels (coordinates) that we want to ignore afterward by calculating the distance between two minutias

    Args:
        skel_img (np array): current image
        minutia (list): list of tuples
        threshold (int): the threshold under which we ignore a part of image

    Returns:
        2 list: list of the pixel we need to remove, list of the minutias after deleting the bad ones
    """
    smooth_minutia = []
    ending_points = []
    remove = []

    # Add all ending points to the right array
    for m in minutia:
        # every non-ending point
        if m[2] != 1:
            smooth_minutia.append(m)
        else:
            ending_points.append(m)

    # Case where only ending points
    if smooth_minutia == []:
        return remove, minutia
    else:
        for m in ending_points:
            # get distance between the ending points and the first other point encountered (across the shape)
            # the cache is set with the current ending point to prevent going back
            dist = freeman_travel(skel_img, minutia, (m[0], m[1]), [(m[0], m[1])])
            
            if len(dist) < threshold:
                minutia.remove(m)
                remove += dist

    return remove, minutia



# Draw a white border around the edges of the image
def circling_img(img):
    res = np.array([[255]*(img.shape[1]+2)]*(img.shape[0]+2))
    res[1:img.shape[0]+1, 1:img.shape[1]+1] = img
    return res