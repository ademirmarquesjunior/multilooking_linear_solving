# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:47:16 2021

@author: adeju
"""


import numpy as np
from sympy import symbols
# from scipy import linalg
from scipy import optimize
import rasterio as rt
import math
from os import listdir
import matplotlib.pyplot as plt
from time import time
import cv2



def pixel_positions(ref_index, index_array, num_images):
    # uses array of indexes with the same size of the image depth (# of bands)
    # return image indexes and pixels positions offset

    image_index = np.zeros((num_images), dtype=np.uint8)
    offset_i = np.zeros((num_images), dtype=np.uint8)
    offset_j = np.zeros((num_images), dtype=np.uint8)

    squared_size = int(np.sqrt(num_images))

    position = np.where(np.reshape(list(range(num_images)),
                                   (squared_size, squared_size)) == ref_index)
    x, y = position[0][0], position[1][0]

    index = 0
    for i in range(squared_size):
        for j in range(squared_size):

            image_index[index] = index_array[x + i, y + j]

            if x + i > squared_size:
                offset_i[index] = 1
            else:
                offset_i[index] = 0

            if y + i > squared_size:
                offset_j[index] = 1
            else:
                offset_j[index] = 0

            index = index + 1

    return image_index, offset_i, offset_j


def blit(dest, src, loc):
    # Paste array values into other larger array into determined position
    # https://stackoverflow.com/questions/28676187/numpy-blit-copy-part-of-an-array-to-another-one-with-a-different-size
    pos = [i if i >= 0 else None for i in loc]
    neg = [-i if i < 0 else None for i in loc]
    target = dest[[slice(i, None) for i in pos]]
    src = src[[slice(i, j) for i, j in zip(neg, target.shape)]]
    target[[slice(None, i) for i in src.shape]] = src
    return dest


def lin_sis_solve(im):
    # im = image[image_index, 0+offset_i, 0+offset_j]
    # im  = np.zeros(num_images)
    
    squared_size = int(np.sqrt(np.size(im)))

    num_variables = int(math.pow((squared_size + squared_size - 1), 2))
    p = symbols('p0:%d' % num_variables)

    # Create index array helper
    im_index_array = np.reshape(list(range(np.size(im))), (squared_size, squared_size))
    p_index_array = np.reshape(list(range(np.size(p))), (squared_size + squared_size - 1, squared_size + squared_size -1))

    # Create list of unique image combinations
    im_combinations = []
    for i in range(num_images):
        for j in range(num_images):
            if str(np.sort((i, j))) not in im_combinations:
                im_combinations.append(str(np.sort((i, j))))
                
    

    matrix = []
    for combination in im_combinations:

        im_indexes = np.fromstring(combination[1:-1], sep=' ').astype(int)

        # Recover im_indexes position in im_index_array
        x, y = np.where(im_index_array == im_indexes[0])
        positive_variable_indexes = []
        for i in range(squared_size):
            for j in range(squared_size):
                positive_variable_indexes.append(p_index_array[x + i, y + j][0])

        # Create the 
        negative_variable_indexes = []

        # if the indexes are the same the second list must be empty
        if im_indexes[0] != im_indexes[1]:
            x, y = np.where(im_index_array == im_indexes[1])
            for i in range(squared_size):
                for j in range(squared_size):
                    negative_variable_indexes.append(p_index_array[x + i, y + j][0])

        # Create list with unique values between variable lists
        A = [i for i in positive_variable_indexes if i not in negative_variable_indexes]
        B = [i for i in negative_variable_indexes if i not in positive_variable_indexes]

        row_variables = np.zeros(num_variables + 1, np.float64)

        for i in A:
            row_variables[i] = 1

        for i in B:
            row_variables[i] = -1

        if im_indexes[0] == im_indexes[1]:
            row_variables[-1] = (im[im_indexes[0]]*num_images)
        else:
            row_variables[-1] = ((im[im_indexes[0]] - im[im_indexes[1]])*num_images)

        matrix.append(row_variables)
        
    matrix = np.asarray(matrix)
    
    a = matrix[:, 0:-1]
    b = matrix[:, -1]

    x = optimize.lsq_linear(a, b, bounds=(0, 255))
    
    # linalg.solve(a, b)    
    # x = linalg.lstsq(a, b, check_finite=False)[0]
    # x = solve_linear_system(Matrix(np.asarray(matrix)), *p, dict=True, rational=False)

    return x['x']


def subpixel_offset(num_images, ref_index, squared_size):
    position = np.where(np.reshape(list(range(num_images)),
                                   (squared_size, squared_size)) == ref_index)
    x, y = position[0][0], position[1][0]

    x = x*(1/np.sqrt(num_images))
    y = y*(1/np.sqrt(num_images))

    return x, y

def rescale(data, range=(0,1)):
  return np.interp(data, (data.min(), data.max()), range)


def multilooking(num_images, folder, bands_list):

    arr = listdir(folder)
    
    print(arr)
    
    # Create index array helper
    index_array = np.reshape(list(range(9)), (3, 3))
    index_array = np.concatenate((index_array, index_array[:, 0:2]), axis=1)
    index_array = np.concatenate((index_array, index_array[0:2, :]), axis=0)
    
    squared_size = int(np.sqrt(num_images))
    num_variables = int(math.pow((squared_size + squared_size - 1), 2))
    

    try:
        if isinstance(bands_list, list):
            dataset = rt.open(folder+"//"+arr[0])
            image_template = dataset.read()
            bands_list = np.uint(bands_list)
        else:
            dataset = rt.open(folder+"//"+arr[0])
            image_template = dataset.read()
            # image = np.moveaxis(image, 0, -1)
            bands_list = list(range(np.shape(image_template)[2]))
    except Exception as e:
        print(e)
        
    
    
    n_bands = np.shape(image_template)[0]
    input_width = np.shape(image_template)[1]
    input_height = np.shape(image_template)[2]
    image = np.zeros((num_images, n_bands, input_width, input_height))
    for i in range(np.size(arr)):
        image[i, :, :, :] = rt.open(folder+"//"+arr[i]).read()
    
    image_out = np.empty((n_bands, 3*input_width, 3*input_height), dtype=np.float64)
    image_out[:] = np.nan
    
    start = time()

    for b in bands_list:
        # b = 1
       
        band_out = np.zeros((9, 3*input_width , 3*input_height), dtype=np.float64)
        band_out[:] = np.nan

        for i in range(0, input_width-1):
            for j in range(0, input_height-1):

                 # t = 0
                 # i = 0
                 # j = 0
                 
                

                for t in range(0, num_images):
                    image_index, offset_i, offset_j = pixel_positions(
                        t, index_array, num_images)

                    im = image[image_index, b, i+offset_i, j+offset_j]

                    X = lin_sis_solve(im)

                    # Reshape 
                    X = np.reshape(X, (int(np.sqrt(num_variables)), int(np.sqrt(num_variables))))

                    offset = subpixel_offset(num_images, t, squared_size)

                    x = int(round(squared_size*i + offset[0]*squared_size))
                    y = int(round(squared_size*j + offset[1]*squared_size))
                    print(t, x, y)

                    blit(band_out[t, :, :], X, (x, y))
                    
        
                    
        image_out[b] = np.nanmedian(band_out, axis=0)

        plt.figure(4)
        plt.imshow(np.nanmedian(band_out, axis=0), cmap='gray')
        
        plt.figure(2)
        plt.imshow(cv2.medianBlur(np.uint8(np.nanmedian(band_out, axis=0)),3), cmap='gray')
        
        plt.figure(0)
        plt.imshow(image[0, 0, :, :], cmap='gray')
        
    end = time()
    print(end-start)
        
    return image_out
        
        
# folder = "C://Users//adeju//Coding//Multilooking_Zanotta//examples//exp3_real//dados_teste_1"
folder = "C://Users//adeju//Coding//Multilooking_Zanotta//examples//imagens_cameraman_128"
folder = "C://Users//adeju//Coding//Multilooking_Zanotta//examples//mineral"

num_images = 9
bands_list = [0]

image_predicted = multilooking(num_images, folder, bands_list)

plt.imshow(np.reshape(image_predicted, (384, 384)))


from skimage.filters import median
from skimage.morphology import disk, rectangle

med = median(np.reshape(image_predicted, (63, 63)), rectangle(2,2))
plt.figure(3)
plt.imshow(med, cmap='gray')






