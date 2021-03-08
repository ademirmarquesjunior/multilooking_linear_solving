# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:47:16 2021

@author: adeju
"""


import numpy as np
import sympy as sym
from sympy import *
import rasterio as rt
import math
from os import listdir


num_images = 9
squared_size = int(np.sqrt(num_images))
image = np.zeros((num_images, 500, 500), dtype=np.uint8)
image_out = np.zeros((num_images, 3*500, 3*500), dtype=np.uint8)

# Create index array helper
index_array = np.reshape(list(range(9)), (3, 3))
index_array = np.concatenate((index_array, index_array[:, 0:2]), axis=1)
index_array = np.concatenate((index_array, index_array[0:2, :]), axis=0)


def pixel_positions(ref_index, index_array, num_images):
    # uses array of indexes with the same size of the image depth (# of bands)
    # return image indexes and pixels positions offset

    image_index = np.zeros((num_images), dtype=np.uint8)
    offset_i = np.zeros((num_images), dtype=np.uint8)
    offset_j = np.zeros((num_images), dtype=np.uint8)

    # squared_size = int(np.sqrt(num_images))

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

    # p = []
    eqn = []

    # for i in range(int(math.pow((squared_size + squared_size - 1), 2))):
    #     p.append(sym.symbols('p' + str(i)))
        
    p = sym.symbols('p0:%d'%int(math.pow((squared_size + squared_size - 1), 2)))

    # Create index array helper
    im_index_array = np.reshape(list(range(np.size(im))), (squared_size, squared_size))
    p_index_array = np.reshape(list(range(np.size(p))), (squared_size + squared_size - 1, squared_size + squared_size -1))

    # Create list of unique image combinations
    im_combinations = []
    for i in range(num_images):
        for j in range(num_images):
            if str(np.sort((i, j))) not in im_combinations:
                im_combinations.append(str(np.sort((i, j))))

    res = []
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

        # Create the equation expression
        for i in A:
            if "expression" not in locals():
                expression = sym.Add(p[i])
            else:
                expression = sym.Add(expression, p[i])

        for i in B:
            if "expression" not in locals():
                expression = sym.Add(-p[i])
            else:
                expression = sym.Add(expression, -p[i])

        if im_indexes[0] == im_indexes[1]:
            result = (im[im_indexes[0]]*num_images)
            res.append(result)
        else:
            result = ((im[im_indexes[0]] - im[im_indexes[1]])*num_images)
            res.append(result)
        eqn.append(sym.Eq(expression, result))
        del(expression)

    # print(eqn[0])

    return solve(eqn, *p)


def subpixel_offset(num_images, ref_index, squared_size):
    position = np.where(np.reshape(list(range(num_images)),
                                   (squared_size, squared_size)) == ref_index)
    x, y = position[0][0], position[1][0]

    x = x*(1/np.sqrt(num_images))
    y = y*(1/np.sqrt(num_images))

    return x, y


def multilooking(num_images, folder, bands_list):

    folder = "C://Users//adeju//Coding//Multilooking_Zanotta//examples//exp3_real//dados_teste_1"
    arr = listdir(folder)

    try:
        if isinstance(bands_list, list):
            dataset = rt.open(arr[0])
            image = dataset.read()
            bands_list = np.uint(bands_list)
        else:
            dataset = rt.open(arr[0])
            image = dataset.read()
            # image = np.moveaxis(image, 0, -1)
            bands_list = list(range(np.shape(image)[2]))
    except Exception as e:
        print(e)

    bands_list = [1]

    for b in bands_list:
        num_images = np.size(arr)
        width = np.shape(image)[1]
        height = np.shape(image)[2]

        image = np.zeros((num_images, width, height))
        for i in range(np.size(arr)):
            image[i, :, :] = rt.open(arr[i]).read(b)

        for i in range(0, width):
            for j in range(0, height):
                
                # t = 0

                for t in range(0, num_images):
    
                    image_index, offset_i, offset_j = pixel_positions(
                        t, index_array, num_images)
    
                    X = lin_sis_solve(image[image_index, i+offset_i, j+offset_j])
                    print(X)
    
                    X = np.reshape(X, (int(np.sqrt(X)), int(np.sqrt(X))))
    
                    # offsets
                    tile_size_offset = int(np.shape(X)[0]/2)
                    tile_size_offset = int(5/2)
                    offset = subpixel_offset(num_images, 0, squared_size)
                    x = int(round(squared_size*(i + offset[0])-tile_size_offset))
                    y = int(round(squared_size*(j + offset[0]))-tile_size_offset)
    
                    blit(image_out[t, :, :], index_array, (x, y))

