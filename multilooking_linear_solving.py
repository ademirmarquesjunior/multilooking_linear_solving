# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:47:16 2021

@author: adeju
"""


import numpy as np

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
    import sympy as sym

    # im = image[image_index, i+offset_i, j+offset_j]

    p = []
    for i in range(25):
        p.append(sym.symbols('p' + str(i)))

    eqn = []

    eqn.append(sym.Eq(p[0] + p[1] + p[2] + p[15] + p[16] + p[17],
                      (im[0]-im[3])*9))
    eqn.append(sym.Eq(p[6] + p[7] + p[8] - p[21] - p[22] - p[23],
                      (im[3] - im[6])*9))
    eqn.append(sym.Eq(p[0] + p[0] + p[1] + p[4] + p[5] + p[6] - p[14] - p[15]
                      - p[16] - p[19] - p[20] - p[21], (im[0] - im[6])*9))
    eqn.append(sym.Eq(p[1] + p[2] + p[3] - p[16] - p[17] - p[18],
                      (im[1] - im[4])*9))
    eqn.append(sym.Eq(p[6] + p[7] + p[8] - p[21] - p[22] - p[23],
                      (im[4] - im[7])*9))
    eqn.append(sym.Eq(p[1] + p[2] + p[3] + p[6] + p[7] + p[8] - p[16] - p[17]
                      - p[18] - p[21] - p[22] - p[23], (im[1] - im[7])*9))
    eqn.append(sym.Eq(p[2] + p[3] + p[4] - p[17] - p[18] - p[19],
                      (im[2] - im[5])*9))
    eqn.append(sym.Eq(p[7] + p[8] + p[9] - p[22] - p[23] - p[24],
                      (im[5] - im[8])*9))
    eqn.append(sym.Eq(p[2] + p[3] + p[4] + p[7] + p[8] + p[9] - p[17] - p[18]
                      - p[19] - p[22] - p[23] - p[24], (im[2] - im[8])*9))
    eqn.append(sym.Eq(p[0] + p[5] + p[10] - p[3] - p[8] - p[13],
                      (im[0] - im[1])*9))
    eqn.append(sym.Eq(p[1] + p[6] + p[11] - p[4] - p[9] - p[14],
                      (im[1] - im[2])*9))
    eqn.append(sym.Eq(p[0] + p[5] + p[10] + p[1] + p[6] + p[11] - p[3] - p[8]
                      - p[13] - p[4] - p[9] - p[14], (im[0] - im[2])*9))
    eqn.append(sym.Eq(p[5] + p[10] + p[15] - p[8] - p[13] - p[18],
                      (im[3] - im[4])*9))
    eqn.append(sym.Eq(p[6] + p[11] + p[16] - p[9] - p[14] - p[19],
                      (im[4] - im[5])*9))
    eqn.append(sym.Eq(p[5] + p[10] + p[15] + p[6] + p[11] + p[16] - p[8]
                      - p[13] - p[18] - p[9] - p[14] - p[19],
                      (im[3] - im[5])*9))
    eqn.append(sym.Eq(p[0] + p[1] + p[2] + p[5] + p[10] - p[8] - p[13] - p[18]
                      - p[17] - p[16], (im[0] - im[4])*9))
    eqn.append(sym.Eq(p[1] + p[2] + p[3] + p[6] + p[11] - p[9] - p[14] - p[19]
                      - p[18] - p[17], (im[1] - im[5])*9))
    eqn.append(sym.Eq(p[5] + p[6] + p[7] + p[10] + p[15] - p[13] - p[18]
                      - p[26] - p[22] - p[21], (im[3] - im[7])*9))
    eqn.append(sym.Eq(p[6] + p[7] + p[8] + p[11] + p[16] - p[14] - p[19]
                      - p[24] - p[23] - p[22], (im[4] - im[8])*9))
    eqn.append(sym.Eq(p[0] + p[1] + p[2] + p[5] + p[10] + p[6] + p[11] - p[8]
                      - p[13] - p[18] - p[17] - p[9] - p[14] - p[19],
                      (im[0] - im[5])*9))
    eqn.append(sym.Eq(p[10] + p[11] + p[15] + p[16] + p[20] + p[21] + p[22]
                      - p[7] - p[8] - p[9] - p[13] - p[14] - p[18] - p[19],
                      (im[6] - im[5])*9))
    eqn.append(sym.Eq(p[5] + p[6] + p[10] + p[11] + p[15] + p[16] + p[17]
                      - p[2] - p[3] - p[4] - p[8] - p[9] - p[13] - p[14],
                      (im[3] - im[2])*9))
    eqn.append(sym.Eq(p[5] + p[6] + p[7] + p[10] + p[11] + p[15] + p[16]
                      - p[13] - p[14] - p[18] - p[19] - p[22] - p[23] - p[24],
                      (im[3] - im[8])*9))
    eqn.append(sym.Eq(p[0] + p[1] + p[2] + p[5] + p[6] + p[7] + p[10] + p[11]
                      - p[13] - p[14] - p[17] - p[18] - p[19] - p[22] - p[23]
                      - p[24], (im[0] - im[8])*9))
    eqn.append(sym.Eq(p[10] + p[11] + p[15] + p[16] + p[17] + p[20] + p[21]
                      + p[22] - p[2] - p[3] - p[4] - p[7] - p[8] - p[9] - p[13]
                      - p[14], (im[6] - im[2])*9))
    eqn.append(sym.Eq(p[0] + p[1] + p[2] + p[5] + p[6] + p[7] + p[10] + p[11]
                      + p[12], im[0]*9))
    eqn.append(sym.Eq(p[1] + p[2] + p[3] + p[6] + p[7] + p[8] + p[11] + p[12]
                      + p[13], im[1]*9))
    eqn.append(sym.Eq(p[2] + p[3] + p[4] + p[7] + p[8] + p[9] + p[12] + p[13]
                      + p[14], im[2]*9))
    eqn.append(sym.Eq(p[5] + p[6] + p[7] + p[10] + p[11] + p[12] + p[15]
                      + p[16] + p[17], im[3]*9))
    eqn.append(sym.Eq(p[6] + p[7] + p[8] + p[11] + p[12] + p[13] + p[16]
                      + p[17] + p[18], im[4]*9))
    eqn.append(sym.Eq(p[7] + p[8] + p[9] + p[12] + p[13] + p[14] + p[17]
                      + p[17] + p[19], im[5]*9))  # Erro de valor repetido
    eqn.append(sym.Eq(p[10] + p[11] + p[12] + p[15] + p[16] + p[17] + p[20]
                      + p[21] + p[22], im[6]*9))
    eqn.append(sym.Eq(p[11] + p[12] + p[13] + p[16] + p[17] + p[18] + p[21]
                      + p[22] + p[23], im[7]*9))
    eqn.append(sym.Eq(p[12] + p[13] + p[14] + p[17] + p[18] + p[19] + p[22]
                      + p[23] + p[24], im[8]*9))

    return sym.solve(eqn, p)


def subpixel_offset(num_images, ref_index, squared_size):
    position = np.where(np.reshape(list(range(num_images)),
                                   (squared_size, squared_size)) == ref_index)
    x, y = position[0][0], position[1][0]

    x = x*(1/np.sqrt(num_images))
    y = y*(1/np.sqrt(num_images))

    return x, y


for i in range(0, np.shape(image)[1]):
    for j in range(0, np.shape(image)[2]):
        for t in range(0, num_images):

            image_index, offset_i, offset_j = pixel_positions(
                t, index_array, num_images)

            X = lin_sis_solve(image[image_index, i+offset_i, j+offset_j])

            X = np.reshape(X, (int(np.sqrt(X)), int(np.sqrt(X))))

            # offsets
            tile_size_offset = int(np.shape(X)[0]/2)
            tile_size_offset = int(5/2)
            offset = subpixel_offset(num_images, 0, squared_size)
            x = int(round(squared_size*(i + offset[0])-tile_size_offset))
            y = int(round(squared_size*(j + offset[0]))-tile_size_offset)

            blit(image_out[t, :, :], index_array, (x, y))

