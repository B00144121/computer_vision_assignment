import numpy as np

def dilation(binary_img):
    rows, cols = binary_img.shape
    output = np.zeros_like(binary_img, dtype=np.uint8)


import numpy as np

def dilation(binary_img):
    rows, cols = binary_img.shape
    output = np.zeros_like(binary_img, dtype=np.uint8)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if np.any(binary_img[i-1:i+2, j-1:j+2] == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


def erosion(binary_img):
    rows, cols = binary_img.shape
    output = np.zeros_like(binary_img, dtype=np.uint8)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if np.all(binary_img[i-1:i+2, j-1:j+2] == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output

def closing(binary_img):
    dilated = dilation(binary_img)
    closed = erosion(dilated)
    return closed

