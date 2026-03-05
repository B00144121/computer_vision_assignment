import numpy as np

def connected_components(binary_img):
    rows, cols = binary_img.shape
    labels = np.zeros_like(binary_img, dtype=np.int32)
    label = 1

    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 255 and labels[i, j] == 0:
                flood_fill(binary_img, labels, i, j, label)
                label += 1

    return labels