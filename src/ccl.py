import numpy as np

def connected_components(binary_img):
    rows, cols = binary_img.shape
    labels = np.zeros_like(binary_img, dtype=np.int32)

    label = 1
    stats = [None]


    for i in range(rows):
        for j in range(cols):

            if binary_img[i, j] == 255 and labels[i, j] == 0:
             
             stack = [(i, j)]
             labels[i, j] = label

             area = 0
             minx = j
             maxx = j
             miny = i
             maxy = i

             while stack:
                y, x = stack.pop()
                area += 1

                if x < minx: minx = x
                if x > maxx: maxx = x
                if y < miny: miny = y
                if y > maxy: maxy = y

                neighbours = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                for ny, nx in neighbours:
                    if 0 <= ny < rows and 0 <= nx < cols:
                        if binary_img[ny, nx] == 255 and labels[ny, nx] == 0:
                            stack.append((ny, nx))
                            labels[ny, nx] = label

                stats.append((area, minx, miny, maxx, maxy)) 
                label += 1
    return labels, stats
 