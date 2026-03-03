import numpy as np

def compute_historgram(gray):
    hist = np.zeros(256, dtype=np.int64)
    pixels = gray.flatten()


    for v in pixels:
        hist[int(v)] += 1




    return hist