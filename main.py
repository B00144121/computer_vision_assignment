import cv2 as cv
import numpy as np
import time

from src.thresholding import otsu_threshold, apply_threshold
from src.morphology import closing
from src.ccl import connected_components, largest_component_mask

image_paths = [
    "images/Oring11.jpg"
]

# loops through each image path
for path in image_paths:

    # reads image into memory
    img = cv.imread(path)

    if img is None:
        print("Error: Could not read image at path: " + path)
        continue

    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    start = time.perf_counter()

    t = otsu_threshold(gray)

    binary = apply_threshold(gray, t)
    closed = closing(binary)

    foreground = 255 - closed
    cv.imshow("Foreground for CCL", foreground)

    labels, stats = connected_components(foreground)
    print("components found:", len(stats) - 1)

    ring_mask = largest_component_mask(labels, stats)
    cv.imshow("Largest Component", ring_mask)

    end = time.perf_counter()

    print(path, "threshold:", t)
    print(path, "threshold:", t, "|time:", f"{(end-start): .4f} seconds")

    cv.imshow("Original", img)
    cv.imshow("Grayscale", gray)
    cv.imshow("Binary Original", binary)
    cv.imshow("Binary Closed", closed)

cv.waitKey(0)
cv.destroyAllWindows()