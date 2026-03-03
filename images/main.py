import cv2 as cv
import numpy as np
import time

from src.thresholding import otsu_threshold, apply_threshold
from src.morphology import close, fill_holes
from src.ccl import connected_components, largest_compnent_mask
from src.analysis import classfy

image_paths = {
    "images/Oring1.jpg",
    "images/Oring2.jpg"

}

#loops through each image path
for path in image_paths:

    #reads images into memory
    img = cv.imread(path)

    #convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Start timing entire pipeline
    start = time.perf_counter()

    t = otsu_threshold(gray)

    binary = apply_threshold(gray, t)

    cleaned = close(binary, k=3)

    cleaned = fill_holes(cleaned)

    labels, stats = connected_components(cleaned)
    #extract largest objects(O ring)
    mask = largest_compnent_mask(labels, stats)

    #Feature Analysis
    result, features = classfy(mask)

    display = (mask * 225).astype(np.unit8)
    display = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

#Put classification result on screen
cv.putText(display,
           f"{result}", 
           (10, 30), 
           cv.FONT_HERSHEY_SIMPLEX, 
           1, 
           (0, 255, 0), 
           2)

cv.imshow("Result", display)
cv.waitKey(0)
cv.destroyAllWindows()