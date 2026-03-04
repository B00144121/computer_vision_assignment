import cv2 as cv
import numpy as np
import time

from src.thresholding import otsu_threshold, apply_threshold
from src.morphology import closing


image_paths = [
    
    "images/Oring11.jpg"
]

#loops through each image path
for path in image_paths:

    #reads images into memory
    img = cv.imread(path)

    if img is None:
        print("Error: Could not read image at path: " + path)
        continue

    #convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    start = time.perf_counter()

    

    t = otsu_threshold(gray)

    binary = apply_threshold(gray, t)
    close =closing(binary)

    end = time.perf_counter()

    print(path, "threshold:", t)
    print(path, "threshold:", t, "|time:", f"{(end-start): .4f} seconds")

    cv.imshow("Original", img)
    cv.imshow("Grayscale", gray)
    cv.imshow("Binary Original", binary)
    cv.imshow("Binary Closed", close)
    


    

    

#Put classification result on screen

cv.waitKey(0)
cv.destroyAllWindows()