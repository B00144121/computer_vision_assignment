import numpy as np

def analyse_ring(ring_mask):

    ring_area = np.sum(ring_mask == 255)

    ys, xs = np.where(ring_mask == 255)

    if len(xs) == 0:
        return "FAIL", 0

    minx = np.min(xs)
    maxx = np.max(xs)
    miny = np.min(ys)
    maxy = np.max(ys)

    bbox_area = (maxx - minx + 1) * (maxy - miny + 1)

    if bbox_area == 0:
        return "FAIL", ring_area

    ratio = ring_area / bbox_area

    if ratio > 0.25:
        result = "PASS"
    else:
        result = "FAIL"

    return result, ratio