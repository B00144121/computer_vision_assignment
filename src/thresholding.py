import numpy as np
from src.histogram import compute_historgram

def otsu_threshold(gray):
    hist = compute_historgram(gray)

    total_pixels = gray.shape[0] * gray.shape[1]

    sum_total = 0
    for i in range(256):
        sum_total += i * hist[i]

    sum_background = 0
    weight_background = 0
    weight_foreground = 0

    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    return threshold

def apply_threshold(gray, t):
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray > t] = 255
    return binary