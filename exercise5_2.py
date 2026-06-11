# Exercise 5.1
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import numpy as np
import cv2
import scipy.ndimage as ndim
from typing import Tuple

def edge_detector_1d(np_img: np.ndarray, dm: np.ndarray, dn: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert all(isinstance(arr, np.ndarray) for arr in (np_img, dm, dn))
    fm = ndim.correlate1d(np_img, dm, axis=1)
    fn = ndim.correlate1d(np_img, dn, axis=0)
    magnitude = (fm**2 + fn**2)**0.5
    phase = np.arctan2(fn, fm)
    return fm, fn, magnitude, phase

gray_img = np.array(
    Image.open("chessboard_angled.png", formats=("PNG", )).convert("L"),
    "uint8"
)

dm1 = np.array([-1, 1, 0])
dn1 = np.array([-1, 1, 0])
dm2 = np.array([-1, 0, 1])
dn2 = np.array([-1, 0, 1])
dm3 = np.array([0, -1, 1])
dn3 = np.array([0, -1, 1])


fm1, fn1, magnitude1, phase1 = edge_detector_1d(gray_img, dm1, dn1)
fm2, fn2, magnitude2, phase2 = edge_detector_1d(gray_img, dm2, dn2)
fm3, fn3, magnitude3, phase3 = edge_detector_1d(gray_img, dm3, dn3)

plt.subplot(5, 3, 1), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Gradient 1")
plt.subplot(5, 3, 2), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Gradient 2")
plt.subplot(5, 3, 3), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Gradient 3")
plt.subplot(5, 3, 4), plt.axis("off")
plt.imshow(fm1, "gray")
plt.subplot(5, 3, 5), plt.title("Fm vertical edges"), plt.axis("off")
plt.imshow(fm2, "gray")
plt.subplot(5, 3, 6), plt.axis("off")
plt.imshow(fm3, "gray")
plt.subplot(5, 3, 7), plt.axis("off")
plt.imshow(fn1, "gray")
plt.subplot(5, 3, 8), plt.title("Fn horizontal edges"), plt.axis("off")
plt.imshow(fn2, "gray")
plt.subplot(5, 3, 9), plt.axis("off")
plt.imshow(fn3, "gray")
plt.subplot(5, 3, 10), plt.axis("off")
plt.imshow(magnitude1, "gray")
plt.subplot(5, 3, 11), plt.title("Absolute Value"), plt.axis("off")
plt.imshow(magnitude2, "gray")
plt.subplot(5, 3, 12), plt.axis("off")
plt.imshow(magnitude3, "gray")
plt.subplot(5, 3, 13), plt.axis("off")
plt.imshow(phase1, "gray")
plt.subplot(5, 3, 14), plt.title("Phase"), plt.axis("off")
plt.imshow(phase2, "gray")
plt.subplot(5, 3, 15), plt.axis("off")
plt.imshow(phase3, "gray")

plt.tight_layout()
plt.savefig("filtered.png")