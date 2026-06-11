# Exercise 5.3
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image
import scipy.ndimage as ndim
from typing import Tuple


def sobel_filter(
    np_img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert isinstance(np_img, np.ndarray)
    fm = ndim.sobel(np_img, axis=1)  #horizontal derivate (detects vertical edges)
    fn = ndim.sobel(np_img, axis=0)  #vertical derivate (detects horizontal edges)
    magnitude = (fm**2 + fn**2)**0.5 
    phase = np.arctan2(fn, fm)
    return fm, fn, magnitude, phase


def edge_detector(
    np_img: np.ndarray, dm: np.ndarray, dn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert all(isinstance(arr, np.ndarray) for arr in (np_img, dm, dn))
    fm = ndim.correlate(np_img, dm)
    fn = ndim.correlate(np_img, dn)
    magnitude = (fm**2 + fn**2)**0.5
    phase = np.arctan2(fn, fm)
    return fm, fn, magnitude, phase


dm = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]
dn =  [[-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1]]
ddm =  [[-2, -1, 0],
      [-1, 0, 1],
      [0, 1, 2]]
ddn =  [[0, -1, -2],
      [1, 0, -1],
      [2, 1, 0]]

gray_img = np.array(Image.open("chessboard_angled.png").convert("L"), "uint8")

fm1, fn1, magnitude, phase = sobel_filter(gray_img)
fm2, fn2, magnitude2, phase2 = edge_detector(gray_img, dm, dn)
fm3, fn3, magnitude3, phase3 = edge_detector(gray_img, ddm, ddn)

plt.subplot(5, 3, 1), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Sobel 1")
plt.subplot(5, 3, 2), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Sobel 2")
plt.subplot(5, 3, 3), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Sobel 3")
plt.subplot(5, 3, 4), plt.axis("off")
plt.imshow(fm1, "gray")
plt.subplot(5, 3, 5), plt.axis("off")
plt.imshow(fm2, "gray"), plt.title("Fm Vertical Edges")
plt.subplot(5, 3, 6), plt.axis("off")
plt.imshow(fm3, "gray")
plt.subplot(5, 3, 7), plt.axis("off")
plt.imshow(fn1, "gray")
plt.subplot(5, 3, 8), plt.axis("off")
plt.imshow(fn2, "gray"), plt.title("Fm Horizontal Edges")
plt.subplot(5, 3, 9), plt.axis("off")
plt.imshow(fn3, "gray")
plt.subplot(5, 3, 10), plt.axis("off")
plt.imshow(magnitude, "gray")
plt.subplot(5, 3, 11), plt.axis("off")
plt.imshow(magnitude2, "gray"), plt.title("Absolute Value")
plt.subplot(5, 3, 12), plt.axis("off")
plt.imshow(magnitude3, "gray")
plt.subplot(5, 3, 13), plt.axis("off")
plt.imshow(phase, "gray")
plt.subplot(5, 3, 14), plt.axis("off")
plt.imshow(phase2, "gray"), plt.title("Phase")
plt.subplot(5, 3, 15), plt.axis("off")
plt.imshow(phase3, "gray")

plt.tight_layout()
plt.savefig("sobel_filtered.png")