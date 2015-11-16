import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color.colorconv import rgb2gray

from do import shakti


def compute_laplacian(I):
    if len(I.shape) != 2 or I.dtype != np.float32:
        raise RuntimeError

    L = np.empty(I.shape, dtype=np.float32)
    shakti.compute_laplacian(L, I)
    return L

I = imread('/home/david/Dropbox/wallpapers/Kingfisher.jpg')
I = rgb2gray(I).astype(np.float32)
L = compute_laplacian(I)

plt.imshow(L, interpolation='nearest', cmap=plt.get_cmap('gray'))
plt.show()
