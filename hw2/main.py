import cv2
import numpy as np
import matplotlib.pyplot as plt
import math




def get_low_high_psf(cont_scene, alpha = 2.5, k_l_size=23):
    psf_l = cv2.getGaussianKernel(k_l_size, 0)
    psf_l = psf_l * np.transpose(psf_l)
    k_h_size = math.floor(k_l_size / alpha)
    psf_h = np.empty((k_h_size, k_h_size))
    for x, y in np.ndindex(psf_l.shape):
        n_x, n_y = (math.floor(alpha * x), math.floor(alpha * y))
        if n_x >= k_l_size or n_y >= k_l_size or x >= k_h_size or y >= k_h_size:
            continue
        psf_h[x, y] = alpha * psf_l[n_x, n_y]
    plt.imshow(psf_l, interpolation='none')
    plt.show()
    plt.imshow(psf_h, interpolation='none')
    plt.show()


if __name__ == '__main__':
    cont_scene = cv2.imread('DIPSourceHW2.png')
    get_low_high_psf(cont_scene)
