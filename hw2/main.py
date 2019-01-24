import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2, fftshift


def construct_psf_h_from_psf_l(alpha, k_l_size, psf_l):
    k_h_size = math.floor(k_l_size / alpha) + 1
    psf_h = np.zeros((k_h_size, k_h_size))
    for x, y in np.ndindex(psf_l.shape):
        n_x, n_y = (math.floor(alpha * x), math.floor(alpha * y))
        if n_x >= k_l_size or n_y >= k_l_size or x >= k_h_size or y >= k_h_size:
            continue
        psf_h[x, y] = alpha * psf_l[n_x, n_y]
    return psf_h


def get_low_high_psf_gaussian(alpha = 2.5, k_l_size=25):
    psf_l = cv2.getGaussianKernel(k_l_size, 0)
    psf_l = psf_l * np.transpose(psf_l)
    psf_h = construct_psf_h_from_psf_l(alpha, k_l_size, psf_l)
    return psf_l, psf_h


def get_low_high_psf_box(alpha=2.5, k_l_size=25):
    psf_l = np.zeros((k_l_size, k_l_size), dtype=np.int)
    psf_l[1:k_l_size-1, 1:k_l_size-1] = 1
    psf_h = construct_psf_h_from_psf_l(alpha, k_l_size, psf_l)
    return psf_l, psf_h


def get_psfs(alpha = 2.5, verbose=False):
    psf_l_g, psf_h_g = get_low_high_psf_gaussian()
    psf_l_b, psf_h_b = get_low_high_psf_box()
    if verbose:
        plt.imshow(psf_l_g, interpolation='none')
        plt.show()
        plt.imshow(psf_h_g, interpolation='none')
        plt.show()
        plt.imshow(psf_l_b, interpolation='none')
        plt.show()
        plt.imshow(psf_h_b, interpolation='none')
        plt.show()
    return psf_l_g, psf_h_g, psf_l_b, psf_h_b


def get_image_by_psf(cont_scene, psf):
    return fftconvolve(cont_scene, psf[:, :, np.newaxis], mode='same')


def normalize(x):
    x -= np.min(x)
    return x / np.max(x)


def find_k(psf_l, psf_h, verbose=False):
    psf_L = fft2(psf_l)
    psf_H = fft2(psf_h, shape=psf_L.shape)
    K = psf_L / psf_H
    k = abs(ifft2(K))
    if verbose:
        plt.imshow(k)
        plt.show()
    return k


def create_images(cont_scene, psfs, verbose=False):
    images = []
    for psf in psfs:
        psf = normalize(psf)
        img = get_image_by_psf(cont_scene, psf)
        img = normalize(img)
        images.append(img)
        if verbose:
            plt.imshow(img)
            plt.show()
    return images


def get_ks(psfs, verbose=False):
    psf_l_g, psf_h_g, psf_l_b, psf_h_b = psfs
    k_g = find_k(psf_l_g, psf_h_g, verbose)
    k_b = find_k(psf_l_b, psf_h_b, verbose)
    return k_g, k_b


if __name__ == '__main__':
    cont_scene = cv2.imread('../DIPSourceHW2.png')
    psfs = get_psfs()
    # images = create_images(cont_scene, psfs)
    psf_l, psf_h = psfs[0], psfs[1]
    ks = get_ks(psfs, verbose=True)

