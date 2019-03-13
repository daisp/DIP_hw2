import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2, fftshift
from skimage.restoration import wiener, denoise_tv_bregman, denoise_tv_chambolle
from hw2.sporco.admm.tvl1 import TVL1Deconv
from hw2.sporco.admm.tvl2 import TVL2Deconv


def construct_psf_h_from_psf_l(alpha, k_l_size, psf_l):
    k_h_size = math.floor(k_l_size / alpha) + 1
    psf_h = np.zeros((k_h_size, k_h_size))
    for x, y in np.ndindex(psf_l.shape):
        n_x, n_y = (math.floor(alpha * x), math.floor(alpha * y))
        if n_x >= k_l_size or n_y >= k_l_size or x >= k_h_size or y >= k_h_size:
            continue
        psf_h[x, y] = alpha * psf_l[n_x, n_y]
    return psf_h


def get_low_high_psf_gaussian(alpha, k_l_size=25):
    psf_l = cv2.getGaussianKernel(k_l_size, 0)
    psf_l = psf_l * np.transpose(psf_l)
    psf_h = construct_psf_h_from_psf_l(alpha, k_l_size, psf_l)
    return psf_l, psf_h


def get_low_high_psf_box(alpha, k_l_size=25):
    psf_l = np.zeros((k_l_size, k_l_size), dtype=np.int)
    psf_l[1:k_l_size - 1, 1:k_l_size - 1] = 1
    psf_h = construct_psf_h_from_psf_l(alpha, k_l_size, psf_l)
    return psf_l, psf_h


def get_psfs(alpha=2.5, verbose=False):
    psf_l_g, psf_h_g = get_low_high_psf_gaussian(alpha)
    psf_l_b, psf_h_b = get_low_high_psf_box(alpha)
    if verbose:
        plt.imshow(psf_l_g, interpolation='none', cmap='gray')
        plt.show()
        plt.imshow(psf_h_g, interpolation='none', cmap='gray')
        plt.show()
        plt.imshow(psf_l_b, interpolation='none', cmap='gray')
        plt.show()
        plt.imshow(psf_h_b, interpolation='none', cmap='gray')
        plt.show()
    return psf_l_g, psf_h_g, psf_l_b, psf_h_b


def get_image_by_psf(cont_scene, psf):
    return fftconvolve(cont_scene, psf, mode='same')


def normalize(x):
    x -= np.min(x)
    return x / np.max(x)


def create_images(cont_scene, psfs, verbose=False):
    images = []
    for psf in psfs:
        psf = normalize(psf)
        img = get_image_by_psf(cont_scene, psf)
        img = normalize(img)
        images.append(img)
        if verbose:
            plt.imshow(img, cmap='gray')
            plt.show()
    return images


def find_k(psf_l, psf_h, verbose=False):
    psf_L = fft2(psf_l)
    psf_H = fft2(cv2.copyMakeBorder(psf_h, *([(psf_l.shape[0] - psf_h.shape[0]) // 2] * 4), cv2.BORDER_CONSTANT, 0))
    K = psf_L / psf_H
    k = abs(fftshift(ifft2(K)))
    if verbose:
        plt.imshow(k, cmap='gray')
        plt.show()
    return k


def get_ks(psfs, verbose=False):
    psf_l_g, psf_h_g, psf_l_b, psf_h_b = psfs
    k_g = find_k(psf_l_g, psf_h_g, verbose)
    k_b = find_k(psf_l_b, psf_h_b, verbose)
    return k_g, k_b


def estimate_images(low_res_k_tuple, verbose=False):
    for img, k in low_res_k_tuple:
        # restored_img_wiener = wiener(img, k, 1)
        restored_img = TVL2Deconv(k, img, 0.01).solve()
        if verbose:
            # fig = plt.figure()
            # plt.imshow(restored_img_wiener, cmap='gray')
            # fig.suptitle("Wiener")
            # plt.show()
            fig = plt.figure()
            plt.imshow(restored_img, cmap='gray')
            fig.suptitle("TV")
            plt.show()
    pass


if __name__ == '__main__':
    cont_scene = cv2.imread('../DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE)
    psfs = get_psfs(verbose=False)
    images_from_psf = create_images(cont_scene, psfs, verbose=False)
    ks = get_ks(psfs, verbose=False)
    imgs_to_estimate = [(images_from_psf[0], ks[0]), (images_from_psf[2], ks[1])]
    est_imgs = estimate_images(imgs_to_estimate, verbose=True)
