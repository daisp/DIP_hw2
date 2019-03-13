import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2, fftshift
from skimage.restoration import wiener
from hw2.sporco.admm.tvl1 import TVL1Deconv
from hw2.sporco.admm.tvl2 import TVL2Deconv
from scipy.misc import imresize


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
        fig = plt.figure()
        plt.imshow(psf_l_g, interpolation='none', cmap='gray')
        fig.suptitle('Gaussian PSF L')
        plt.savefig(f'results/psf_l_g.png', bbox_inches='tight')
        plt.show()

        fig = plt.figure()
        plt.imshow(psf_h_g, interpolation='none', cmap='gray')
        fig.suptitle('Gaussian PSF H')
        plt.savefig(f'results/psf_h_g.png', bbox_inches='tight')
        plt.show()

        fig = plt.figure()
        plt.imshow(psf_l_b, interpolation='none', cmap='gray')
        fig.suptitle('Box PSF L')
        plt.savefig(f'results/psf_l_b.png', bbox_inches='tight')
        plt.show()

        fig = plt.figure()
        plt.imshow(psf_h_b, interpolation='none', cmap='gray')
        fig.suptitle('Box PSF H')
        plt.savefig(f'results/psf_h_b.png', bbox_inches='tight')
        plt.show()
    return psf_l_g, psf_h_g, psf_l_b, psf_h_b


def get_image_by_psf(cont_scene, psf):
    return fftconvolve(cont_scene, psf, mode='same')


def normalize(x):
    x -= np.min(x)
    return x / np.max(x)


def create_images(cont_scene, psfs, verbose=False):
    images = []
    i = 0
    filename_switcher = {
        0: 'l_g',
        1: 'h_g',
        2: 'l_b',
        3: 'h_b'
    }
    title_switcher = {
        0: 'Gaussian L',
        1: 'Gaussian H',
        2: 'Box L',
        3: 'Box H'
    }
    for psf in psfs:
        psf = normalize(psf)
        img = get_image_by_psf(cont_scene, psf)
        img = normalize(img)
        images.append(img)
        if verbose:
            filename = filename_switcher.get(i)
            title = title_switcher.get(i)
            fig = plt.figure()
            plt.imshow(img, cmap='gray')
            fig.suptitle(f'Blurred {title}')
            plt.savefig(f'results/blurred_{filename}.png', bbox_inches='tight')
            plt.show()
        i += 1
    return images


def find_k(psf_l, psf_h, verbose, filename, title):
    psf_L = fft2(psf_l)
    psf_H = fft2(cv2.copyMakeBorder(psf_h, *([(psf_l.shape[0] - psf_h.shape[0]) // 2] * 4), cv2.BORDER_CONSTANT, 0))
    K = psf_L / psf_H
    k = abs(fftshift(ifft2(K)))
    if verbose:
        fig = plt.figure()
        plt.imshow(k, cmap='gray')
        fig.suptitle(f'K {title}')
        plt.savefig(f'results/{filename}.png', bbox_inches='tight')
        plt.show()
    return k


def get_ks(psfs, verbose=False):
    psf_l_g, psf_h_g, psf_l_b, psf_h_b = psfs
    k_g = find_k(psf_l_g, psf_h_g, verbose, filename='k_g', title='Gaussian')
    k_b = find_k(psf_l_b, psf_h_b, verbose, filename='k_b', title='Box')
    return k_g, k_b


def estimate_images(low_res_k_tuple, verbose=False):
    count = 0
    filename_switcher = {
        0: 'wiener_g',
        1: 'least_squares_tv_g',
        2: 'wiener_b',
        3: 'least_squares_tv_b'
    }
    title_switcher = {
        0: 'Gaussian Wiener',
        1: 'Gaussian Least Squares with TV Prior',
        2: 'Box Wiener',
        3: 'Box Least Squares with TV Prior'
    }
    for img, k in low_res_k_tuple:
        restored_img_wiener = wiener(img, k, 1)
        restored_img_least_squares_tv = TVL2Deconv(k, img, 0.01).solve()
        if verbose:
            fig = plt.figure()
            plt.imshow(restored_img_wiener, cmap='gray')
            fig.suptitle(title_switcher.get(count))
            plt.savefig(f'results/{filename_switcher.get(count)}.png', bbox_inches='tight')
            plt.show()
            count += 1

            fig = plt.figure()
            plt.imshow(restored_img_least_squares_tv, cmap='gray')
            fig.suptitle(title_switcher.get(count))
            plt.savefig(f'results/{filename_switcher.get(count)}.png', bbox_inches='tight')
            plt.show()
            count += 1


def upsample_images(imgs, verbose=False):
    count = 0
    filename_switcher = {
        0: 'bilinear_g',
        1: 'bicubic_g',
        2: 'bilinear_b',
        3: 'bicubic_b'
    }
    title_switcher = {
        0: 'Gaussian Bilinear',
        1: 'Gaussian Bicubic',
        2: 'Box Bilinear',
        3: 'Box Bicubic'
    }
    for image in imgs:
        restored_bilinear = imresize(image[0], (346, 550), interp='bilinear')
        restored_bicubic = imresize(image[0], (346, 550), interp='bicubic')
        if verbose:
            fig = plt.figure()
            plt.imshow(restored_bilinear, cmap='gray')
            fig.suptitle(title_switcher.get(count))
            plt.savefig(f'results/{filename_switcher.get(count)}.png', bbox_inches='tight')
            plt.show()
            count += 1

            fig = plt.figure()
            plt.imshow(restored_bicubic, cmap='gray')
            fig.suptitle(title_switcher.get(count))
            plt.savefig(f'results/{filename_switcher.get(count)}.png', bbox_inches='tight')
            plt.show()
            count += 1


if __name__ == '__main__':
    cont_scene = cv2.imread('../DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE)
    psfs = get_psfs(verbose=False)
    images_from_psf = create_images(cont_scene, psfs, verbose=False)
    ks = get_ks(psfs, verbose=False)
    imgs_to_estimate = [(images_from_psf[0], ks[0]), (images_from_psf[2], ks[1])]
    # estimate_images(imgs_to_estimate, verbose=True)
    upsample_images(imgs_to_estimate, verbose=True)
