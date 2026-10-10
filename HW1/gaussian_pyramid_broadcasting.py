import numpy as np
import matplotlib.pyplot as plt
import os

img_dir = "./assets/sources"
imgs = []

def isValidPosition(img, x, y):
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]

# correlation and convolution
def cross_correlation_2d(img, kernel, mode):
    height, width = img.shape[0], img.shape[1]
    kernel_height, kernel_width = kernel.shape
    if mode == "discard":
        out = np.zeros((height - kernel_height + 1, width - kernel_width + 1) + img.shape[2:], dtype = np.float64)
        for k in range(kernel_height):
            for l in range(kernel_width):
                out += kernel[k, l] * img[k:k + height - kernel_height + 1,
                                         l:l + width - kernel_width + 1]
    if mode == "keep":
        ph = (kernel_height - 1) // 2
        pw = (kernel_width - 1) // 2
        pad = ((ph, ph), (pw, pw)) + ((0, 0),) * (img.ndim - 2)
        padded = np.pad(img, pad, mode = "constant")
        out = np.zeros_like(img, dtype = np.float64)
        for k in range(-ph, ph + 1):
            for l in range(-pw, pw + 1):
                out += kernel[k + ph, l + pw] * padded[k + ph:k + ph + height,
                                             l + pw:l + pw + width]
    return out

def convolve_2d(img, kernel, mode):
    kernel = np.flip(kernel, axis = (0, 1))
    return cross_correlation_2d(img, kernel, mode)

# Gaussian blur and low-pass filtering
def gaussian_blur_kernel_2d(k, sigma):
    row = np.arange(-(k // 2), k // 2 + 1, dtype = int)
    mesh = np.meshgrid(row, row)
    kernel = np.exp(-(mesh[0]**2 + mesh[1]**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def low_pass(img, k, sigma, mode):
    kernel = gaussian_blur_kernel_2d(k, sigma)
    return convolve_2d(img, kernel, mode)

# Subsampling
def subsampling(img, step, mode):
    height, width = img.shape[0], img.shape[1]
    g_img = low_pass(img, 3, 1, "keep") if mode == "filter" else img
    return g_img[::step, ::step]

def gaussian_pyramid(img, levels, step, mode):
    out = [img]
    prev = img.copy()
    for i in range(levels - 1):
        prev = subsampling(prev, step, mode)
        out.append(prev)
    return out

if __name__ == "__main__":
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(img_dir, fname)
            img = plt.imread(full_path)
            imgs.append(img)
    img = imgs[0]
    if img.max() > 1.0:
        img = img / 255.0
    levels = 4
    gaussian_pyramid_with_filter = gaussian_pyramid(img, levels, 2, "filter")
    gaussian_pyramid_without_filter = gaussian_pyramid(img, levels, 2, "non-filter")
    fig, axes = plt.subplots(nrows = 2, ncols = levels, figsize = (10, 5))
    for i in range(levels):
        axes[0, i].imshow(np.clip(gaussian_pyramid_with_filter[i], 0, 255))
        axes[0, i].set_title(f"F.{i + 1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(gaussian_pyramid_without_filter[i], 0, 255))
        axes[1, i].set_title(f"N.{i + 1}")
        axes[1, i].axis("off")
    plt.tight_layout()
    fig.savefig("./assets/results/result_broadcasting.png", dpi = 300, bbox_inches = "tight")
    plt.show()