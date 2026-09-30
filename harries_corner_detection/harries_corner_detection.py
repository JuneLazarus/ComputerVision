import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

img_dir = "./assets/images"
imgs = []


def imageloader():
    os.makedirs(f"{img_dir}", exist_ok=True)
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(img_dir, fname)
            img = plt.imread(full_path)
            imgs.append((img, os.path.splitext(fname)[0]))


# correlation and convolution
def cross_correlation_2d(img, kernel):
    height, width = img.shape[0], img.shape[1]
    kernel_height, kernel_width = kernel.shape
    ph = (kernel_height - 1) // 2
    pw = (kernel_width - 1) // 2
    pad = ((ph, ph), (pw, pw)) + ((0, 0),) * (img.ndim - 2)
    padded = np.pad(img, pad, mode = "reflect")
    out = np.zeros_like(img, dtype = np.float64)
    for k in range(-ph, ph + 1):
        for l in range(-pw, pw + 1):
            out += kernel[k + ph, l + pw] * padded[k + ph:k + ph + height,
                                         l + pw:l + pw + width]
    return out


def convolve_2d(img, kernel):
    kernel = np.flip(kernel, axis = (0, 1))
    return cross_correlation_2d(img, kernel)


# Gaussian blur and low-pass filtering
def gaussian_kernel(k, sigma):
    row = np.arange(-(k // 2), k // 2 + 1, dtype = int)
    mesh = np.meshgrid(row, row)
    kernel = np.exp(-(mesh[0]**2 + mesh[1]**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def low_pass(img, k, sigma):
    kernel = gaussian_kernel(k, sigma)
    return convolve_2d(img, kernel)


def grayScale(img):
    if img.ndim == 2:
        gray = img.astype(np.float64)
    else:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    if gray.max() <= 1.0:
        gray = gray * 255.0
    return gray.astype(np.uint8)


def sobel_gradient(img):
    sobelX = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    Gx = convolve_2d(img, sobelX)
    Gy = convolve_2d(img, sobelY)
    return Gx, Gy


def calculate_response_value(Gx, Gy):
    Ixx = Gx ** 2
    Iyy = Gy ** 2
    Ixy = Gx * Gy
    kernel = gaussian_kernel(5, 2)
    Sxx = cross_correlation_2d(Ixx, kernel)
    Syy = cross_correlation_2d(Iyy, kernel)
    Sxy = cross_correlation_2d(Ixy, kernel)
    return Sxx * Syy - Sxy ** 2 - 0.05 * (Sxx + Syy) ** 2

def NMS(R):
    h, w = R.shape
    threshold = max(np.percentile(R, 97), 0.01 * R.max())
    out = (R > threshold) & (R > 0)
    out[ : , : 3] = False
    out[ : , -3: ] = False
    out[ : 3, : ] = False
    out[ -3: , : ] = False
    padded = np.pad(R, 2)
    for i in range(5):
        for j in range(5):
            out &= R >= padded[i: h + i, j: w + j]
    return out


def harries_corner_detection(img, fname):
    gray = grayScale(img)
    Gx, Gy = sobel_gradient(gray)
    R = calculate_response_value(Gx, Gy)
    ys, xs = np.where(NMS(R))
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  
    for x, y in zip(xs, ys):
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite(f"./assets/results/{fname}_.png", out)
    print(f"{fname}_.png saved with {len(xs)} corners detected")
    return out


if __name__ == "__main__":
    imageloader()
    os.makedirs("./assets/results", exist_ok=True)
    os.makedirs("./assets/outputs", exist_ok=True)
    fig, axes = plt.subplots(nrows = len(imgs), ncols = 2, figsize = (10, 5), squeeze=False)
    for i, (img, fname) in enumerate(imgs):
        axes[i, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i, 0].set_title(f"{fname}.Origin")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(harries_corner_detection(img, fname), cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{fname}.Corners")
        axes[i, 1].axis("off")
    plt.tight_layout()
    fig.savefig("./assets/outputs/output.png", dpi = 300, bbox_inches = "tight")
    plt.show()
    print("finished")