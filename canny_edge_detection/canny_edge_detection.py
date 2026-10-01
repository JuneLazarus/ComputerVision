import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

img_dir = "./assets/images"
imgs = []

def imageloader():
    os.makedirs(f"{img_dir}", exist_ok=True)
    os.makedirs("./assets/results", exist_ok=True)
    os.makedirs("./assets/outputs", exist_ok=True)
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(img_dir, fname)
            img = plt.imread(full_path)
            imgs.append((img, os.path.splitext(fname)[0]))



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
        padded = np.pad(img, pad, mode = "reflect")
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


def sobel_edge_detection(img):
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
    Gx = convolve_2d(img, sobelX, "keep")
    Gy = convolve_2d(img, sobelY, "keep")
    G = np.hypot(Gx, Gy)
    ax, ay = np.abs(Gx), np.abs(Gy)
    t = 0.41421356                        
    ang = np.full(Gx.shape, 2, np.uint8)  
    ang[ay <= t * ax] = 0                 
    m = ~((ay <= t * ax) | (ax <= t * ay))
    ang[m] = np.where((Gx * Gy)[m] > 0, 1, 3)
    return G, ang


def grayScale(img):
    if img.ndim == 2:
        gray = img.astype(np.float64)
    else:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    if gray.max() <= 1.0:
        gray = gray * 255.0
    return gray.astype(np.uint8)


def NMS(G, ang):
    padded = np.pad(G, 1)
    origin = padded[1: -1, 1: -1]
    L = padded[1: -1, : -2]
    UL = padded[: -2, : -2]
    U = padded[: -2, 1: -1]
    UR = padded[: -2, 2: ]
    R = padded[1: -1, 2: ]
    DR = padded[2: , 2: ]
    D = padded[2: , 1: -1]
    DL = padded[2: , : -2]
    neighbour = np.stack([L, UL, U, UR, R, DR, D, DL])
    idx = np.stack([ang, ang + 4])
    n1, n2 = np.take_along_axis(neighbour, idx, axis=0)
    keep = (origin > n1) & (origin >= n2) & (origin > 0)
    return np.where(keep, origin, 0)


def double_threshold_hysteresis(edges):
    if edges.max() == 0:
        return np.zeros_like(edges, dtype=bool)
    hi = np.percentile(edges, 96)
    lo = hi * 0.4
    strong = edges >= hi
    weak = (edges >=lo) & (edges < hi)
    out = strong.copy()
    while True:
        padded = np.pad(out, 1)
        neighbour = np.zeros_like(out)
        for i in range(3):
            for j in range(3):
                neighbour |= padded[i: out.shape[0] + i, j: out.shape[1] + j]
        new = neighbour & weak & ~out
        if not new.any():
            break
        out |= new
    return out


def canny_edge_detection(img, fname):
    gray = grayScale(img)
    X = low_pass(gray, 5, 1, "keep")
    G, ang = sobel_edge_detection(X)
    nms = NMS(G, ang)
    out = double_threshold_hysteresis(nms)
    result = cv2.cvtColor(out.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"./assets/results/{fname}_.png", result)
    print(f"{fname}_.png saved")
    return result


if __name__ == "__main__":
    imageloader()
    fig, axes = plt.subplots(nrows = len(imgs), ncols = 2, figsize = (10, 5), squeeze=False)
    for i, (img, fname) in enumerate(imgs):
        axes[i, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i, 0].set_title(f"{fname}.Origin")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(canny_edge_detection(img, fname), cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{fname}.Blobs")
        axes[i, 1].axis("off")
    plt.tight_layout()
    fig.savefig("./assets/outputs/output.png", dpi = 300, bbox_inches = "tight")
    plt.show()
