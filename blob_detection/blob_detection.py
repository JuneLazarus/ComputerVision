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


def laplace_kernel(sigma):
    k = int(np.ceil(3 * sigma))
    row = np.arange(-k, k + 1, dtype = int)
    mesh = np.meshgrid(row, row)
    transition = (mesh[0]**2 + mesh[1]**2) / sigma**2
    kernel = np.exp(-transition / 2.0) * (transition - 2.0) / (2.0 * np.pi * sigma**2)
    kernel -= kernel.mean()
    return kernel


def grayScale(img):
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    if gray.max() > 1.0:
        gray /= 255.0
    return gray.astype(np.float32)


def locateBlobs(gray):
    h, w = gray.shape
    layers = max(3, int(np.floor(np.log(min(h, w) / 80) / np.log(1.26) + 1e-9)) + 1)
    sigmas = max(4, min(h, w) / 80) * 1.26**np.arange(layers)
    response_value = [cv2.filter2D(gray, -1, laplace_kernel(sigma)) for sigma in sigmas]
    L = np.stack(response_value)
    padded = np.pad(L, 1)
    ge = np.ones_like(L, dtype = bool)
    le = np.ones_like(L, dtype = bool)
    c, w, h = L.shape
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i, j, k) == (1, 1, 1):
                    continue
                neigh = padded[i: c + i, j: w + j, k: h + k]
                ge &= (L > neigh)
                le &= (L < neigh)
    out = (ge | le) & (np.abs(L) > 0.05)
    out[0, : , : ] = out[-1, : , : ] = False
    out[ : , 0, : ] = out[ : , -1, : ] = False
    out[ : , : , 0] = out[ : , : , -1] = False
    candidates = [(x, y, int(round(1.414 * sigmas[k])), l) for (k, y, x), l in zip(np.argwhere(out), np.abs(L[out]))]
    candidates.sort(key = lambda c: -c[3])
    blobs = []
    for x, y, r, _ in candidates:
        if all((x - bx) ** 2 + (y - by) ** 2 > (0.5 * (r + br)) ** 2 for bx, by, br in blobs):
            blobs.append((x, y, r))
        if len(blobs) >= int(round(h * w / 500)):
            break
    return blobs


def blob_detection(img, fname):
    gray = grayScale(img)
    blobs = locateBlobs(gray)
    out = cv2.cvtColor((np.clip(gray, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y, r in blobs:
        cv2.circle(out, (x, y), r, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"./assets/results/{fname}_.png", out)
    print(f"{fname}_.png saved with {len(blobs)} blobs detected")
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
        axes[i, 1].imshow(cv2.cvtColor(blob_detection(img, fname), cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{fname}.Blobs")
        axes[i, 1].axis("off")
    plt.tight_layout()
    fig.savefig("./assets/outputs/output.png", dpi = 300, bbox_inches = "tight")
    plt.show()
    print("finished")