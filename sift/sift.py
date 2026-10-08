import matplotlib.pyplot as plt
import matplotlib.patches as pths
import numpy as np
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


def grayScale(img):
    if img.ndim == 2:
        gray = img.astype(np.float64)
    else:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    if gray.max() > 1.0:
        gray /= 255.0
    return gray.astype(np.float64)


def gaussian_kernel(sigma):
    k = int(np.ceil(3 * sigma))
    row = np.arange(-k, k + 1, dtype = int)
    mesh = np.meshgrid(row, row)
    kernel = np.exp(-(mesh[0]**2 + mesh[1]**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def laplace_kernel(sigma):
    k = int(np.ceil(3 * sigma))
    row = np.arange(-k, k + 1, dtype = int)
    mesh = np.meshgrid(row, row)
    transition = (mesh[0]**2 + mesh[1]**2) / sigma**2
    kernel = np.exp(-transition / 2.0) * (transition - 2.0) / (2.0 * np.pi * sigma**2)
    kernel -= kernel.mean()
    return kernel


def subsampling(img, step):
    height, width = img.shape[0], img.shape[1]
    out = np.zeros(((height + 1) // 2, (width + 1) // 2) + img.shape[2:], dtype = np.float64)
    for i in range(0, height, step):
        for j in range(0, width, step):
            out[i // 2, j // 2] = img[i, j]
    return out


def gaussian(dx, dy, sigma):
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))


def sift(img, s):

    gray = grayScale(img)


    sigma0 = 1.6
    layers = max(3, min(gray.shape) // 240)
    k = 2 ** (1 / s)
    prev_gauss = cv2.filter2D(gray, -1, gaussian_kernel(sigma0))
    out = []

    # construct gaussian pyramid and dog pyramid
    for layer in range(layers):
        gauss = [prev_gauss]
        for i in range(1, s + 3):
            sigma_ = sigma0 * np.sqrt((k ** i) ** 2 - (k ** (i - 1)) ** 2)
            kernel = gaussian_kernel(sigma_)
            gauss.append(cv2.filter2D(gauss[-1], -1, kernel))
        dog = [gauss[s + 1] - gauss[s] for s in range(s + 2)]
        prev_gauss = subsampling(gauss[s], 2)

        # find local extrema
        L = np.stack(dog)
        padded = np.pad(L, 1)
        ge = np.ones_like(L, dtype = bool)
        le = np.ones_like(L, dtype = bool)
        c, w, h = L.shape
        for dc in range(3):
            for dw in range(3):
                for dh in range(3):
                    if (dc, dw, dh) == (1, 1, 1):
                        continue
                    neigh = padded[dc: c + dc, dw: w + dw, dh: h + dh]
                    ge &= (L > neigh)
                    le &= (L < neigh)
        candidate = (ge | le) & (np.abs(L) > 0.03)
        candidate[0, : , : ] = candidate[-1, : , : ] = False
        candidate[ : , : 3, : ] = candidate[ : , -3: , : ] = False
        candidate[ : , : , : 3] = candidate[ : , : , -3: ] = False


        # calculate gradient and corresponding orientation
        gs = [[]]
        for r in range(1, s + 1):
            g_img = gauss[r]
            gx = np.zeros_like(g_img); gy = np.zeros_like(g_img)
            gx[:, 1:-1] = g_img[:, 2:] - g_img[:, :-2]
            gy[1:-1, :] = g_img[2:, :] - g_img[:-2, :]
            gm = np.hypot(gx, gy)
            gdeg = (np.degrees(np.arctan2(gy, gx))) % 360
            gs.append((gx, gy, gm, gdeg))


        for r, y, x in np.argwhere(candidate):


            # hessian matrix
            g = np.array([(L[r + 1, y, x] - L[r - 1, y, x]) / 2,
                         (L[r, y + 1, x] - L[r, y - 1, x]) / 2,
                         (L[r, y, x + 1] - L[r, y, x - 1]) / 2])
            drr = L[r + 1, y, x] + L[r - 1, y, x] - 2 * L[r, y, x]
            dxx = L[r, y, x + 1] + L[r, y, x - 1] - 2 * L[r, y, x]
            dyy = L[r, y + 1, x] + L[r, y - 1, x] - 2 * L[r, y, x]
            drx = (L[r + 1, y, x + 1] - L[r + 1, y, x - 1] - L[r - 1, y, x + 1] + L[r - 1, y, x - 1]) / 4
            dry = (L[r + 1, y + 1, x] - L[r + 1, y - 1, x] - L[r - 1, y + 1, x] + L[r - 1, y - 1, x]) / 4
            dxy = (L[r, y + 1, x + 1] - L[r, y + 1, x - 1] - L[r, y - 1, x + 1] + L[r, y - 1, x - 1]) / 4
            H = np.array([[drr, dry, drx], [dry, dyy, dxy], [drx, dxy, dyy]])
            if abs(np.linalg.det(H)) < 1e-9:
                continue
            offset = -np.linalg.solve(H, g)
            fr = sigma0 * k ** (r + offset[0])
            fy = y + offset[1]
            fx = x + offset[2]
            if np.any(np.abs(offset) > 0.5):
                continue
            if abs(L[r, y, x] + 0.5 * g @ offset) < 0.03:
                continue
            R = (dxx + dyy) ** 2 / (dxx * dyy - dxy ** 2 + 1e-9)
            if R <= 0.0 or R >= 10.0:
                continue


            # calculate main orientation
            gx, gy, gm, gdeg = gs[r]
            sigma = sigma0 * k ** r
            radius = int(round(4.5 * sigma))
            hog  = np.zeros(36)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy, xx = y + dy, x + dx
                    if not (1 <= xx < h - 1 and 1 <= yy < w - 1) or (dy ** 2 + dx ** 2) > radius ** 2:
                        continue
                    hog[int(gdeg[yy, xx] // 10 % 36)] += gm[yy, xx] * gaussian(dx, dy, 1.5 * sigma)

            bins = [int(np.argmax(hog))]
            for b in range(36):
                if b == bins[0] or b == (bins[0] + 1) % 36 or b == (bins[0] - 1) % 36:
                    continue
                if hog[b] >= 0.8 * hog[bins[0]]:
                    bins.append(b)

            orientations = []
            for b in bins:
                hl, hc, hr =  hog[(b - 1) % 36], hog[b], hog[(b + 1) % 36]
                if hc <= hl or hc <= hr:
                    continue
                offset = 0.5 * (hl - hr) / (hl - 2 * hc + hr) if abs(hl - 2 * hc + hr) > 1e-9 else 0.0
                offset = np.clip(offset, -0.5, 0.5)
                orientations.append(int((b + offset) * 10) % 360)

            
            # calculate descriptor
            for idx, orientation in enumerate(orientations):
                ct, st = np.cos(np.radians(orientation)), np.sin(np.radians(orientation))
                descriptor_ = np.zeros((4, 4, 8))
                for nx in range(-8, 8):
                    for ny in range(-8, 8):
                        rx = ct * nx - st * ny
                        ry = st * nx + ct * ny
                        ox, oy = int(round(x + rx)), int(round(y + ry))
                        if not(1 <= ox < h - 1 and 1 <= oy < w - 1):
                            continue
                        bx, by = int((nx + 8) // 4), int((ny + 8) // 4)
                        if not(0 <= bx < 4 and 0 <= by < 4):
                            continue
                        bOrientation = int((gdeg[oy, ox] - orientation) % 360 // 45) % 8
                        descriptor_[by, bx, bOrientation] += gm[oy, ox] * gaussian(rx, ry, 8.0)
                
                descriptor = descriptor_.flatten()
                descriptor /= (np.linalg.norm(descriptor) + 1e-9)
                descriptor = np.clip(descriptor, 0, 0.2)
                descriptor /= (np.linalg.norm(descriptor) + 1e-9)

                out.append((fx * 2 ** layer, fy * 2 ** layer, fr * 2 ** layer, orientation, descriptor, idx == 0))
    return out


def showSift(img, fname, sift_result):
    gray = grayScale(img)
    out = cv2.cvtColor((np.clip(gray, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y, sigma, orientation, descriptor, isMainOrientation in sift_result:
        cv2.circle(out, (int(x), int(y)), int(round(sigma * 3)), (0, 255, 0), 1)
        rad = np.radians(orientation)
        cv2.line(out, (int(x), int(y)), (int(x + sigma * 3 * np.cos(rad)), int(y + sigma * 3 * np.sin(rad))), (0, 255, 0), 1)
        scale = 2.0 * sigma
        ct, st = np.cos(np.radians(orientation)), np.sin(np.radians(orientation))
        
        if not isMainOrientation:
            continue
        descriptor = descriptor.reshape((4, 4, 8))
        for by in range(4):
            for bx in range(4):
                cx = (bx - 1.5) * scale
                cy = (by - 1.5) * scale
                gx = x + ct * cx - st * cy
                gy = y + st * cx + ct * cy
                vals = descriptor[by, bx]
                for o in range(8):
                    ang = orientation + o * 45
                    L = vals[o] * scale * 3
                    ex = gx + L * np.cos(np.radians(ang))
                    ey = gy + L * np.sin(np.radians(ang))
                    cv2.line(out, (int(gx), int(gy)), (int(ex), int(ey)), (0, 255, 255), 1)

    cv2.imwrite(f"./assets/results/{fname}_.png", out)
    print(f"{fname}_ saved")
    return out


def dod(descriptor1, descriptor2):
    d1 = np.array(descriptor1)
    d2 = np.array(descriptor2)
    dist = np.linalg.norm(d1 - d2)
    return dist


def locateMatches(img1, fname1, sift1, img2, fname2, sift2):

    matches = []
    for x1, y1, _, _, descriptor1, _ in sift1:
        best_match = None
        best_dist = float("inf")
        second_best_dist = float("inf")
        for x2, y2, _, _, descriptor2, _ in sift2:
            dist = dod(descriptor1, descriptor2)
            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_match = (x2, y2)
            elif dist < second_best_dist:
                second_best_dist = dist
        if best_dist < 0.75 * second_best_dist:
            matches.append(((x1, y1), best_match, best_dist))
    matches.sort(key = lambda x: x[2])
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))

    gray1 = grayScale(img1)
    gray2 = grayScale(img2)
    ax1.imshow(gray1, cmap = "gray" if gray1.ndim == 2 else None)
    ax2.imshow(gray2, cmap = "gray" if gray2.ndim == 2 else None)
    for (x1, y1), (x2, y2), _ in matches[:min(len(matches), 50)]:
        color = np.random.rand(3,)
        color.clip(0.4, 1)
        ax1.plot(x1, y1, 'ro', markersize=2)
        ax2.plot(x2, y2, 'ro', markersize=2)
        connection = pths.ConnectionPatch(xyA = (x1, y1), coordsA = ax1.transData, 
                          xyB = (x2, y2), coordsB = ax2.transData, 
                          color = color, linewidth = 0.8)
        fig.add_artist(connection)
    ax1.set_title(f"{fname1}.1")
    ax2.set_title(f"{fname2}.2")
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    fig.savefig("./assets/outputs/match.png", dpi = 300, bbox_inches = "tight")
    plt.show()
    print(f"{len(matches)} matches found.")


if __name__ == "__main__":
    imageloader()
    if not imgs:
        print("No images found in the directory.")
        exit(1)
    sifts = []
    fig, axes = plt.subplots(nrows = len(imgs), ncols = 2, figsize = (10, 5), squeeze=False)
    for i, (img, fname) in enumerate(imgs):
        axes[i, 0].imshow(img, cmap = "gray" if img.ndim == 2 else None)
        axes[i, 0].set_title(f"{fname}.Origin")
        axes[i, 0].axis("off")
        sifts.append(sift(img, 3))
        out = showSift(img, fname, sifts[-1])
        axes[i, 1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{fname}.Blobs")
        axes[i, 1].axis("off")
    plt.tight_layout()
    fig.savefig("./assets/outputs/output.png", dpi = 300, bbox_inches = "tight")
    plt.show()
    if len(imgs) >= 2:
        locateMatches(imgs[0][0], imgs[0][1], sifts[0], imgs[1][0], imgs[1][1], sifts[1])
    print("finished.")