import matplotlib.pyplot as plt
import matplotlib.patches as pths
import numpy as np
import random
import cv2
import os


random.seed(0)

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
    imgs.sort(key = lambda x: x[1])


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
    out = np.zeros(((height + 1) // step, (width + 1) // step) + img.shape[2:], dtype = np.float64)
    for i in range(0, height, step):
        for j in range(0, width, step):
            out[i // step, j // step] = img[i, j]
    return out


def gaussian(dx, dy, sigma):
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))


def sift(img, s = 3):

    gray = grayScale(img)

    sigma0 = 1.6
    layers = max(3, min(gray.shape) // 240)
    k = 2 ** (1 / s)
    step = 2
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
        prev_gauss = subsampling(gauss[s], step)

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
            H = np.array([[drr, dry, drx], [dry, dyy, dxy], [drx, dxy, dxx]])
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

                out.append((fx * step ** layer, fy * step ** layer, fr * step ** layer, orientation, descriptor, idx == 0))
    return out


def showSift(img, fname, sift):
    gray = grayScale(img)
    out = cv2.cvtColor((np.clip(gray, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y, sigma, orientation, descriptor, isMainOrientation in sift:
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
    print(f"{fname}_.png saved")
    return out


def dod(descriptor1, descriptor2):
    d1 = np.array(descriptor1)
    d2 = np.array(descriptor2)
    dist = np.linalg.norm(d1 - d2)
    return dist


def RANSAC(matches):
    iterations = 1000
    best_sam = 0
    diff = 3
    out = []

    for _ in range(iterations):
        cur = []
        dims = random.sample(range(len(matches)), 4)
        A = np.empty((0, 9), dtype = np.float64)
        for dim in dims:
            x1, y1, x2, y2 = matches[dim]
            newx = np.array([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            newy = np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
            A = np.vstack([A, newx, newy])
        _, _, Vt = np.linalg.svd(A, full_matrices = False)
        v = Vt[-1, : ]
        v /= (np.linalg.norm(v) + 1e-9)
        h = v.reshape((3, 3))

        sam = 0
        for x1, y1, x2, y2 in matches:
            x, y, w = h @ np.array([x1, y1, 1])
            if abs(x / (w + 1e-9) - x2) < diff and abs(y / (w + 1e-9) - y2) < diff:
                sam += 1
                cur.append((x1, y1, x2, y2))
        if sam > best_sam:
            best_sam = sam
            out = cur

    return out


# locate no loop
def locateMatches(sift1, sift2):

    desc1 = np.stack([s[4] for s in sift1]).astype(np.float64)
    desc2 = np.stack([s[4] for s in sift2]).astype(np.float64)
    N, M = desc1.shape[0], desc2.shape[0]

    sq1 = np.sum(desc1**2, axis=1)[ :, None]
    sq2 = np.sum(desc2**2, axis=1)[None, : ]
    dist_ = sq1 + sq2 - 2.0 * np.matmul(desc1, desc2.T)
    dist = np.sqrt(np.maximum(0.0, dist_))

    idx = dist.argmin(axis = 1)
    best_dist = dist[np.arange(N), idx]
    second_best_dist = np.partition(dist, 1, axis = 1)[:, 1]
    cand = (best_dist < 0.75 * second_best_dist) & (best_dist < 0.8)

    cross_idx = dist.argmin(axis = 0)

    rows = np.nonzero(cand)[0]
    cross_matched = cross_idx[idx[rows]] == rows
    cand = rows[cross_matched]
    
    coord1 = [(s[0], s[1]) for s in sift1]
    coord2 = [(s[0], s[1]) for s in sift2] 
    matches = [(coord1[i][0], coord1[i][1], coord2[idx[i]][0], coord2[idx[i]][1]) for i in cand]

    return RANSAC(matches)


def showMatches(imgs, sifts):

    for i in range(len(imgs) - 1):

        img1, name1 = imgs[i]
        img2, name2 = imgs[i + 1]
        gray1 = grayScale(img1)
        gray2 = grayScale(img2)
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
        ax1, ax2 = axes
        ax1.imshow(gray1, cmap = "gray" if gray1.ndim == 2 else None)
        ax2.imshow(gray2, cmap = "gray" if gray2.ndim == 2 else None)
        
        matches = locateMatches(sifts[i], sifts[i + 1])
        print(f"fig.{name1} and fig.{name2}: {len(matches)} matches found.")
        for x1, y1, x2, y2 in matches[: min(8, len(matches))]:
            color = np.random.rand(3,)
            color = color.clip(0.4, 1)
            ax1.plot(x1, y1, 'ro', markersize=2) 
            ax2.plot(x2, y2, 'ro', markersize=2)
            connection = pths.ConnectionPatch(xyA = (x1, y1), coordsA = ax1.transData, 
                              xyB = (x2, y2), coordsB = ax2.transData, 
                              color = color, linewidth = 0.8)
            fig.add_artist(connection)

        ax1.set_title(f"fig.{name1}")
        ax2.set_title(f"fig.{name2}")
        ax1.axis("off")
        ax2.axis("off")

        fig.suptitle("Matches", fontsize = 16)
        plt.tight_layout()
        plt.subplots_adjust(top = 1)
        fig.savefig(f"./assets/outputs/{name1} & {name2}.png", dpi = 300, bbox_inches = "tight")
        # plt.show()
        print(f"match results of {name1} & {name2} saved.")
        


if __name__ == "__main__":
    imageloader()
    if not imgs:
        print("No images found in the directory.")
        exit(1)
    sifts = []

    '''
    fig, axes = plt.subplots(nrows = len(imgs), ncols = 2, figsize = (10, 5), squeeze=False)
    for i, (img, name) in enumerate(imgs):
        axes[i, 0].imshow(img, cmap = "gray" if img.ndim == 2 else None)
        axes[i, 0].set_title(f"{name}.Origin")
        axes[i, 0].axis("off")
        sifts.append(sift(img))
        print(f"fig.{name} sift calculated.")
        out = showSift(img, name, sifts[-1])
        axes[i, 1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{name}.Sift")
        axes[i, 1].axis("off")
    fig.suptitle("Sift Descriptors", fontsize = 16)
    plt.tight_layout()
    fig.savefig("./assets/outputs/output.png", dpi = 300, bbox_inches = "tight")
    # plt.show()
    '''
    
    for img, name in imgs:
        sifts.append(sift(img))
        fig, axis = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
        axis[0].imshow(img, cmap = "gray" if img.ndim == 2 else None)
        axis[0].set_title(f"{name}.Origin")
        axis[0].axis("off")
        out = showSift(img, name, sifts[-1])
        axis[1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        axis[1].set_title(f"{name}.Sift")
        axis[1].axis("off")
    
    if len(imgs) >= 2:
        showMatches(imgs, sifts)

    print("finished.")
