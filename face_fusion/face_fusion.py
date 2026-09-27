import cv2, dlib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

img_dir = "./assets/images/"
imgs = []


def meshing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./assets/models/shape_predictor_68_face_landmarks.dat")
    faces = detector(gray, 3)
    if (not faces):
        print("No face detected\n")
        return None
    shape = predictor(gray, faces[0])
    pts = np.array([[p.x, p.y] for p in shape.parts()])
    h, w = img.shape[:2]
    border = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    return np.vstack([pts, border])


def warping(img, canvas, src_pts, tar_pts):
    s = cv2.boundingRect(np.float32([src_pts]))
    t = cv2.boundingRect(np.float32([tar_pts]))
    sl = np.array([(p[0] - s[0], p[1] - s[1]) for p in src_pts], np.float32)
    tl = np.array([(p[0] - t[0], p[1] - t[1]) for p in tar_pts], np.float32)
    mask = np.zeros((t[3], t[2], 3),np.float32)
    cv2.fillConvexPoly(mask, np.int32(tl), (1, 1, 1))
    M = cv2.getAffineTransform(sl, tl)
    warped = cv2.warpAffine(img[s[1]: s[1] + s[3], s[0]: s[0] + s[2]], M, (t[2], t[3]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT_101)
    regionOfInterest = canvas[t[1]: t[1] + t[3], t[0]: t[0] + t[2]]
    regionOfInterest[:] = regionOfInterest * (1 - mask) + warped * mask



def fusing(image, target):
    img, src_pts = image
    tar, tar_pts = target
    triangles = Delaunay(tar_pts)
    out = np.zeros_like(tar, np.float32)
    for i in triangles.simplices:
        warping(img, out, src_pts[i], tar_pts[i])
    return out * 0.4 + tar * 0.6


if __name__ == "__main__":
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(img_dir, fname)
            img = cv2.imread(full_path)
            imgs.append(img)
    usable = []
    for img in imgs:
        pts = meshing(img)
        if pts is not None:
            usable.append((img, pts))
    if len(usable) >= 2:
        out = fusing(usable[0], usable[1])
        cv2.imwrite(f"./assets/outputs/out.png", np.clip(out, 0, 255).astype(np.uint8))
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 5))
        result = [usable[0][0], usable[1][0], out]
        for i in range(3):
            axes[i].imshow(cv2.cvtColor(np.clip(result[i], 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"F.{i + 1}")
            axes[i].axis("off")
        plt.tight_layout()
        fig.savefig("./assets/outputs/result.png", dpi = 300, bbox_inches = "tight")
        plt.show()
        print("fusing successfully\n")
    else:
        print("fusing failed\n")
        
        '''
        pts, tri = meshing(img)
        out = img.copy()
        for a, b, c in tri.simplices:
            t = np.array([pts[a], pts[b], pts[c]], np.int32)
            cv2.polylines(out, [t], True, (255, 0, 0), 1)
        print("Face meshed")
        cv2.imwrite(f"./assets/outputs/{os.path.splitext(fname)[0]}_mesh.png", out)
        '''