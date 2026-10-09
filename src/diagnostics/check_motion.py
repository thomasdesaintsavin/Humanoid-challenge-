import cv2, numpy as np, glob

c = np.load("data/processed/calib.npz")
cross = c["image_pts"]
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())

def first_frames(name, n=15):
    return glob.glob(f"data/raw/calib/{name}.*")[0], n

def marker_corners(video, n=15):
    cap = cv2.VideoCapture(video); out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        corners, ids, _ = det.detectMarkers(f)
        if ids is not None and 0 in ids.flatten():
            out.append(corners[list(ids.flatten()).index(0)].reshape(4, 2))
    return np.median(out, axis=0) if out else None

ref = marker_corners(glob.glob("data/raw/calib/calib_marker.*")[0])
for name in ("calib_marker", "calib_gripper", "calib_touch"):
    m = marker_corners(glob.glob(f"data/raw/calib/{name}.*")[0])
    if m is None:
        print(f"{name:14s} marqueur non détecté")
        continue
    shift = np.linalg.norm(m - ref, axis=1)
    print(f"{name:14s} décalage des 4 coins du marqueur (px) : {shift.round(1)}  moyenne {shift.mean():.1f}")

# image de contrôle : tes croix calibrées dessinées sur la première image de chaque vidéo
x0, y0 = (cross.min(0) - 250).astype(int).clip(0)
x1, y1 = (cross.max(0) + 250).astype(int)
for name in ("calib_marker", "calib_touch"):
    cap = cv2.VideoCapture(glob.glob(f"data/raw/calib/{name}.*")[0])
    ok, f = cap.read()
    for j, (x, y) in enumerate(cross):
        cv2.circle(f, (int(x), int(y)), 22, (0, 0, 255), 3)
        cv2.putText(f, f"{j+1:02d}", (int(x) + 28, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
    cv2.imwrite(f"results/check_{name}_circles.png", cv2.resize(f[y0:y1, x0:x1], None, fx=0.6, fy=0.6))
    print("sauvé : results/check_%s_circles.png" % name)