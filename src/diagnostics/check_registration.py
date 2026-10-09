import cv2, numpy as np, glob
from scipy.ndimage import label, binary_closing

c = np.load("data/processed/calib.npz")
Hinv = np.linalg.inv(c["H"])                  # image -> table (m)
world = c["world"][:, :2] * 100               # croix 01..08, cm
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())

def video(name):
    return glob.glob(f"data/raw/calib/{name}.*")[0]

def read_frame(name, idx=10):
    cap = cv2.VideoCapture(video(name))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    assert ok, name
    return f

def marker_center(f):
    corners, ids, _ = det.detectMarkers(f)
    return corners[list(ids.flatten()).index(0)].reshape(4, 2).mean(axis=0)

def bg_transform(a, b):
    # transformation (rotation, échelle, translation) de l'image a vers l'image b, calculée sur le décor
    ga = cv2.cvtColor(cv2.resize(a, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(6000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    pa = np.float32([ka[x.queryIdx].pt for x in m]) * 2
    pb = np.float32([kb[x.trainIdx].pt for x in m]) * 2
    A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4)
    assert A is not None, "pas assez de points communs entre les deux images"
    return A, int(inl.sum()), len(m)

ref = read_frame("calib_marker")
m_ref = marker_center(ref)
ctr = np.array([ref.shape[1] / 2, ref.shape[0] / 2])
grid_ctr = cv2.perspectiveTransform(np.float32([[c["world"][:, :2].mean(0)]]), c["H"]).reshape(2)

for name in ("calib_gripper", "calib_touch"):
    f = read_frame(name)
    A, ninl, nm = bg_transform(ref, f)
    move = lambda p: A[:, :2] @ p + A[:, 2] - p
    ang = np.degrees(np.arctan2(A[1, 0], A[0, 0])); sc = np.hypot(A[0, 0], A[1, 0])
    print(f"{name}: décor ({ninl}/{nm} points concordants) : décalage au centre de la grille {move(grid_ctr).round(0)} px, "
          f"au centre de l'image {move(ctr).round(0)} px, rotation {ang:.2f} deg, échelle {sc:.3f}")
    print(f"   marqueur : décalage {(marker_center(f) - m_ref).round(0)} px")

# les 8 contacts de calib_touch
d = np.load("data/processed/calib_touch.npz")
t = d["t"]; fps = float(d["fps"]); lm = d["landmarks2d"]; ap = d["aperture"] * 100

def fill(x):
    x = x.copy()
    for col in ([x] if x.ndim == 1 else list(x.T)):
        bad = np.isnan(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return x

mid = fill((lm[:, 4] + lm[:, 8]) / 2); ap = fill(ap)
h = max(1, int(0.1 * fps))
speed = np.full(len(mid), 1e9)
for i in range(h, len(mid) - h):
    speed[i] = np.linalg.norm(mid[i + h] - mid[i - h]) / (2 * h / fps)
mask = binary_closing((speed < 300) & (ap < 4), structure=np.ones(int(0.25 * fps)))
lab, n = label(mask)
segs = [np.where(lab == k)[0] for k in range(1, n + 1)]
segs = [s for s in segs if len(s) >= 0.4 * fps]
assert len(segs) >= 8, "moins de 8 pauses"
segs = sorted(sorted(segs, key=len, reverse=True)[:8], key=lambda s: s[0])

f = read_frame("calib_touch")
A, _, _ = bg_transform(ref, f)
Ainv = cv2.invertAffineTransform(A)
shift_marker = marker_center(f) - m_ref

def to_table(px):
    return cv2.perspectiveTransform(px.reshape(1, 1, 2).astype(np.float64), Hinv).ravel() * 100

print("\ncroix | sans correction | correction par le décor | correction par le marqueur   (erreur xy, cm)")
res = []
for k, s in enumerate(segs):
    p = np.median(mid[s], axis=0)
    cands = [p, Ainv[:, :2] @ p + Ainv[:, 2], p - shift_marker]
    errs = [np.linalg.norm(to_table(q) - world[k]) for q in cands]
    res.append(errs)
    print(f"  {k+1:02d}  |      {errs[0]:5.1f}      |          {errs[1]:5.1f}          |          {errs[2]:5.1f}")
print("moyenne :", np.array(res).mean(axis=0).round(1))