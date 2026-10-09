import numpy as np, cv2, glob
from scipy.ndimage import label, binary_closing

d = np.load("data/processed/calib_touch.npz")
c = np.load("data/processed/calib.npz")
fps = float(d["fps"]); t = d["t"]
lm = d["landmarks2d"]                    # (N, 21, 2) en pixels
ap = d["aperture"] * 100                 # cm
cross_px = c["image_pts"]                # les 8 croix cliquées, en pixels
mid = (lm[:, 4] + lm[:, 8]) / 2          # milieu pouce-index, en pixels

for k in range(2):                       # combler les images sans détection
    bad = np.isnan(mid[:, k])
    if bad.any() and (~bad).any():
        mid[bad, k] = np.interp(t[bad], t[~bad], mid[~bad, k])
bad = np.isnan(ap)
if bad.any() and (~bad).any():
    ap[bad] = np.interp(t[bad], t[~bad], ap[~bad])

# pauses : point presque immobile (en pixels) et pince fermée
h = max(1, int(0.1 * fps))
speed = np.full(len(mid), 1e9)
for i in range(h, len(mid) - h):
    speed[i] = np.linalg.norm(mid[i + h] - mid[i - h]) / (2 * h / fps)
mask = (speed < 300) & (ap < 4)
mask = binary_closing(mask, structure=np.ones(int(0.25 * fps)))   # recolle les petits trous
lab, n = label(mask)
segs = [np.where(lab == k)[0] for k in range(1, n + 1)]
segs = [s for s in segs if len(s) >= 0.4 * fps]
print(len(segs), "pauses détectées :")
for k, s in enumerate(segs):
    print(f"  p{k+1}: {t[s[0]]:.1f}s -> {t[s[-1]]:.1f}s")

video = glob.glob("data/raw/calib/calib_touch.*")[0]
cap = cv2.VideoCapture(video)
HALF, S = 450, 0.5                       # vignette de 900 px, affichée à moitié
tiles = []
for k, s in enumerate(segs):
    idx = int(s[len(s) // 2])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    if not ok:
        continue
    Himg, Wimg = img.shape[:2]
    cx, cy = np.median(mid[s], axis=0)
    x0, y0 = int(cx - HALF), int(cy - HALF)
    tile = np.zeros((2 * HALF, 2 * HALF, 3), np.uint8)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + 2 * HALF, Wimg), min(y0 + 2 * HALF, Himg)
    tile[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    tile = cv2.resize(tile, None, fx=S, fy=S)
    for j, (px, py) in enumerate(cross_px):          # croix : ronds rouges
        u, v = (px - x0) * S, (py - y0) * S
        if 0 <= u < tile.shape[1] and 0 <= v < tile.shape[0]:
            cv2.circle(tile, (int(u), int(v)), 14, (0, 0, 255), 2)
            cv2.putText(tile, f"{j+1:02d}", (int(u) + 16, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for j in (4, 8):                                  # pointes pouce / index : points verts
        u, v = (lm[idx, j, 0] - x0) * S, (lm[idx, j, 1] - y0) * S
        cv2.circle(tile, (int(u), int(v)), 5, (0, 255, 0), -1)
    cv2.putText(tile, f"p{k+1} t={t[idx]:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(tile)

cols = 3
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite("results/calib_touch_crops.png", np.vstack(rows))
print("sauvé : results/calib_touch_crops.png")