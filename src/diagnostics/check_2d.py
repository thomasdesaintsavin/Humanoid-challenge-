import numpy as np, cv2
from scipy.ndimage import label, binary_closing

d = np.load("data/processed/calib_touch.npz")
c = np.load("data/processed/calib.npz")
fps = float(d["fps"]); t = d["t"]
lm = d["landmarks2d"]; ap = d["aperture"] * 100
world = c["world"][:, :2] * 100
Hinv = np.linalg.inv(c["H"])                      # image -> table (m)
R_t, _ = cv2.Rodrigues(c["rvec"]); t_t = c["tvec"].ravel(); K = c["K"]

def fill(x):
    x = x.copy()
    for col in ([x] if x.ndim == 1 else list(x.T)):
        bad = np.isnan(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return x

mid = fill((lm[:, 4] + lm[:, 8]) / 2)             # milieu pouce-index, en pixels
ap = fill(ap)
Pc = fill((R_t @ d["pinch_table"].T).T + t_t)     # point 3D PnP, repère caméra
proj = (K @ Pc.T).T; proj = proj[:, :2] / proj[:, 2:3]   # sa projection en pixels

h = max(1, int(0.1 * fps))
speed = np.full(len(mid), 1e9)
for i in range(h, len(mid) - h):
    speed[i] = np.linalg.norm(mid[i + h] - mid[i - h]) / (2 * h / fps)
mask = binary_closing((speed < 300) & (ap < 4), structure=np.ones(int(0.25 * fps)))
lab, n = label(mask)
segs = [np.where(lab == k)[0] for k in range(1, n + 1)]
segs = [s for s in segs if len(s) >= 0.4 * fps]
segs = sorted(sorted(segs, key=len, reverse=True)[:8], key=lambda s: s[0])

print("croix | image -> table (sans PnP)          | écart PnP vs doigt dans l'image")
errs = []
for k, s in enumerate(segs):
    px = np.median(mid[s], axis=0)
    xy = cv2.perspectiveTransform(px.reshape(1, 1, 2), Hinv).ravel() * 100
    e = np.linalg.norm(xy - world[k]); errs.append(e)
    gap = np.linalg.norm(np.median(proj[s], axis=0) - px)
    print(f"  {k+1:02d}  | dx {xy[0]-world[k,0]:+5.1f} dy {xy[1]-world[k,1]:+5.1f} erreur {e:4.1f} cm | {gap:6.0f} px")
print(f"erreur moyenne image -> table : {np.mean(errs):.1f} cm")