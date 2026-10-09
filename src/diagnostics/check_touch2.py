import numpy as np, cv2
from scipy.ndimage import label, binary_closing

d = np.load("data/processed/calib_touch.npz")
c = np.load("data/processed/calib.npz")
fps = float(d["fps"]); t = d["t"]
lm = d["landmarks2d"]; ap = d["aperture"] * 100
world = c["world"][:, :2] * 100
Hinv = np.linalg.inv(c["H"])
R_t, _ = cv2.Rodrigues(c["rvec"]); t_t = c["tvec"].ravel()

def fill(x):
    x = x.copy()
    for col in ([x] if x.ndim == 1 else list(x.T)):
        bad = np.isnan(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return x

mid = fill((lm[:, 4] + lm[:, 8]) / 2); ap = fill(ap)
Pc = fill((R_t @ d["pinch_table"].T).T + t_t)           # point 3D, repère caméra (m)

h = max(1, int(0.1 * fps))
speed = np.full(len(mid), 1e9)
for i in range(h, len(mid) - h):
    speed[i] = np.linalg.norm(mid[i + h] - mid[i - h]) / (2 * h / fps)
mask = binary_closing((speed < 300) & (ap < 4), structure=np.ones(int(0.25 * fps)))
lab, n = label(mask)
segs = [np.where(lab == k)[0] for k in range(1, n + 1)]
segs = [s for s in segs if len(s) >= 0.4 * fps]
assert len(segs) >= 8, f"seulement {len(segs)} pauses"
segs = sorted(sorted(segs, key=len, reverse=True)[:8], key=lambda s: s[0])   # 8 plus longues = croix 01..08

nz = R_t[:, 2]
print("croix | image->table (sans PnP) | PnP : erreur xy, hauteur | facteur de profondeur")
E1, E2, H3, R = [], [], [], []
for k, s in enumerate(segs):
    px = np.median(mid[s], axis=0)
    xy = cv2.perspectiveTransform(px.reshape(1, 1, 2).astype(np.float64), Hinv).ravel() * 100
    e1 = np.linalg.norm(xy - world[k])
    Pm = np.median(Pc[s], axis=0)
    X = (R_t.T @ (Pm - t_t)) * 100
    e2 = np.linalg.norm(X[:2] - world[k])
    u = Pm / np.linalg.norm(Pm)
    ratio = (nz @ t_t) / (nz @ u) / np.linalg.norm(Pm)
    E1.append(e1); E2.append(e2); H3.append(-X[2]); R.append(ratio)
    print(f"  {k+1:02d}  |        {e1:4.1f} cm        |     {e2:4.1f} cm, {-X[2]:+5.1f} cm     |      {ratio:.2f}")
print(f"moyennes : image->table {np.mean(E1):.1f} cm | PnP xy {np.mean(E2):.1f} cm, hauteur {np.mean(H3):+.1f} cm "
      f"| facteur de profondeur médian {np.median(R):.3f}")
print(f"décalage du marqueur dans cette vidéo : médiane {np.median(d['marker_shift']):.0f} px")