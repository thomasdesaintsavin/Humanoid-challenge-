import numpy as np, cv2, glob, os

# --- à remplir (cm) : distances au centre du cercle tracé sous le bol ---------
D4_CM = 20     # depuis la croix 04
D8_CM = 25    # depuis la croix 08
# -------------------------------------------------------------------------------
if D4_CM is None or D8_CM is None:
    raise SystemExit("remplis D4_CM et D8_CM en haut du script")
by = (D4_CM**2 - D8_CM**2 + 100) / 20          # croix 04 = (30, 0), croix 08 = (30, 10)
bx = 30 + np.sqrt(max(D4_CM**2 - by**2, 0))
print(f"centre du bol dans le repère table : ({bx:.1f}, {by:.1f}) cm")
B = np.array([bx, by]) / 100

c = np.load("data/processed/calib.npz")
Kinv = np.linalg.inv(c["K"])
R_t, _ = cv2.Rodrigues(c["rvec"]); t_t = c["tvec"].ravel()
C = -R_t.T @ t_t                                # centre de la caméra, repère table (z vers le bas)

rows = []
for p in sorted(glob.glob("data/processed/traj_pickplace_*.npz")):
    d = np.load(p); fps = float(d["fps"])
    ir = int(round(float(d["t_release"]) * fps)); k = int(0.2 * fps)
    px = np.median(d["pix"][max(ir - k, 0):ir + 1], axis=0)    # pixel de la pince au lâcher
    u = Kinv @ np.array([px[0], px[1], 1.0]); u /= np.linalg.norm(u)
    dv = R_t.T @ u                                              # direction du rayon, repère table
    lam = ((B - C[:2]) @ dv[:2]) / (dv[:2] @ dv[:2])            # point du rayon le plus proche du bol
    pt = C + lam * dv
    rows.append((os.path.basename(p)[5:-4], -pt[2] * 100, np.linalg.norm(pt[:2] - B) * 100))

h = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
print(f"{len(rows)} démos | hauteur de lâcher estimée : médiane {np.median(h):.1f} cm, "
      f"écart-type {h.std():.1f}, min {h.min():.1f}, max {h.max():.1f}")
print(f"écart horizontal restant au centre du bol : médiane {np.median(m):.1f} cm, max {m.max():.1f}")
for n, hh, mm in rows:
    if mm > 6 or hh < 3:
        print("  à regarder :", n, f"hauteur {hh:.1f} cm, écart {mm:.1f} cm")