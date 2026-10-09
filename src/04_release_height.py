import numpy as np, cv2, glob, os

# to fill in (cm): distances to the centre of the circle drawn under the bowl
D4_CM = 20     # from cross 04
D8_CM = 25     # from cross 08

if D4_CM is None or D8_CM is None:
    raise SystemExit("fill in D4_CM and D8_CM at the top of the script")
by = (D4_CM**2 - D8_CM**2 + 100) / 20          # cross 04 = (30, 0), cross 08 = (30, 10)
bx = 30 + np.sqrt(max(D4_CM**2 - by**2, 0))
print(f"bowl centre in the table frame: ({bx:.1f}, {by:.1f}) cm")
B = np.array([bx, by]) / 100

c = np.load("data/processed/calib.npz")
Kinv = np.linalg.inv(c["K"])
R_t, _ = cv2.Rodrigues(c["rvec"]); t_t = c["tvec"].ravel()
C = -R_t.T @ t_t                                # camera centre, table frame (z axis points down)

rows = []
for p in sorted(glob.glob("data/processed/traj_pickplace_*.npz")):
    d = np.load(p); fps = float(d["fps"])
    ir = int(round(float(d["t_release"]) * fps)); k = int(0.2 * fps)
    px = np.median(d["pix"][max(ir - k, 0):ir + 1], axis=0)    # pixel of the pinch point at release
    u = Kinv @ np.array([px[0], px[1], 1.0]); u /= np.linalg.norm(u)
    dv = R_t.T @ u                                              # ray direction, table frame
    lam = ((B - C[:2]) @ dv[:2]) / (dv[:2] @ dv[:2])            # point of the ray closest to the bowl
    pt = C + lam * dv
    rows.append((os.path.basename(p)[5:-4], -pt[2] * 100, np.linalg.norm(pt[:2] - B) * 100))

h = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
print(f"{len(rows)} demos | estimated release height: median {np.median(h):.1f} cm, "
      f"std {h.std():.1f}, min {h.min():.1f}, max {h.max():.1f}")
print(f"remaining horizontal gap to the bowl centre: median {np.median(m):.1f} cm, max {m.max():.1f}")
for n, hh, mm in rows:
    if mm > 6 or hh < 3:
        print("  to check:", n, f"height {hh:.1f} cm, gap {mm:.1f} cm")