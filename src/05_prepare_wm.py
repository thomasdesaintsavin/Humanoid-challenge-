import cv2, numpy as np, glob, os

OUT = 64            # image side in pixels (the crop is slightly stretched, which does not bother the model)
STEP = 3            # keep 1 frame out of 3: 30 fps -> 10 fps
MARGIN = 500        # margin around the scene, in 4K pixels

c = np.load("data/processed/calib.npz")
H, REF, cross = c["H"], c["marker"].astype(np.float32), c["image_pts"]
W_IMG, H_IMG = [int(v) for v in c["size"]]
bowl_px = cv2.perspectiveTransform(np.float32([[[0.490, -0.062]]]), H).reshape(2)   # bowl centre (49, -6.2 cm)
pts = np.vstack([cross, bowl_px])
MARGIN_TOP = 1000      # the hand rises above the bowl: more room at the top
x0, y0 = np.maximum((pts.min(0) - [MARGIN, MARGIN_TOP]).astype(int), 0)
x1, y1 = np.minimum((pts.max(0) + MARGIN).astype(int), [W_IMG, H_IMG])
RW, RH = int(x1 - x0), int(y1 - y0)
print(f"crop zone: x {x0}-{x1}, y {y0}-{y1} ({RW}x{RH} px)")

aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                cv2.aruco.DetectorParameters())

def video_affine(cap, n):
    # similarity transform from this video's marker position to the reference view
    found = []
    for i in np.linspace(0, n - 1, 15).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        s = 1080 / f.shape[1]
        g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(g)
        if ids is not None and 0 in ids.flatten():
            found.append(corners[list(ids.flatten()).index(0)].reshape(4, 2) / s)
    if not found:
        return None
    A, _ = cv2.estimateAffinePartial2D(np.median(found, axis=0).astype(np.float32), REF, method=cv2.LMEDS)
    return A

stems = sorted(os.path.basename(p)[5:-4] for p in glob.glob("data/processed/traj_pickplace_*.npz"))
frames, hand, ep_start, ep_pos, names = [], [], [], [], []
for stem in stems:
    d = np.load(f"data/processed/{stem}.npz")
    cap = cv2.VideoCapture(glob.glob(f"data/raw/train/{stem}.*")[0])
    A = video_affine(cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if A is None:
        print("marker not found, video skipped:", stem)
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    M = A.copy(); M[0, 2] -= x0; M[1, 2] -= y0          # image -> reference view -> crop zone
    lm, ap = d["landmarks2d"], d["aperture"]            # already registered to the reference view
    ep_start.append(len(frames)); ep_pos.append(int(stem.split("_pos")[1][:2])); names.append(stem)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STEP == 0 and i < len(ap):
            roi = cv2.warpAffine(f, M, (RW, RH), flags=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(cv2.resize(roi, (OUT, OUT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            pinch = (lm[i, 4] + lm[i, 8]) / 2
            present = float(not np.isnan(pinch).any() and not np.isnan(ap[i]))
            # hand state: pinch x, pinch y (normalised to the crop), opening / 15 cm, hand-present flag
            hand.append([(pinch[0] - x0) / RW, (pinch[1] - y0) / RH, ap[i] / 0.15, 1.0] if present else [0, 0, 0, 0])
        i += 1
    print(f"{stem}: {len(frames) - ep_start[-1]} frames", flush=True)

frames = np.array(frames, np.uint8); hand = np.array(hand, np.float32)
np.savez_compressed("data/processed/wm_data.npz", frames=frames, hand=hand, ep_start=np.array(ep_start),
                    ep_pos=np.array(ep_pos), names=np.array(names), roi=np.array([x0, y0, x1, y1]), step=STEP)
print(f"\n{len(names)} videos, {len(frames)} frames, hand present in {hand[:, 3].mean()*100:.0f} % | "
      f"{os.path.getsize('data/processed/wm_data.npz') / 1e6:.0f} MB")

# preview: 8 frames of the first video, enlarged 4x, with the pinch point in red
s0, s1 = ep_start[0], (ep_start[1] if len(ep_start) > 1 else len(frames))
tiles = []
for k in np.linspace(s0, s1 - 1, 8).astype(int):
    t = cv2.resize(frames[k], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    if hand[k, 3] > 0:
        cv2.circle(t, (int(hand[k, 0] * OUT * 4), int(hand[k, 1] * OUT * 4)), 6, (255, 0, 0), -1)
    tiles.append(t)
sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite("results/wm_preview.png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print("preview: results/wm_preview.png")