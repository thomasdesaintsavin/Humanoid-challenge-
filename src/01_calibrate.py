import cv2, numpy as np, glob, sys

VIDEO = glob.glob("data/raw/calib/calib_marker.*")[0]
FRAME = 30          # index of the frame used for calibration
F_FACTOR = 0.78     # approximate focal length = F_FACTOR * longest image side (to be validated)
STEP_CM = 10.0      # spacing between neighbouring crosses

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ok, img = cap.read()
assert ok, "unreadable video"
H_img, W_img = img.shape[:2]
print("image", W_img, "x", H_img)

# Real positions of the crosses in metres, table frame (z = 0):
# crosses 01..04 on the line y = 0, crosses 05..08 on the line y = 0.10
world = np.array([[(n % 4) * STEP_CM / 100, (n // 4) * STEP_CM / 100, 0.0] for n in range(8)])

SCALE = 1000 / max(W_img, H_img)
small = cv2.resize(img, None, fx=SCALE, fy=SCALE)
cv2.namedWindow("calib")


def click(image):
    pts = []

    def cb(e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.setMouseCallback("calib", cb)
    cv2.imshow("calib", image)
    while not pts:
        if cv2.waitKey(20) == 27:
            sys.exit("cancelled")
    return pts[0]


# Two clicks per cross: a rough click, then a precise click on a zoomed crop
image_pts = []
for n in range(8):
    disp = small.copy()
    for px, py in image_pts:
        cv2.circle(disp, (int(px * SCALE), int(py * SCALE)), 4, (0, 255, 0), -1)
    cv2.putText(disp, f"click cross {n+1:02d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    x, y = click(disp)
    cx, cy = x / SCALE, y / SCALE
    R = 60
    x0, y0 = int(max(cx - R, 0)), int(max(cy - R, 0))
    zoom = cv2.resize(img[y0:y0 + 2 * R, x0:x0 + 2 * R], None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    cv2.putText(zoom, f"cross {n+1:02d}: click the centre", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    zx, zy = click(zoom)
    image_pts.append((x0 + zx / 8, y0 + zy / 8))
cv2.destroyAllWindows()
image_pts = np.array(image_pts, np.float64)

# 1) Table -> image homography: enough for positions on the table (no focal length needed)
Hm, _ = cv2.findHomography(world[:, :2], image_pts)
back = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(-1, 2)
err_cm = np.linalg.norm(back - world[:, :2], axis=1) * 100
print("grid error (cm) per cross:", np.round(err_cm, 2), "| mean", round(err_cm.mean(), 2))

# 2) Camera pose (approximate focal length), used for the height of the hand
f = F_FACTOR * max(W_img, H_img)
K = np.array([[f, 0, W_img / 2], [0, f, H_img / 2], [0, 0, 1]], np.float64)
ok, rvec, tvec = cv2.solvePnP(world, image_pts, K, None)
proj, _ = cv2.projectPoints(world, rvec, tvec, K, None)
proj = proj.reshape(-1, 2)
rms = np.sqrt(np.mean(np.sum((proj - image_pts) ** 2, axis=1)))
print(f"pose: reprojection error {rms:.1f} px | camera-to-origin distance {np.linalg.norm(tvec)*100:.0f} cm")

# 3) Marker position: reference used to detect a camera displacement
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
corners, ids, _ = det.detectMarkers(img)
if ids is not None and 0 in ids.flatten():
    marker = corners[list(ids.flatten()).index(0)].reshape(4, 2)
    print("marker: OK")
else:
    marker = np.zeros((4, 2))
    print("marker: NOT DETECTED")

np.savez("data/processed/calib.npz", K=K, rvec=rvec, tvec=tvec, H=Hm, world=world,
         image_pts=image_pts, marker=marker, size=(W_img, H_img), f_factor=F_FACTOR)

# Check image: green = your clicks, red = recomputed positions
vis = img.copy()
for p, q in zip(image_pts, proj):
    cv2.circle(vis, (int(p[0]), int(p[1])), 12, (0, 255, 0), 3)
    cv2.circle(vis, (int(q[0]), int(q[1])), 6, (0, 0, 255), -1)
cv2.imwrite("results/calib_check.png", cv2.resize(vis, None, fx=SCALE, fy=SCALE))
print("saved: data/processed/calib.npz and results/calib_check.png")