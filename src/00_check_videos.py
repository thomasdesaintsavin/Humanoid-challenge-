import cv2, glob, os

# ArUco detector (4x4 dictionary, same as marker0.png)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

# All videos under data/raw (mp4 and mov, upper or lower case)
paths = []
for ext in ("mp4", "MP4", "mov", "MOV"):
    paths += glob.glob(f"data/raw/**/*.{ext}", recursive=True)

if not paths:
    print("No video found in data/raw")

for path in sorted(paths):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    if not ok:
        print("UNREADABLE:", path)
        continue
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = frame.shape[:2]
    corners, ids, _ = detector.detectMarkers(frame)
    marker = "marker OK" if ids is not None and 0 in ids.flatten() else "MARKER NOT DETECTED"
    print(f"{os.path.basename(path):35s} {w}x{h} {fps:.0f}fps {n/fps:.1f}s  {marker}")