import cv2, glob, os

# Détecteur ArUco (dictionnaire 4x4, comme marker0.png)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

# Toutes les vidéos de data/raw (mp4 et mov, majuscules ou minuscules)
paths = []
for ext in ("mp4", "MP4", "mov", "MOV"):
    paths += glob.glob(f"data/raw/**/*.{ext}", recursive=True)

if not paths:
    print("Aucune vidéo trouvée dans data/raw")

for path in sorted(paths):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    if not ok:
        print("ILLISIBLE :", path)
        continue
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = frame.shape[:2]
    corners, ids, _ = detector.detectMarkers(frame)
    marker = "marqueur OK" if ids is not None and 0 in ids.flatten() else "MARQUEUR NON DETECTE"
    print(f"{os.path.basename(path):35s} {w}x{h} {fps:.0f}fps {n/fps:.1f}s  {marker}")