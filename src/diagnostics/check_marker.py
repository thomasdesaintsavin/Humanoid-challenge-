import cv2, glob, os, sys

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())
paths = sys.argv[1:] or sorted(glob.glob("data/raw/train/*.MOV") + glob.glob("data/raw/heldout/*.MOV"))
for p in paths:
    cap = cv2.VideoCapture(p)
    tested = found = i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i % 3:
            continue
        s = 1080 / f.shape[1]
        g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        _, ids, _ = det.detectMarkers(g)
        tested += 1
        found += int(ids is not None and 0 in ids.flatten())
    frac = found / max(tested, 1)
    print(f"{os.path.basename(p):34s} marqueur détecté sur {100*frac:3.0f} %" + ("   <-- PROBLEME" if frac < 0.9 else ""))