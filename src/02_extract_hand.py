import cv2, numpy as np, mediapipe as mp, sys, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = "models/hand_landmarker.task"
DET_W = 1080          # largeur utilisée pour détecter (plus rapide qu'en 4K)
HAND_SCALE = 1.0      # pas de correction d'échelle pour l'instant

c = np.load("data/processed/calib.npz")
K, rvec_t, tvec_t = c["K"], c["rvec"], c["tvec"].ravel()
REF_MARKER = c["marker"].astype(np.float32)     # coins du marqueur dans l'image de référence
R_t, _ = cv2.Rodrigues(rvec_t)                  # table -> caméra
aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                cv2.aruco.DetectorParameters())


def run(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lm_raw, wl, mk = [], [], []
    writer = None
    with vision.HandLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            s = DET_W / W
            small = cv2.resize(frame, None, fx=s, fy=s)
            # marqueur (coins en pixels de l'image pleine résolution)
            corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            m = np.full((4, 2), np.nan)
            if ids is not None and 0 in ids.flatten():
                m = corners[list(ids.flatten()).index(0)].reshape(4, 2) / s
            # main
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(i * 1000 / fps))
            p2 = np.full((21, 2), np.nan); w3 = np.full((21, 3), np.nan)
            if res.hand_landmarks:
                p2 = np.array([[l.x * W, l.y * H] for l in res.hand_landmarks[0]])
                w3 = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]])
            lm_raw.append(p2); wl.append(w3); mk.append(m)
            # vidéo de contrôle : points de la main (vert) et marqueur (bleu)
            vis = small.copy()
            for x, y in p2 * s:
                if not np.isnan(x):
                    cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
            if not np.isnan(m).any():
                cv2.polylines(vis, [(m * s).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), 3)
            if writer is None:
                os.makedirs("results", exist_ok=True)
                writer = cv2.VideoWriter(f"results/{stem}_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            i += 1
    writer.release()

    lm_raw = np.array(lm_raw); wl = np.array(wl); mk = np.array(mk)
    N = len(lm_raw); t = np.arange(N) / fps

    # 1) marqueur par image ; images sans détection : interpolation (valeur tenue aux extrémités)
    good = ~np.isnan(mk).any(axis=(1, 2))
    print(f"{stem}: marqueur détecté sur {good.mean()*100:.0f} % des images")
    if good.sum() == 0:
        raise SystemExit("marqueur jamais détecté : impossible de recaler cette vidéo")
    mkf = mk.reshape(N, 8).copy()
    for k in range(8):
        mkf[:, k] = np.interp(t, t[good], mkf[good, k])
    mkf = mkf.reshape(N, 4, 2)

    # 2) recalage de chaque image sur la référence (translation + rotation + échelle)
    lm_reg = np.full_like(lm_raw, np.nan)
    shift = np.zeros(N)
    for i in range(N):
        A, _ = cv2.estimateAffinePartial2D(mkf[i].astype(np.float32), REF_MARKER, method=cv2.LMEDS)
        if A is None:
            A = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        lm_reg[i] = lm_raw[i] @ A[:, :2].T + A[:, 2]
        shift[i] = np.linalg.norm(mkf[i].mean(0) - REF_MARKER.mean(0))
    print(f"  décalage du marqueur vs référence : médiane {np.median(shift):.0f} px, "
          f"variation pendant la vidéo {shift.max()-shift.min():.0f} px")

    # 3) pose 3D de la main (PnP) avec les points recalés
    pinch = np.full((N, 3), np.nan); aper = np.full(N, np.nan)
    for i in range(N):
        if np.isnan(lm_reg[i]).any():
            continue
        w3, p2 = wl[i], lm_reg[i]
        aper[i] = np.linalg.norm(w3[4] - w3[8])
        ok_, rv, tv = cv2.solvePnP(w3, p2, K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok_:
            continue
        rv, tv = cv2.solvePnPRefineLM(w3, p2, K, None, rv, tv)
        if tv[2] <= 0:
            continue
        Rh, _ = cv2.Rodrigues(rv)
        p_cam = (Rh @ ((w3[4] + w3[8]) / 2) + tv.ravel()) * HAND_SCALE
        pinch[i] = R_t.T @ (p_cam - tvec_t)       # repère table
    height = -pinch[:, 2]                         # z de la table pointe vers le bas

    np.savez(f"data/processed/{stem}.npz", t=t, pinch_table=pinch, aperture=aper,
             landmarks2d=lm_reg, landmarks2d_raw=lm_raw, marker_shift=shift, fps=fps)

    det = ~np.isnan(aper)
    print(f"  main détectée sur {det.mean()*100:.0f} % des {N} images")
    if det.any():
        print(f"  ouverture : min {np.nanmin(aper)*100:.1f} cm, max {np.nanmax(aper)*100:.1f} cm")
        print(f"  hauteur du point de pince : min {np.nanmin(height)*100:.1f} cm, max {np.nanmax(height)*100:.1f} cm")

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, aper * 100); ax[0].set_ylabel("ouverture (cm)")
    ax[1].plot(t, height * 100); ax[1].set_ylabel("hauteur (cm)")
    ax[2].plot(t, shift); ax[2].set_ylabel("décalage marqueur (px)"); ax[2].set_xlabel("temps (s)")
    plt.tight_layout(); plt.savefig(f"results/{stem}_curves.png", dpi=120); plt.close()


paths = sys.argv[1:] or sorted(glob.glob("data/raw/train/*.MOV") + glob.glob("data/raw/heldout/*.MOV"))
for p in paths:
    run(p)