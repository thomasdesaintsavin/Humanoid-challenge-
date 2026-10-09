import numpy as np, cv2, glob, os, re
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- à mesurer à la règle et à remplacer --------------------------------------
OBJ_H_CM = 1.8      # hauteur de l'objet posé sur la table
BOWL_H_CM = 4.5    # hauteur du bol (bord supérieur)
# --- hauteurs SUPPOSEES (pas mesurées sur la vidéo) ---------------------------
H_GRASP = max(1.0, OBJ_H_CM / 2)     # bouts des doigts quand on tient l'objet
H_REL = 13.8                         # estimé par 04_release_height.py (lignes de vue vers le bol connu)
H_CARRY = H_REL                      # hypothèse : la main reste à cette hauteur pendant le transport
H_APPROACH = H_CARRY                 # entrée et sortie de la main dans le cadre

c = np.load("data/processed/calib.npz")
Kinv = np.linalg.inv(c["K"])
R_t, _ = cv2.Rodrigues(c["rvec"]); t_t = c["tvec"].ravel(); nz = R_t[:, 2]
world = c["world"][:, :2] * 100                      # croix 01..08, cm


def fill_short_gaps(x, fps, max_gap_s=0.4):
    x = np.array(x, float); flat = x.ndim == 1
    if flat:
        x = x[:, None]
    n = len(x); valid = ~np.isnan(x).any(axis=1); out = x.copy(); idx = np.arange(n); i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap_s * fps:
            for k in range(x.shape[1]):
                out[i:j, k] = np.interp(idx[i:j], [i - 1, j], [x[i - 1, k], x[j, k]])
        i = j
    return out[:, 0] if flat else out


def longest_run(mask):
    best, i, n = (0, 0), 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best                                      # [début, fin[


def hysteresis(ap, t_close, t_open):
    state = np.zeros(len(ap), bool)
    s = ap[0] < (t_close + t_open) / 2               # True = pince fermée
    for i, a in enumerate(ap):
        if s and a > t_open:
            s = False
        elif (not s) and a < t_close:
            s = True
        state[i] = s
    return state


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def height_profile(n, ig, ir):
    h = np.empty(n); idx = np.arange(n)
    a = idx <= ig                                    # approche : descente vers l'objet
    h[a] = H_APPROACH + (H_GRASP - H_APPROACH) * smooth(idx[a] / max(ig, 1))
    b = (idx > ig) & (idx <= ir)                     # transport : montée, palier, descente vers le bol
    u = (idx[b] - ig) / max(ir - ig, 1)
    h[b] = H_GRASP + (H_CARRY - H_GRASP) * smooth(u / 0.3) + (H_REL - H_CARRY) * smooth((u - 0.7) / 0.3)
    r = idx > ir                                     # retrait de la main
    h[r] = H_REL + (H_APPROACH - H_REL) * smooth((idx[r] - ir) / max(n - 1 - ir, 1))
    return h


def table_xy(pix, h_cm):
    # intersection du rayon de vue de chaque pixel avec le plan situé à h_cm au-dessus de la table
    u = (Kinv @ np.c_[pix, np.ones(len(pix))].T).T
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    lam = (nz @ t_t - np.asarray(h_cm) / 100) / (u @ nz)
    return (((lam[:, None] * u) - t_t) @ R_t)[:, :2] * 100


def process(path):
    stem = os.path.basename(path)[:-4]
    m = re.match(r"(pickplace|heldout)_pos(\d\d)_rep(\d)$", stem)
    pos = int(m.group(2)); nominal = world[pos - 1] if m.group(1) == "pickplace" else None
    d = np.load(path); fps = float(d["fps"])
    ap = fill_short_gaps(d["aperture"] * 100, fps)
    mid = fill_short_gaps((d["landmarks2d"][:, 4] + d["landmarks2d"][:, 8]) / 2, fps)
    valid = ~np.isnan(ap) & ~np.isnan(mid).any(axis=1)
    i0, i1 = longest_run(valid)
    rec = dict(stem=stem, ok=False, flags=[], pos=pos)
    w = int(0.4 * fps) | 1
    if i1 - i0 < max(w + 2, int(1.5 * fps)):
        rec["flags"].append("main détectée trop peu de temps")
        return rec
    ap_s = savgol_filter(ap[i0:i1], w, 2)
    pix = savgol_filter(mid[i0:i1], w, 2, axis=0)
    t = np.arange(i1 - i0) / fps
    lo, hi = np.percentile(ap_s, [5, 95])
    rec.update(t=t, ap=ap_s)
    if hi - lo < 4:
        rec["flags"].append("ouverture quasi constante : pas de geste de pince")
        return rec
    t_close, t_open = lo + 0.40 * (hi - lo), lo + 0.65 * (hi - lo)
    state = hysteresis(ap_s, t_close, t_open)
    ig, ir = longest_run(state)
    rec.update(run=(ig, ir))
    if ir - ig < 0.8 * fps:
        rec["flags"].append("phase fermée trop courte")
        return rec
    if ig == 0 or ir >= len(state):
        rec["flags"].append("pince pas ouverte avant ou après la saisie")
        return rec
    h = height_profile(len(t), ig, ir)
    xy = table_xy(pix, h)
    k = int(0.2 * fps)
    g_xy = np.median(xy[ig:min(ig + k, ir) + 1], axis=0)
    r_xy = np.median(xy[max(ir - k, ig):ir + 1], axis=0)
    err = None if nominal is None else float(np.linalg.norm(g_xy - nominal))
    if err is not None and err > 4:
        rec["flags"].append(f"saisie à {err:.1f} cm de la croix")
    rec.update(ok=True, xy=xy, g_xy=g_xy, r_xy=r_xy, err=err)
    np.savez(f"data/processed/traj_{stem}.npz", t=t, pix=pix, h_cm=h, xy_cm=xy, aperture_cm=ap_s,
             closed=state, t_grasp=t[ig], t_release=t[ir], grasp_xy=g_xy, release_xy=r_xy,
             pos=pos, fps=fps)
    return rec


paths = sorted(glob.glob("data/processed/pickplace_pos*_rep*.npz") + glob.glob("data/processed/heldout_pos*_rep*.npz"))
recs = [process(p) for p in paths]

print(f"{'vidéo':24s} état   saisie  lâcher   erreur saisie")
for r in recs:
    if r["ok"]:
        e = "" if r["err"] is None else f"{r['err']:.1f} cm"
        print(f"{r['stem']:24s} OK    {r['t'][r['run'][0]]:5.1f}s  {r['t'][r['run'][1]]:5.1f}s   {e}  {'; '.join(r['flags'])}")
    else:
        print(f"{r['stem']:24s} A VOIR {'; '.join(r['flags'])}")
ok = [r for r in recs if r["ok"]]
errs = [r["err"] for r in ok if r["err"] is not None]
rel = np.array([r["r_xy"] for r in ok if r["stem"].startswith("pickplace")])
print(f"\n{len(ok)}/{len(recs)} vidéos exploitables")
if errs:
    print(f"erreur de saisie vs croix : moyenne {np.mean(errs):.1f} cm, médiane {np.median(errs):.1f}, max {np.max(errs):.1f}")
if len(rel):
    print(f"point de lâcher : moyenne ({rel[:, 0].mean():.1f}, {rel[:, 1].mean():.1f}) cm, "
          f"écart-type ({rel[:, 0].std():.1f}, {rel[:, 1].std():.1f}) cm")

n = len(recs); cols = 6; rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows), squeeze=False)
for ax in axs.ravel():
    ax.axis("off")
for ax, r in zip(axs.ravel(), recs):
    ax.axis("on")
    if "t" in r:
        ax.plot(r["t"], r["ap"], lw=1)
        if "run" in r:
            ax.axvspan(r["t"][r["run"][0]], r["t"][min(r["run"][1], len(r["t"]) - 1)], color="orange", alpha=0.3)
    ax.set_title(r["stem"].replace("pickplace_", "").replace("_rep", " r") + ("" if r["ok"] else " !"),
                 fontsize=7, color="k" if r["ok"] else "r")
    ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig("results/traj_overview.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(11, 4)); cmap = plt.get_cmap("tab10")
for r in ok:
    if r["stem"].startswith("pickplace"):
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], color=cmap(r["pos"] - 1), lw=0.8, alpha=0.7)
        ax.scatter(*r["r_xy"], color=cmap(r["pos"] - 1), marker="s", s=14)
ax.scatter(world[:, 0], world[:, 1], c="k", marker="x")
for j, wv in enumerate(world):
    ax.text(wv[0] + 0.5, wv[1] + 0.5, f"{j + 1:02d}")
ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
plt.tight_layout(); plt.savefig("results/traj_topdown.png", dpi=110)
print("sauvé : results/traj_overview.png, results/traj_topdown.png, data/processed/traj_*.npz")