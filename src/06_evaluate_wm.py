"""Evaluate the trained world model(s) on the held-out start positions (03 and 06).

Needs data/processed/wm_data.npz and checkpoints/wm_act.pt (wm_noact.pt is optional).
Writes results/wm_curves.png, results/rollout_carry.png and results/rollout_carry.gif.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

os.makedirs("results", exist_ok=True)
D = np.load("data/processed/wm_data.npz")
frames = torch.from_numpy(D["frames"])          # (N, S, S, 3) uint8
hand = torch.from_numpy(D["hand"])              # (N, 4): pinch x, pinch y, opening, hand present
ep_start, ep_pos = D["ep_start"], D["ep_pos"]
names = [str(n) for n in D["names"]]
N, S = frames.shape[0], frames.shape[1]
ep_end = np.r_[ep_start[1:], N]
te = np.where(np.array([p in {3, 6} for p in ep_pos]))[0]   # test videos: positions 03 and 06
print(f"{N} frames {S}x{S}, {len(te)} test videos")

SIGMA = 2.5 * S / 64
yy, xx = torch.meshgrid(torch.arange(S).float(), torch.arange(S).float(), indexing="ij")


def heat(h):
    gx, gy = (h[:, 0] * S).view(-1, 1, 1), (h[:, 1] * S).view(-1, 1, 1)
    g = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * SIGMA ** 2))
    return (g * h[:, 3].view(-1, 1, 1)).unsqueeze(1)


def cond(h0, h1, use_action=True):
    B = h0.shape[0]
    if not use_action:
        return torch.zeros(B, 4, S, S)
    a0 = (h0[:, 2] * h0[:, 3]).view(B, 1, 1, 1).expand(B, 1, S, S)
    a1 = (h1[:, 2] * h1[:, 3]).view(B, 1, 1, 1).expand(B, 1, S, S)
    return torch.cat([heat(h0), heat(h1), a0, a1], 1)


def get(idx, K):
    ar = idx.view(-1, 1) + torch.arange(K + 1).view(1, -1)
    return frames[ar].permute(0, 1, 4, 2, 3).float() / 255, hand[ar]


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class WM(nn.Module):
    """Small U-Net that predicts the change from the current frame, given the hand maps."""

    def __init__(s, c=32):
        super().__init__()
        s.e1, s.e2, s.e3, s.mid = block(7, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 4 * c)
        s.d2, s.d1 = block(6 * c, 2 * c), block(3 * c, c)
        s.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(s, x, cd):
        a = s.e1(torch.cat([x, cd], 1))
        b = s.e2(F.avg_pool2d(a, 2))
        m = s.mid(s.e3(F.avg_pool2d(b, 2)))
        u = s.d2(torch.cat([F.interpolate(m, scale_factor=2), b], 1))
        u = s.d1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return (x + s.out(u)).clamp(0, 1)


def rollout(model, x0, hseq, use_action=True):
    x, out = x0, []
    for k in range(hseq.shape[1] - 1):
        x = model(x, cond(hseq[:, k], hseq[:, k + 1], use_action))
        out.append(x)
    return torch.stack(out, 1)


@torch.no_grad()
def evaluate(model, use_action=True, swap=False, H=20, stride=5):
    fm, fc, nm, nc, dn = [], [], 0, 0, 0
    for j, e in enumerate(te):
        e2 = te[(j + 1) % len(te)]
        L, L2 = ep_end[e] - ep_start[e], ep_end[e2] - ep_start[e2]
        ts = np.arange(0, L - H, stride)
        f, h = get(torch.tensor(ep_start[e] + ts), H)
        if swap:  # use the hand of another test video
            _, h = get(torch.tensor(ep_start[e2] + np.minimum(ts, L2 - H - 1)), H)
        p = rollout(model, f[:, 0], h, use_action)
        tgt = f[:, 1:]
        cp = f[:, :1].expand_as(tgt)
        mov = ((tgt - f[:, :1]).abs().mean(2) > 0.04).float()   # changing pixels (measurement only)
        dm, dc = (p - tgt).abs().mean(2), (cp - tgt).abs().mean(2)
        fm.append(dm.mean((2, 3)))
        fc.append(dc.mean((2, 3)))
        nm = nm + (dm * mov).sum((0, 2, 3))
        nc = nc + (dc * mov).sum((0, 2, 3))
        dn = dn + mov.sum((0, 2, 3))
    d = dn.clamp(min=1)
    return (torch.cat(fm).mean(0).numpy(), torch.cat(fc).mean(0).numpy(),
            (nm / d).numpy(), (nc / d).numpy())


def report(m_act, m_noact):
    res = {"model (real hand)": evaluate(m_act),
           "model (hand from another video)": evaluate(m_act, swap=True)}
    if m_noact is not None:
        res["model without action"] = evaluate(m_noact, use_action=False)
    cf, cm = res["model (real hand)"][1], res["model (real hand)"][3]
    t = (np.arange(20) + 1) * 0.1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, (a, _, b, _) in res.items():
        ax[0].plot(t, a, label=name)
        ax[1].plot(t, b, label=name)
    ax[0].plot(t, cf, "k--", label="copy the last frame")
    ax[1].plot(t, cm, "k--")
    ax[0].set_title("whole image")
    ax[1].set_title("changing pixels only")
    for a in ax:
        a.set_xlabel("prediction horizon (s)")
        a.set_ylabel("mean absolute error")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/wm_curves.png", dpi=120)
    for k in (5, 10, 20):
        row = " | ".join(f"{n} {v[2][k - 1]:.4f}" for n, v in res.items())
        print(f"horizon {k / 10:.1f} s, changing pixels: {row} | copy-last {cm[k - 1]:.4f}")
    for k in (5, 10, 20):
        row = " | ".join(f"{n} {v[0][k - 1]:.4f}" for n, v in res.items())
        print(f"horizon {k / 10:.1f} s, whole image: {row} | copy-last {cf[k - 1]:.4f}")
    w = np.where(res["model (real hand)"][2] >= cm)[0]
    if len(w) == 0:
        print("trust horizon (model better than copy-last on changing pixels): beyond 2 s")
    elif w[0] == 0:
        print("trust horizon (model better than copy-last on changing pixels): none")
    else:
        print(f"trust horizon (model better than copy-last on changing pixels): {w[0] * 0.1:.1f} s")


def green_centroid(x):
    # x: (B, T, 3, S, S) in [0, 1] -> centroid (B, T, 2) in pixels, found (B, T)
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    m = ((g - torch.maximum(r, b)) > 0.07) & (g > 0.20)
    n = m.sum((2, 3)).float()
    cx = (m.float() * xx).sum((2, 3)) / n.clamp(min=1)
    cy = (m.float() * yy).sum((2, 3)) / n.clamp(min=1)
    return torch.stack([cx, cy], -1), n >= 3


@torch.no_grad()
def best_t0(e, H=20, step=2):
    # window in which the green box moves the most (grasp and carry)
    L = ep_end[e] - ep_start[e]
    best, best_d = 0, -1.0
    for t0 in range(0, L - H - 1, step):
        idx = torch.tensor([ep_start[e] + t0, ep_start[e] + t0 + H])
        x = frames[idx].permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
        c, ok = green_centroid(x)
        if ok.all():
            d = (c[0, 1] - c[0, 0]).norm().item()
            if d > best_d:
                best, best_d = t0, d
    return best


@torch.no_grad()
def make_rollout(model, e, e2, t0, H=20, tag="rollout_carry"):
    L2 = ep_end[e2] - ep_start[e2]
    f, h = get(torch.tensor([ep_start[e] + t0]), H)
    _, h2 = get(torch.tensor([ep_start[e2] + min(t0, L2 - H - 1)]), H)
    real = f[0, 1:]
    pred = rollout(model, f[:, 0], h)[0]
    swap = rollout(model, f[:, 0], h2)[0]

    def up(x):
        x = F.interpolate(x, scale_factor=4, mode="nearest").clamp(0, 1)
        return (x * 255).byte().permute(0, 2, 3, 1).numpy()

    R, P, W = up(real), up(pred), up(swap)
    cols = (4, 9, 14, 19)  # 0.5 s, 1 s, 1.5 s, 2 s
    strip = Image.fromarray(np.vstack([np.hstack([X[k] for k in cols]) for X in (R, P, W)]))
    dr = ImageDraw.Draw(strip)
    for r_, lab in enumerate(["real", "prediction", "prediction with the hand of another video"]):
        dr.text((4, r_ * S * 4 + 4), lab, fill=(255, 255, 0))
    strip.save(f"results/{tag}.png")
    ims = []
    for k in range(H):
        im = Image.fromarray(np.hstack([R[k], P[k], W[k]]))
        dr = ImageDraw.Draw(im)
        for c_, lab in enumerate(["real", "prediction", "other hand"]):
            dr.text((c_ * S * 4 + 4, 4), lab, fill=(255, 255, 0))
        ims.append(im)
    ims[0].save(f"results/{tag}.gif", save_all=True, append_images=ims[1:], duration=200, loop=0)
    print(f"{names[e]}, window starts at frame {t0}; third row uses the hand from {names[e2]}")


def load(path):
    m = WM()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


m_act = load("checkpoints/wm_act.pt")
m_noact = load("checkpoints/wm_noact.pt") if os.path.exists("checkpoints/wm_noact.pt") else None
print("model without action:", "found" if m_noact is not None else "not found (skipped)")
report(m_act, m_noact)
e = te[0]
make_rollout(m_act, e, te[len(te) // 2], best_t0(e))
print("saved: results/wm_curves.png, results/rollout_carry.png, results/rollout_carry.gif")