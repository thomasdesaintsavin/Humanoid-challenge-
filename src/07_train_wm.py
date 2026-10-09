"""Train the action-conditioned world model (with the hand, and without it).

Reads data/processed/wm_data.npz and writes <out>/wm_act.pt and <out>/wm_noact.pt.
A GPU is strongly recommended: the reported runs took about 24 minutes each
(4000 steps) on a Colab T4. Results are not bit-identical to the reported ones.

Quick smoke test on CPU:
  python src/07_train_wm.py --steps 20 --bs 8 --models act --out /tmp/wm_smoke
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/wm_data.npz")
ap.add_argument("--out", default="checkpoints_retrain")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--models", choices=["act", "noact", "both"], default="both")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load(args.data)
frames = torch.from_numpy(D["frames"]).to(dev)   # (N, S, S, 3) uint8
hand = torch.from_numpy(D["hand"]).to(dev)       # (N, 4): x, y, opening, present
ep_start, ep_pos = D["ep_start"], D["ep_pos"]
names = [str(n) for n in D["names"]]
N, S = frames.shape[0], frames.shape[1]
ep_end = np.r_[ep_start[1:], N]
rep = np.array([int(n[-1]) for n in names])
is_test = np.array([p in {3, 6} for p in ep_pos])   # never seen in training
is_val = (~is_test) & (rep == 5)
is_train = (~is_test) & (~is_val)
print(f"device: {dev} | {N} frames {S}x{S} | videos: {is_train.sum()} train, "
      f"{is_val.sum()} val, {is_test.sum()} test")

SIGMA = 2.5 * S / 64
ar_ = torch.arange(S, device=dev).float()
yy, xx = torch.meshgrid(ar_, ar_, indexing="ij")


def heat(h):  # Gaussian blob at the pinch position (zero if the hand is absent)
    gx, gy = (h[:, 0] * S).view(-1, 1, 1), (h[:, 1] * S).view(-1, 1, 1)
    g = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * SIGMA ** 2))
    return (g * h[:, 3].view(-1, 1, 1)).unsqueeze(1)


def cond(h0, h1, use_action=True):  # pinch now, pinch next, opening now, opening next
    B = h0.shape[0]
    if not use_action:
        return torch.zeros(B, 4, S, S, device=dev)
    a0 = (h0[:, 2] * h0[:, 3]).view(B, 1, 1, 1).expand(B, 1, S, S)
    a1 = (h1[:, 2] * h1[:, 3]).view(B, 1, 1, 1).expand(B, 1, S, S)
    return torch.cat([heat(h0), heat(h1), a0, a1], 1)


def get(idx, K):  # K+1 consecutive frames and the matching hand states
    ar = idx.view(-1, 1) + torch.arange(K + 1, device=dev).view(1, -1)
    return frames[ar].permute(0, 1, 4, 2, 3).float() / 255, hand[ar]


def starts(mask, K, stride=1):  # possible window starts inside the selected videos
    out = []
    for e in np.where(mask)[0]:
        out += list(range(ep_start[e], ep_end[e] - K, stride))
    return torch.tensor(out, device=dev)


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
        nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class WM(nn.Module):
    """Small U-Net that predicts the change from the current frame."""

    def __init__(s, c=32):
        super().__init__()
        s.e1, s.e2 = block(7, c), block(c, 2 * c)
        s.e3, s.mid = block(2 * c, 4 * c), block(4 * c, 4 * c)
        s.d2, s.d1 = block(6 * c, 2 * c), block(3 * c, c)
        s.out = nn.Conv2d(c, 3, 3, padding=1)
        nn.init.zeros_(s.out.weight)   # starts as "copy the last frame"
        nn.init.zeros_(s.out.bias)

    def forward(s, x, cd):
        a = s.e1(torch.cat([x, cd], 1))
        b = s.e2(F.avg_pool2d(a, 2))
        m = s.mid(s.e3(F.avg_pool2d(b, 2)))
        u = s.d2(torch.cat([F.interpolate(m, scale_factor=2), b], 1))
        u = s.d1(torch.cat([F.interpolate(u, scale_factor=2), a], 1))
        return (x + s.out(u)).clamp(0, 1)


def rollout(model, x0, hseq, use_action=True):  # feeds the model its own outputs
    x, out = x0, []
    for k in range(hseq.shape[1] - 1):
        x = model(x, cond(hseq[:, k], hseq[:, k + 1], use_action))
        out.append(x)
    return torch.stack(out, 1)


def loss_fn(pred, tgt, prev):  # L1, pixels that change weighted 6x
    mov = ((tgt - prev).abs().mean(2, keepdim=True) > 0.04).float()
    return ((1 + 5 * mov) * (pred - tgt).abs()).mean()


def train(use_action, tag, steps, bs, K=4, lr=1e-3):
    torch.manual_seed(0)
    model = WM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    tr, va = starts(is_train, K), starts(is_val, K, stride=3)
    g = torch.Generator().manual_seed(1)
    vsel = va[torch.randperm(len(va), generator=g)[:512].to(dev)]
    print(f"[{tag}] train windows: {len(tr)} | validation windows: {len(vsel)}")
    best_v, best_state, best_it = 1e9, None, 0
    t0 = time.time()
    for it in range(1, steps + 1):
        model.train()
        f, h = get(tr[torch.randint(len(tr), (bs,), device=dev)], K)
        gain = 0.85 + 0.3 * torch.rand(bs, 1, 1, 1, 1, device=dev)   # brightness
        bias = 0.1 * (torch.rand(bs, 1, 1, 1, 1, device=dev) - 0.5)
        f = (f * gain + bias).clamp(0, 1)
        pred = rollout(model, f[:, 0], h, use_action)
        loss = loss_fn(pred, f[:, 1:], f[:, :-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == steps or it == 50:
            model.eval()
            with torch.no_grad():
                f, h = get(vsel, K)
                p = rollout(model, f[:, 0], h, use_action)
                v = (p - f[:, 1:]).abs().mean().item()
                c = (f[:, :1] - f[:, 1:]).abs().mean().item()
            if v < best_v:
                best_v, best_it = v, it
                best_state = {k: x.detach().clone()
                              for k, x in model.state_dict().items()}
            print(f"[{tag}] step {it:5d} | loss {loss.item():.4f} | "
                  f"val: model {v:.4f} vs copy-last {c:.4f} | "
                  f"{time.time() - t0:.0f} s", flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out, f"{tag}.pt"))
    print(f"[{tag}] best validation checkpoint: step {best_it} (error {best_v:.4f})")


if args.models in ("act", "both"):
    train(True, "wm_act", args.steps, args.bs)
if args.models in ("noact", "both"):
    train(False, "wm_noact", args.steps, args.bs)