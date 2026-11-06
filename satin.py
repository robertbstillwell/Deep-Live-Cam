"""Satin-column generation for thin strokes (lettering, outlines).

Given a boolean mask of thin regions (at U px/mm), skeletonize it, split the
skeleton into branches, prune spurs, and return for each branch a centre path
with a local half-width taken from the distance transform.
"""
import numpy as np, cv2
from scipy import ndimage
from skimage.morphology import skeletonize


def _neighbors(skel):
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)


def _order_chain(pixels):
    """Order the (y,x) pixels of one 8-connected chain from an end; closed loops too."""
    pts = {tuple(p) for p in pixels}
    def nb(p):
        y, x = p
        out = []
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (y + dy, x + dx)
            if q in pts: out.append(q)
        return out
    ends = [p for p in pts if len(nb(p)) == 1]
    start = ends[0] if ends else next(iter(pts))
    order = [start]; seen = {start}; cur = start
    while True:
        nxt = [q for q in nb(cur) if q not in seen]
        if not nxt: break
        # prefer 4-connected continuation
        nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        cur = nxt[0]; order.append(cur); seen.add(cur)
    closed = not ends and len(order) > 3
    return np.array(order, float), closed


def skeleton_paths(mask, U, dist, min_len_mm=0.8, spur_factor=0.9):
    """Return list of (path_xy[N,2] in px, halfwidth[N] in px, closed) for the mask."""
    skel = skeletonize(mask)
    nbr = _neighbors(skel)
    junction = skel & (nbr >= 3)
    # thicken junctions a little so branches separate cleanly
    jd = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    branches = skel & ~jd
    n, lab = cv2.connectedComponents(branches.astype(np.uint8), connectivity=8)
    jn, jlab, jstats, jcent = cv2.connectedComponentsWithStats((jd & skel).astype(np.uint8), connectivity=8)
    paths = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(ys) == 0: continue
        chain, closed = _order_chain(np.column_stack([ys, xs]))
        # attach adjacent junction centroids at each end so columns meet
        touches_junction = [False, False]
        if not closed:
            for ei, idx in enumerate((0, -1)):
                y, x = chain[idx]
                y0, y1 = int(max(y - 2, 0)), int(min(y + 3, mask.shape[0]))
                x0, x1 = int(max(x - 2, 0)), int(min(x + 3, mask.shape[1]))
                js = np.unique(jlab[y0:y1, x0:x1]); js = js[js != 0]
                if len(js):
                    cy, cx = jcent[js[0]][1], jcent[js[0]][0]
                    pt = np.array([[cy, cx]])
                    chain = np.vstack([pt, chain]) if idx == 0 else np.vstack([chain, pt])
                    touches_junction[ei] = True
        xy = chain[:, ::-1].copy()          # (x, y)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1); length = seg.sum()
        hw = dist[chain[:, 0].astype(int).clip(0, mask.shape[0] - 1), chain[:, 1].astype(int).clip(0, mask.shape[1] - 1)]
        w_med = float(np.median(hw)) if len(hw) else 0
        # prune: a spur is a branch with a free end that's shorter than ~its own width
        free_end = (not closed) and (not all(touches_junction))
        if length < min_len_mm * U: continue
        if free_end and any(touches_junction) and length < spur_factor * w_med: continue
        paths.append((xy, hw, closed))
    return paths


def smooth_path(xy, hw, step_px, win):
    """Resample to `step_px` spacing then box-smooth position and width."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    L = np.concatenate(([0], np.cumsum(seg)))
    if L[-1] < step_px * 2: return xy, hw
    t = np.arange(0, L[-1], step_px)
    rx = np.interp(t, L, xy[:, 0]); ry = np.interp(t, L, xy[:, 1]); rw = np.interp(t, L, hw)
    k = np.ones(win) / win
    def sm(a):
        if len(a) <= win: return a
        pad = np.pad(a, (win // 2, win - 1 - win // 2), mode='edge')
        return np.convolve(pad, k, 'valid')
    return np.column_stack([sm(rx), sm(ry)]), sm(rw)


def satin_column(xy, hw, density_px, pull_px, min_hw_px, max_hw_px):
    """Zigzag across an open path with per-point half-width. Returns (N,2) points."""
    n = len(xy)
    if n < 2: return None
    k = min(4, n - 1)
    idx = np.arange(n)
    tang = xy[np.clip(idx + k, 0, n - 1)] - xy[np.clip(idx - k, 0, n - 1)]
    tang /= np.linalg.norm(tang, axis=1)[:, None] + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    w = np.clip(hw + pull_px, min_hw_px, max_hw_px)
    a = xy + nrm * w[:, None]; b = xy - nrm * w[:, None]
    step = density_px / 2.0
    pts, side, acc_a, acc_b = [], 0, 0.0, 0.0
    for i in range(n):
        if i:
            acc_a += np.hypot(*(a[i] - a[i - 1])); acc_b += np.hypot(*(b[i] - b[i - 1]))
        if not pts or max(acc_a, acc_b) >= step or i == n - 1:
            pts.append(a[i] if side == 0 else b[i]); side ^= 1; acc_a = acc_b = 0.0
    return np.array(pts)
