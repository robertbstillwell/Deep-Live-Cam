"""Auto-digitize the Crimson Tide patch photo into DST / PES embroidery files.

Pipeline: photo -> patch mask -> 4-colour quantisation -> cleanup -> per-colour
tatami fill (scan-line sections, staggered needle points, serpentine rows,
tie-in/tie-off, trims between islands) -> pyembroidery -> .dst / .pes / preview.
"""
import math, sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
import pyembroidery as pe

SRC = '/root/.claude/uploads/d354b3f8-a0eb-5581-9f0b-47100bf9b24a/973c3e36-image.jpg'
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
DESIGN_H_MM = 90.0           # finished height of the design
U = 10                       # working grid: 10 px per mm (1 px == 0.1 mm == 1 embroidery unit)

# ---------------------------------------------------------------- image prep
im = np.array(Image.open(SRC).convert('RGB'))
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# patch mask: bright background connected to the image border
bright = (gray > 225).astype(np.uint8)
n, lab = cv2.connectedComponents(bright)
border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
mask = (~np.isin(lab, list(border - {0}))).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(mask); cv2.drawContours(mask, cnts, -1, 1, -1)
# trim the fuzzy photographed merrow edge so the black outline is the outer edge
mask = cv2.erode(mask, np.ones((45, 45), np.uint8))

ys, xs = np.where(mask)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
scale = DESIGN_H_MM * U / (y1 - y0)                       # working px per source px
W = int(round((x1 - x0) * scale)) + 1
H = int(round((y1 - y0) * scale)) + 1

# classify: 0 black, 1 dark grey, 2 light grey, 3 white
g = cv2.medianBlur(gray, 13)
g = cv2.bilateralFilter(g, 11, 45, 11)
cls = np.digitize(g, [62, 128, 196]).astype(np.uint8)
# white vs light grey decided by the local average brightness of the light pixels
# in a ~2 mm window, so thread sheen specks don't register as white
light = (cls >= 2).astype(np.float32)
win = int(2.0 / (DESIGN_H_MM / (y1 - y0))) | 1
num = cv2.blur(g.astype(np.float32) * light, (win, win)); den = cv2.blur(light, (win, win))
lmean = np.where(den > 0, num / np.maximum(den, 1e-6), 0)
cls[(cls >= 2) & (lmean > 197)] = 3
cls[(cls >= 2) & (lmean <= 197)] = 2
cls_c = cls[y0:y1 + 1, x0:x1 + 1]
m_c = mask[y0:y1 + 1, x0:x1 + 1]
cls_w = cv2.resize(cls_c, (W, H), interpolation=cv2.INTER_NEAREST)
m_w = cv2.resize(m_c, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

# cleanup: drop islands below ~1.2 mm² and hand the pixels to the nearest class
def clean(cls_w, min_area):
    unknown = np.zeros(cls_w.shape, bool)
    for c in range(4):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((cls_w == c).astype(np.uint8), connectivity=4)
        small = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
        small = small[small != 0]
        unknown |= np.isin(lab, small)
    if unknown.any():
        _, (iy, ix) = ndimage.distance_transform_edt(unknown, return_indices=True)
        cls_w = cls_w[iy, ix]
    return cls_w

cls_w = cv2.medianBlur(cls_w, 5)
cls_w = clean(cls_w, int(1.2 * U * U))
cls_w = cv2.medianBlur(cls_w, 5)
cls_w = clean(cls_w, int(1.2 * U * U))
# white: open to kill thread-sheen speckles, then drop small white islands (< 3 mm^2)
wk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
w_open = cv2.morphologyEx((cls_w == 3).astype(np.uint8), cv2.MORPH_OPEN, wk) > 0
n, lab, stats, _ = cv2.connectedComponentsWithStats(w_open.astype(np.uint8), connectivity=4)
keep = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3 * U * U])
lost = (cls_w == 3) & ~keep
cls_w[lost] = 2                      # fallen-back speckles become light grey
cls_w[~m_w] = 255

pal = np.array([[0, 0, 0], [90, 90, 95], [165, 162, 167], [240, 240, 240], [255, 255, 255]], np.uint8)
Image.fromarray(pal[np.where(cls_w == 255, 4, cls_w)]).save(f'{OUT}/crimson_tide_patch_layers.png')

# ---------------------------------------------------------------- fill engine
def rot_mats(angle_deg, shape):
    h, w = shape
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2.0 - c[0]
    M[1, 2] += nh / 2.0 - c[1]
    return M, cv2.invertAffineTransform(M), (nw, nh)

def row_segments(row):
    """[(x_start, x_end)] of True runs in a boolean row (inclusive pixel indices)."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))

def sections_for(mask, spacing, min_seg):
    """Decompose a mask into scan-line sections (one segment per row, rows overlap)."""
    rows = range(0, mask.shape[0], spacing)
    open_secs, done = [], []
    prev_r = None
    for r in rows:
        segs = [s for s in row_segments(mask[r]) if s[1] - s[0] + 1 >= min_seg]
        new_open = []
        used = set()
        for (a, b) in segs:
            match = None
            if prev_r is not None and r - prev_r == spacing:
                for i, sec in enumerate(open_secs):
                    if i in used: continue
                    pa, pb = sec[-1][1], sec[-1][2]
                    if a <= pb and b >= pa:            # overlap
                        match = i; break
            if match is None:
                sec = [(r, a, b)]
            else:
                used.add(match); sec = open_secs[match]; sec.append((r, a, b))
            new_open.append(sec)
        for i, sec in enumerate(open_secs):
            if i not in used: done.append(sec)
        open_secs = new_open
        prev_r = r
    done.extend(open_secs)
    return done

def sew_section(sec, stitch_len, start_top, start_left):
    """Return list of (x, y) in rotated-canvas px for one section, serpentine rows."""
    rows = sec if start_top else sec[::-1]
    pts = []
    left = start_left
    for k, (r, a, b) in enumerate(rows):
        a = a + 0.5; b = b + 0.5
        L = b - a
        # staggered needle points: tatami offset of 1/4 stitch per row
        off = (k * stitch_len * 0.25) % stitch_len
        inner = []
        x = a + off
        while x < b - 0.3 * stitch_len:
            if x > a + 0.3 * stitch_len: inner.append(x)
            x += stitch_len
        xs = [a] + inner + [b]
        if not left: xs = xs[::-1]
        pts.extend((x, float(r) + 0.5) for x in xs)
        left = not left
    return pts

def fill_layer(mask, angle, spacing_mm, stitch_mm, min_seg_mm=0.25):
    """Yield stitch runs (list of (x,y) in design px) for the whole layer."""
    spacing = max(1, int(round(spacing_mm * U)))
    stitch_len = stitch_mm * U
    M, Minv, (nw, nh) = rot_mats(angle, mask.shape)
    rm = cv2.warpAffine(mask.astype(np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST) > 0
    secs = sections_for(rm, spacing, max(1, int(round(min_seg_mm * U))))
    # candidate starts for nearest-neighbour ordering
    def ends(sec):
        (r0, a0, b0), (r1, a1, b1) = sec[0], sec[-1]
        return {(True, True): (a0, r0), (True, False): (b0, r0), (False, True): (a1, r1), (False, False): (b1, r1)}
    remaining = list(range(len(secs)))
    cur = None
    runs = []
    while remaining:
        best = None
        for i in remaining:
            for (top, left), (x, y) in ends(secs[i]).items():
                d = 0 if cur is None else (x - cur[0]) ** 2 + (y - cur[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, i, top, left)
                if cur is None: break
            if cur is None: break
        _, i, top, left = best
        remaining.remove(i)
        pts = sew_section(secs[i], stitch_len, top, left)
        cur = pts[-1]
        # back to design coordinates
        P = np.array(pts, np.float64)
        Q = P @ Minv[:, :2].T + Minv[:, 2]
        runs.append(Q)
    return runs

# ---------------------------------------------------------------- assemble
def dil(m, mm):
    k = int(round(mm * U)) * 2 + 1
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
def ero(m, mm):
    k = int(round(mm * U)) * 2 + 1
    return cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0

layers = [
    # name, hex, class, angle, spacing, stitch, dilate(pull comp / overlap), underlay
    ("Light Grey / Silver", 0xA9A6AB, 2,  0, 0.35, 3.0, 0.00, True),
    ("Dark Grey",           0x5A5A5F, 1, 45, 0.38, 2.6, 0.20, False),
    ("White",               0xF2F2F2, 3,  0, 0.38, 3.0, 0.20, True),
    ("Black",               0x000000, 0, 30, 0.36, 2.4, 0.22, False),
]

pattern = pe.EmbPattern()
TRAVEL_MAX = 2.0 * U      # straight travel stitches instead of trim below this gap

def add_tie(pat, x, y, dx=0.6 * U):
    for k in (0, 1, 0, 1):
        pat.add_stitch_absolute(pe.STITCH, x + k * dx * 0.5, y)

def add_runs(pat, runs, first_in_layer):
    for run in runs:
        if len(run) < 2: continue
        sx, sy = run[0]
        lx, ly = pat.stitches[-1][:2] if pat.stitches else (None, None)
        if lx is None or first_in_layer:
            pat.add_stitch_absolute(pe.JUMP, sx, sy)
            add_tie(pat, sx, sy)
            first_in_layer = False
        else:
            d = math.hypot(sx - lx, sy - ly)
            if d <= TRAVEL_MAX:
                n = max(1, int(math.ceil(d / (2.5 * U))))
                for k in range(1, n + 1):
                    pat.add_stitch_absolute(pe.STITCH, lx + (sx - lx) * k / n, ly + (sy - ly) * k / n)
            else:
                add_tie(pat, lx, ly)
                pat.add_command(pe.TRIM)
                pat.add_stitch_absolute(pe.JUMP, sx, sy)
                add_tie(pat, sx, sy)
        for (x, y) in run:
            pat.add_stitch_absolute(pe.STITCH, x, y)
    return first_in_layer

total = 0
for li, (name, hexcol, c, ang, sp, sl, dl, under) in enumerate(layers):
    m = cls_w == c
    if dl > 0: m = dil(m, dl) & m_w
    if li > 0: pattern.add_command(pe.COLOR_CHANGE)
    pattern.add_thread({"color": hexcol, "description": name, "brand": "Generic"})
    first = True
    before = len(pattern.stitches)
    if under:      # light lattice underlay to stabilise big fills
        um = ero(m, 0.5)
        first = add_runs(pattern, fill_layer(um, ang + 90, 2.0, 4.0, 1.0), first)
    first = add_runs(pattern, fill_layer(m, ang, sp, sl), first)
    if pattern.stitches:
        lx, ly = pattern.stitches[-1][:2]; add_tie(pattern, lx, ly)
    pattern.add_command(pe.TRIM)
    print(f"{name:22s} {len(pattern.stitches) - before:6d} stitches")
pattern.end()

# centre the design on the origin
b = pattern.bounds()
pattern.translate(-(b[0] + b[2]) / 2.0, -(b[1] + b[3]) / 2.0)
b = pattern.bounds()
print(f"size: {(b[2]-b[0])/U:.1f} x {(b[3]-b[1])/U:.1f} mm, stitches: {pattern.count_stitches()}, colours: {len(pattern.threadlist)}, "
      f"colour changes: {pattern.count_color_changes()}")

pe.write_dst(pattern, f'{OUT}/crimson_tide_patch.dst', {"extended headers": True})
pe.write_pes(pattern, f'{OUT}/crimson_tide_patch.pes')
pe.write_exp(pattern, f'{OUT}/crimson_tide_patch.exp')
pe.write_jef(pattern, f'{OUT}/crimson_tide_patch.jef')
pe.write_png(pattern, f'{OUT}/crimson_tide_patch_preview.png', {"linewidth": 3, "scale": 6})
