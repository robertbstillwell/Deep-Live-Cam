"""Auto-digitize the Crimson Tide patch photo into DST / PES embroidery files.

Pipeline: photo -> patch mask -> 4-colour quantisation -> cleanup -> per-colour
tatami fill (scan-line sections, staggered needle points, serpentine rows,
tie-in/tie-off, trims between islands) -> pyembroidery -> .dst / .pes / preview.
"""
import math, sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
import pyembroidery as pe
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from satin import skeleton_paths, smooth_path, satin_column

SRC = '/root/.claude/uploads/d354b3f8-a0eb-5581-9f0b-47100bf9b24a/973c3e36-image.jpg'
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
DEBUG = False
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
g = cv2.medianBlur(gray, 7)
g = cv2.bilateralFilter(g, 9, 40, 9)
cls = np.digitize(g, [58, 125, 196]).astype(np.uint8)
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


# ---------------------------------------------------------------- satin border
from scipy.spatial import cKDTree

BORDER_W_MM = 3.0      # finished satin width, centred on the patch edge

def resample(poly, step):
    """Resample a closed polyline (N,2) to roughly uniform spacing `step`."""
    poly = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = np.concatenate(([0], np.cumsum(seg)))
    t = np.arange(0, L[-1], step)
    return np.column_stack([np.interp(t, L, poly[:, 0]), np.interp(t, L, poly[:, 1])])

def smooth_closed(poly, win):
    k = np.ones(win) / win
    out = np.empty_like(poly)
    for d in range(2):
        out[:, d] = np.convolve(np.concatenate([poly[-win:, d], poly[:, d], poly[:win, d]]), k, 'same')[win:-win]
    return out

def contours_of(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(np.float64) for c in cnts if len(c) > 20]

def offset_points(mask, mm):
    k = int(round(abs(mm) * U)) * 2 + 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m2 = cv2.dilate(mask.astype(np.uint8), el) if mm > 0 else cv2.erode(mask.astype(np.uint8), el)
    pts = [resample(c, 0.5) for c in contours_of(m2)]
    return np.vstack(pts) if pts else None

def border_centreline(mask):
    """Smoothed outer contour of the patch (corners rounded ~1.5 mm), every 0.05 mm."""
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.6 * U) > 0.5
    c = max(contours_of(soft), key=len)
    c = resample(c, 1.0)
    for _ in range(3):
        c = smooth_closed(c, 25)          # 3 passes of a 2.5 mm box filter
    return resample(c, 0.5), soft

def zigzag(centre, soft_mask, half_w_mm, density_mm):
    """Satin / zigzag of total width 2*half_w centred on the path.

    Needle points are offset along smoothed normals; the step along the path is
    chosen so that the longer side (outside of a bend) keeps the requested density."""
    half_w = half_w_mm * U
    n = len(centre); k = 10                              # 0.5 mm tangent stencil
    tang = centre[(np.arange(n) + k) % n] - centre[(np.arange(n) - k) % n]
    tang /= np.linalg.norm(tang, axis=1)[:, None] + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    probe = centre[0] + nrm[0] * half_w
    px, py = int(round(probe[0])), int(round(probe[1]))
    inside = 0 <= px < soft_mask.shape[1] and 0 <= py < soft_mask.shape[0] and soft_mask[py, px]
    if inside: nrm = -nrm                                 # make normals point outward
    outer = centre + nrm * half_w
    inner = centre - nrm * half_w
    step = density_mm * U / 2.0
    pts, side, acc_o, acc_i = [], 0, 0.0, 0.0
    for i in range(n):
        if i:
            acc_o += np.hypot(*(outer[i] - outer[i - 1])); acc_i += np.hypot(*(inner[i] - inner[i - 1]))
        if not pts or max(acc_o, acc_i) >= step:
            pts.append(outer[i] if side == 0 else inner[i]); side ^= 1; acc_o = acc_i = 0.0
    pts.append(pts[0]); pts.append(pts[1])                # close the ring with a small overlap
    return np.array(pts)

def run_stitch(centre, stitch_mm):
    pts = resample(centre, stitch_mm * U)
    return np.vstack([pts, pts[:2]])

centre, soft_mask = border_centreline(m_w)
tack_run   = run_stitch(centre, 2.5)
zz_under   = zigzag(centre, soft_mask, BORDER_W_MM / 2 - 0.35, 2.0)
satin_top  = zigzag(centre, soft_mask, BORDER_W_MM / 2, 0.38)

# make the fills meet the border: fill any sliver between the original edge and the
# (rounded) border centreline with the base grey, and clip everything to the satin's
# outer edge less 0.3 mm so nothing pokes out at rounded corners
def poly_mask(poly):
    m = np.zeros(m_w.shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m > 0
inside_centre = poly_mask(centre)
outer_half = zigzag(centre, soft_mask, BORDER_W_MM / 2 - 0.3, 0.2)[0::2]   # outer needle points only
inside_outer = poly_mask(outer_half)
gap = inside_centre & ~m_w
cls_w[gap] = 2
cls_w[~inside_outer] = 255
m_w = (cls_w != 255)

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
BLACK = 0x000000
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

def block(name, hexcol, runs_list, first_block=False):
    """One colour block (machine stop) made of stitch runs."""
    if not first_block: pattern.add_command(pe.COLOR_CHANGE)
    pattern.add_thread({"color": hexcol, "description": "Black", "brand": "Generic"})
    before = len(pattern.stitches)
    first = True
    for runs in runs_list:
        first = add_runs(pattern, runs, first)
    lx, ly = pattern.stitches[-1][:2]; add_tie(pattern, lx, ly)
    pattern.add_command(pe.TRIM)
    print(f"{name:40s} {len(pattern.stitches) - before:6d} stitches")

# Block 1: placement + tack-down line (sew once on stabilizer, lay twill, sew again)
block("Black - placement line (lay twill after)", BLACK, [[tack_run]], first_block=True)
block("Black - tack-down line", BLACK, [[tack_run]])

SATIN_MAX_W_MM = 3.2      # black strokes narrower than this are sewn as satin columns

def directional_fill(mask, default_angle, sp, sl, big_mm2=300):
    """Fill each connected component with rows across its long axis."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    runs = []
    for i in range(1, n):
        comp = lab == i
        area = stats[i, cv2.CC_STAT_AREA]
        ang = default_angle
        if 30 < area < big_mm2 * U * U:
            ys, xs = np.where(comp)
            cov = np.cov(np.vstack([xs, ys]))
            ev, evec = np.linalg.eigh(cov)
            if ev[1] > 2.0 * ev[0]:                       # clearly elongated
                major = evec[:, 1]
                ang = math.degrees(math.atan2(major[1], major[0])) + 90   # rows across the shape
        runs.extend(fill_layer(comp, ang, sp, sl))
    return runs

def satin_strokes(thin, dist, density_mm=0.40, pull_mm=0.12):
    """Satin columns for thin strokes, with a centre-run underlay per column."""
    runs = []
    for xy, hw, closed in skeleton_paths(thin, U, dist):
        xy, hw = smooth_path(xy, hw, 0.5, 9)
        if closed:
            xy = np.vstack([xy, xy[:1]]); hw = np.append(hw, hw[0])
        col = satin_column(xy, hw, density_mm * U, pull_mm * U, 0.35 * U, SATIN_MAX_W_MM / 2 * U)
        if col is None or len(col) < 4: continue
        # centre-walk underlay forward, satin back along the stroke
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1); L = np.concatenate(([0], np.cumsum(seg)))
        t = np.arange(0, L[-1], 2.0 * U)
        run = np.column_stack([np.interp(t, L, xy[:, 0]), np.interp(t, L, xy[:, 1])])
        run = np.vstack([run, xy[-1:]])
        runs.append(np.vstack([run, col[::-1]]))
    return runs

# ---- black split: wide areas -> fill, thin strokes -> satin; lower colours run under the strokes
m_black0 = (cls_w == 0) & m_w
m_black = dil(m_black0, 0.22) & m_w
_r = SATIN_MAX_W_MM / 2
wide_black = dil(ero(m_black, _r), _r + 0.15) & m_black
thin_black = m_black0 & ~wide_black
thin_black = cv2.morphologyEx(thin_black.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
dist_black = ndimage.distance_transform_edt(m_black0)
_nb = cls_w == 0
_, (_iy, _ix) = ndimage.distance_transform_edt(_nb, return_indices=True)
nearest_nonblack = cls_w[_iy, _ix]            # class of the nearest non-black pixel
if DEBUG: Image.fromarray((wide_black * 120 + thin_black * 255).astype(np.uint8)).save(f'{OUT}/debug_black_split.png')

def satin_coverage(runs, shape):
    cov = np.zeros(shape, np.uint8)
    for r in runs:
        pts = np.round(r).astype(np.int32)
        cv2.fillPoly(cov, [pts], 1)             # zigzag polygon covers the column
    return cov > 0

total = 0
for li, (name, hexcol, c, ang, sp, sl, dl, under) in enumerate(layers):
    m = cls_w == c
    if dl > 0: m = dil(m, dl) & m_w
    if c != 0:
        m = m | (thin_black & (nearest_nonblack == c) & m_w)   # fill continues under satin strokes
    pattern.add_command(pe.COLOR_CHANGE)
    pattern.add_thread({"color": hexcol, "description": name, "brand": "Generic"})
    first = True
    before = len(pattern.stitches)
    if under:      # light lattice underlay to stabilise big fills
        um = ero(m, 0.5)
        first = add_runs(pattern, fill_layer(um, ang + 90, 2.0, 4.0, 1.0), first)
    if c == 0:
        # black: wide areas get a fill, thin strokes (outlines, lettering) get satin,
        # and whatever the satin columns miss gets a small fill so nothing is left bare
        first = add_runs(pattern, directional_fill(wide_black, ang, sp, sl), first)
        n_before = len(pattern.stitches)
        sat = satin_strokes(thin_black, dist_black)
        first = add_runs(pattern, sat, first)
        print(f"{'  satin strokes':40s} {len(pattern.stitches) - n_before:6d} stitches")
        rem = thin_black & ~satin_coverage(sat, m_w.shape)
        rem = cv2.morphologyEx(rem.astype(np.uint8), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        rem = dil(rem, 0.2) & thin_black
        n_before = len(pattern.stitches)
        first = add_runs(pattern, directional_fill(rem, ang, sp, 2.0), first)
        print(f"{'  remainder fill':40s} {len(pattern.stitches) - n_before:6d} stitches")
        if DEBUG: Image.fromarray((rem * 255).astype(np.uint8)).save(f'{OUT}/debug_black_remainder.png')
    elif c == 1:
        first = add_runs(pattern, directional_fill(m, ang, sp, sl), first)
    else:
        first = add_runs(pattern, fill_layer(m, ang, sp, sl), first)
    if pattern.stitches:
        lx, ly = pattern.stitches[-1][:2]; add_tie(pattern, lx, ly)
    pattern.add_command(pe.TRIM)
    print(f"{name:40s} {len(pattern.stitches) - before:6d} stitches")

# Last block: trim the twill close to the tack-down line, then the satin border
block("Black - satin border (trim twill first)", BLACK, [[tack_run], [zz_under], [satin_top]])
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
