import sys, pyembroidery as pe
from PIL import Image, ImageDraw
src, out = sys.argv[1], sys.argv[2]
p = pe.read(src)
S = 8  # px per unit (0.1mm)
b = p.bounds()
W = int((b[2]-b[0])*S)+40; H = int((b[3]-b[1])*S)+40
img = Image.new('RGB', (W, H), (205, 60, 60))   # red "garment" background to show gaps
d = ImageDraw.Draw(img)
threads = p.threadlist or [pe.EmbThread(c) for c in (0xA9A6AB,0x5A5A5F,0xF2F2F2,0x141414)]
ti = 0
col = threads[0].hex_color() if threads else '#000000'
last = None
jumps = 0
for x, y, cmd in p.stitches:
    c = cmd & pe.COMMAND_MASK
    X, Y = (x-b[0])*S+20, (y-b[1])*S+20
    if c == pe.COLOR_CHANGE:
        ti += 1; col = threads[min(ti, len(threads)-1)].hex_color(); last = None
    elif c == pe.STITCH:
        if last is not None:
            d.line([last, (X, Y)], fill=col, width=int(0.35*10*S))
        last = (X, Y)
    elif c == pe.JUMP:
        last = (X, Y); jumps += 1
    elif c in (pe.TRIM, pe.END):
        last = None
img.thumbnail((1600, 1600))
img.save(out)
print('jumps', jumps, 'stitches', p.count_stitches(), 'threads', [t.description for t in threads])
