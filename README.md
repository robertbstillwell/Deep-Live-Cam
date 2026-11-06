# Crimson Tide patch – machine embroidery files

Auto-digitized from a photo of the embroidered patch.

| File | Use |
|---|---|
| `crimson_tide_patch.pes` | Brother / Babylock / Bernina (PES v1 with PEC, thread colours embedded) |
| `crimson_tide_patch.dst` | Tajima and most commercial machines (no colours stored in DST – follow the order below) |
| `crimson_tide_patch.jef` | Janome |
| `crimson_tide_patch.exp` | Melco / Bernina |
| `crimson_tide_patch_preview.png` | Stitch simulation |
| `crimson_tide_patch_layers.png` | Colour-separated artwork the stitches were built from |

**Design:** 79.5 x 90 mm (3.1 x 3.5 in), fits a 4x4 in / 100x100 mm hoop.
**Stitches:** ~15,300, 4 colours, 3 colour changes.

Thread order (sew in this sequence):
1. Light grey / silver – base fill of the "A"
2. Dark grey / pewter – elephant shading
3. White – banner and tusk highlights
4. Black – outlines and "CRIMSON TIDE" lettering

Fill style: tatami fill, 0.35–0.38 mm row spacing, 2.4–3 mm stitch length,
light lattice underlay under the grey and white areas, 0.2 mm pull compensation
on the upper colours, tie-in/tie-off and trims between islands.

Suggested setup: cut-away or heavy tear-away stabilizer, 75/11 needle,
standard 40 wt polyester. For a sew-on patch, stitch on twill and cut out
afterwards, or add a satin border in your embroidery software.

Regenerate / change size: edit `DESIGN_H_MM` in `digitize_patch.py` and run
`python3 digitize_patch.py <output dir>` (needs pyembroidery, opencv-python-headless, scipy, pillow).
