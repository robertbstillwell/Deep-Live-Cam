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

**Design:** 82 x 93 mm (3.25 x 3.65 in) including the satin border, fits a 4x4 in / 100x100 mm hoop.
**Stitches:** ~24,400, 4 thread colours, 6 machine stops.

Sewing sequence (each line is a machine stop):
1. Black – placement line on the stabilizer. STOP: lay the twill over it.
2. Black – tack-down line (sews the twill to the stabilizer).
3. Light grey / silver – base fill of the "A"
4. Dark grey / pewter – elephant shading
5. White – banner and tusk highlights
6. Black – outlines and "CRIMSON TIDE" lettering. STOP: trim the twill ~1 mm outside the tack-down line.
7. Black – 3 mm satin border over the raw edge (run tack + zigzag underlay + satin).

Stitch types:
- Lettering and all black outlines narrower than ~3 mm are **satin columns**
  (0.4 mm density, centre-run underlay, width follows the stroke, 0.12 mm pull
  compensation). Wider black areas and anything the columns miss get a fill.
- Grey, white and elephant shading are tatami fills, 0.35–0.38 mm row spacing,
  2.4–3 mm stitch length, rows run across each shape's long axis, lattice underlay
  under the big grey and white areas, 0.2 mm pull compensation. The grey and white
  fills continue underneath the satin strokes so no fabric shows through.
- Tie-in/tie-off and trims between islands.

Suggested setup: cut-away or heavy tear-away stabilizer hooped alone, patch twill
laid on after the placement line, 75/11 needle, standard 40 wt polyester.
After the final stop, peel/cut the stabilizer close to the border.

Regenerate / change size: edit `DESIGN_H_MM` in `digitize_patch.py` and run
`python3 digitize_patch.py <output dir>` (needs pyembroidery, opencv-python-headless, scipy, scikit-image, pillow; keep `satin.py` next to it).
