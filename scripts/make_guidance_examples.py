"""
make_guidance_examples.py

Build app/static/guidance_examples.png — the illustrated "how to photograph
your specimen" strip shown in the app's guidance banner.

Panels are generated from the project's own lab specimen images and reproduce
behaviour measured on the deployed model:
  - portrait orientation : raw, this costs top-1 89% -> ~32%, but the app now
                           rotates portrait uploads automatically, recovering
                           92% (vs 98% for a landscape original) — so it is
                           shown as "handled", not "avoid"
  - low magnification    : novelty score climbs above threshold below ~216 px
                           of real detail on the beetle body — still a genuine
                           failure the app cannot correct

Run from the carabID root:
    conda run -n ultralytics-env python scripts/make_guidance_examples.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "app/static/example_image.jpg"
OUT  = ROOT / "app/static/guidance_examples.png"

CELL, PAD, HDR = 600, 20, 50
GOOD, BAD, OKAY = (22, 122, 60), (183, 40, 40), (176, 122, 20)

try:
    ftitle = ImageFont.truetype("arialbd.ttf", 32)
    fsub   = ImageFont.truetype("arial.ttf", 24)
except Exception:
    ftitle = fsub = ImageFont.load_default()


def cell(img, title, subtitle, colour):
    """One labelled panel."""
    c = Image.new("RGB", (CELL, CELL + HDR + 22), (255, 255, 255))
    im = img.copy()
    im.thumbnail((CELL - 2 * PAD, CELL - 2 * PAD))
    c.paste(im, ((CELL - im.width) // 2, HDR + (CELL - im.height) // 2))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, CELL, HDR], fill=colour)
    d.text((8, 6), title, fill=(255, 255, 255), font=ftitle)
    d.text((8, CELL + HDR + -10), subtitle, fill=(70, 70, 70), font=fsub)
    d.rectangle([0, 0, CELL - 1, CELL + HDR + 21], outline=(215, 215, 215))
    return c


def main():
    if not SRC.exists():
        raise SystemExit(f"source image not found: {SRC}")
    base = Image.open(SRC).convert("RGB")

    # 1. Good: as trained — dorsal, horizontal, fills frame, plain background
    good = cell(base, "GOOD", "Dorsal, horizontal, sharp, plain background", GOOD)

    # 2. Handled: portrait is rotated back to landscape automatically
    rot = base.rotate(-90, expand=True)
    ok_rot = cell(rot, "OK  Portrait", "Rotated automatically, landscape preferred", OKAY)

    # 3. Bad: too far away — beetle small in the frame, so the body carries
    #    too few pixels and fine detail is lost (measured: novelty score climbs
    #    once the body falls below ~216 px of real detail).
    W0, H0 = base.size
    shrunk = base.resize((int(W0 * 0.60), int(H0 * 0.60)), Image.LANCZOS)
    far = Image.new("RGB", (W0, H0), (226, 227, 229))
    far.paste(shrunk, ((W0 - shrunk.width) // 2, (H0 - shrunk.height) // 2))
    bad_res = cell(far, "AVOID  Too far away", "Body too few pixels, fine detail lost", BAD)

    W = CELL * 3
    H = CELL + HDR + 22
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for i, c in enumerate((good, ok_rot, bad_res)):
        canvas.paste(c, (i * CELL, 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print(f"saved {OUT}  ({canvas.size[0]}x{canvas.size[1]}, "
          f"{OUT.stat().st_size/1024:.0f} KB)")

 
if __name__ == "__main__":
    main()
