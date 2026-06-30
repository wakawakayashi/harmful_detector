#!/usr/bin/env python3
"""
Assemble the v2 training dataset (2 classes: gun, knife).

Base: haris-weapon-detection-dataset-curated/ — already in YOLO format with a native
train/val split (verified: 0 train/val basename overlap, every image has a label).
Classes are 0=gun, 1=knife, matching our scheme. Used in place (not copied).

Extra: gun2/ (SasankYadati Guns, 333 imgs, single class = gun). Labels are corner-pixel
format: line 1 = #objects, then `xmin ymin xmax ymax` in absolute pixels. These are parsed
to normalized YOLO, deduplicated against the WHOLE curated set (train+val) with an average
hash, and the survivors are added to the TRAIN split only — never val — so no gun2 image
can leak across the train/val boundary. All gun2 boxes map to class 0 (gun).

Output:
    gun2_yolo/{images,labels}/   — parsed, deduped gun2 survivors
    dataset_v2.yaml              — train: [curated/train, gun2_yolo], val: curated/val

Run (needs Pillow + numpy -> use conda env 'harmful'):
    conda run -n harmful python src/build_dataset.py
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
CURATED = DATA / "haris-weapon-detection-dataset-curated"
GUN2 = DATA / "gun2"
GUN2_OUT = DATA / "gun2_yolo"
YAML_OUT = ROOT / "configs" / "dataset_v2.yaml"

GUN = 0  # target class id for gun
HAMMING_THRESHOLD = 5  # <= this average-hash distance counts as a duplicate

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def ahash(path: Path) -> int | None:
    """64-bit average hash. None if the image can't be read."""
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((8, 8), Image.BILINEAR)
        a = np.asarray(g, dtype=np.float32)
        bits = (a > a.mean()).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def curated_hashes() -> list[int]:
    """Average hashes of every curated image (train + val)."""
    hashes = []
    imgs = []
    for split in ("train", "val"):
        imgs += [p for p in (CURATED / split / "images").iterdir()
                 if p.suffix.lower() in IMG_EXTS]
    print(f"Hashing curated set: {len(imgs)} images ...")
    for i, p in enumerate(imgs, 1):
        h = ahash(p)
        if h is not None:
            hashes.append(h)
        if i % 2000 == 0:
            print(f"  {i}/{len(imgs)}")
    print(f"  done: {len(hashes)} hashes")
    return hashes


def parse_gun2():
    """Parse gun2 corner-pixel labels -> (image_path, [yolo_lines], hash). Skip repository/ clone."""
    img_dir = GUN2 / "Images"
    lbl_dir = GUN2 / "Labels"
    out = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        stem = lbl.stem
        img = next((img_dir / f"{stem}{e}" for e in IMG_EXTS
                    if (img_dir / f"{stem}{e}").exists()), None)
        if img is None:
            continue
        try:
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            continue
        if W <= 0 or H <= 0:
            continue
        rows = [r.strip() for r in lbl.read_text().splitlines() if r.strip()]
        if not rows:
            continue
        # rows[0] = object count; remaining rows = "xmin ymin xmax ymax"
        lines = []
        for r in rows[1:]:
            parts = r.split()
            if len(parts) != 4:
                continue
            xmin, ymin, xmax, ymax = (float(x) for x in parts)
            if xmax <= xmin or ymax <= ymin:
                continue
            xc = clamp01(((xmin + xmax) / 2) / W)
            yc = clamp01(((ymin + ymax) / 2) / H)
            ww = clamp01((xmax - xmin) / W)
            hh = clamp01((ymax - ymin) / H)
            if ww <= 0 or hh <= 0:
                continue
            lines.append(f"{GUN} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
        if lines:
            out.append((img, lines, ahash(img)))
    return out


def main():
    gun2 = parse_gun2()
    print(f"gun2 parsed: {len(gun2)} images with gun boxes")

    cur = curated_hashes()
    kept, dropped = [], 0
    for img, lines, h in gun2:
        if h is not None and any(hamming(h, c) <= HAMMING_THRESHOLD for c in cur):
            dropped += 1
            continue
        kept.append((img, lines))
    print(f"gun2 dedup vs curated: kept={len(kept)}  dropped(duplicate)={dropped}")

    # Write survivors to gun2_yolo/{images,labels}
    if GUN2_OUT.exists():
        shutil.rmtree(GUN2_OUT)
    (GUN2_OUT / "images").mkdir(parents=True)
    (GUN2_OUT / "labels").mkdir(parents=True)
    boxes = 0
    for img, lines in kept:
        name = f"gun2__{img.stem}{img.suffix.lower()}"
        shutil.copy2(img, GUN2_OUT / "images" / name)
        (GUN2_OUT / "labels" / f"gun2__{img.stem}.txt").write_text("\n".join(lines) + "\n")
        boxes += len(lines)

    # data.yaml — curated used in place, gun2 survivors appended to train
    YAML_OUT.parent.mkdir(exist_ok=True)
    YAML_OUT.write_text(
        f"path: {DATA}\n"
        "train:\n"
        "  - haris-weapon-detection-dataset-curated/train/images\n"
        "  - gun2_yolo/images\n"
        "val: haris-weapon-detection-dataset-curated/val/images\n"
        "nc: 2\n"
        "names: [gun, knife]\n"
    )

    print("\n=== summary ===")
    print(f"train sources : curated/train ({len(list((CURATED/'train'/'images').iterdir()))} imgs) "
          f"+ gun2_yolo ({len(kept)} imgs, {boxes} gun boxes)")
    print(f"val source    : curated/val ({len(list((CURATED/'val'/'images').iterdir()))} imgs)")
    print(f"classes       : 0=gun, 1=knife")
    print(f"wrote         : {YAML_OUT}")


if __name__ == "__main__":
    main()
