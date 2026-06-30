#!/usr/bin/env python3
"""
Assemble the v3 training dataset (3 classes: gun, knife, bat).

Builds on v2 by merging in the Zenodo "Dangerous Items" dataset (record 16422779,
`dangerous_items.zip`). Zenodo's 5 classes are remapped into our 3-class scheme:

    Zenodo id -> name          -> v3 class
    0  machete                 -> 1 (knife)
    1  knife                   -> 1 (knife)
    2  baseball bat            -> 2 (bat)
    3  rifle                   -> 0 (gun)
    4  gun                     -> 0 (gun)

(Verified visually, not from a class file — the zip ships no data.yaml.)

Existing data already uses 0=gun, 1=knife, so it is reused IN PLACE with no
relabeling; only Zenodo is remapped and copied:
  - haris-weapon-detection-dataset-curated/{train,val}  (gun/knife)
  - gun2_yolo/                                           (gun, train-only)

Zenodo images are deduplicated (average hash) against the WHOLE existing set
(curated train+val + gun2) so Zenodo's "public dataset" portion can't leak a
near-duplicate across the train/val boundary. Zenodo's own split is preserved:
its train+test -> v3 train, its val -> v3 val (so `bat` is represented in val).

Output:
  zenodo_yolo/{train,val}/{images,labels}/   — remapped, deduped survivors
  dataset_v3.yaml                            — local paths
  data_colab_v3.yaml                         — Colab paths (/content/dataset)

Run (needs Pillow + numpy -> conda env 'harmful'):
    conda run -n harmful python src/build_dataset_v3.py
"""
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
CURATED = DATA / "haris-weapon-detection-dataset-curated"
GUN2_OUT = DATA / "gun2_yolo"
ZIP = DATA / "dangerous_items.zip"
ZEN_OUT = DATA / "zenodo_yolo"
REF_CACHE = DATA / ".v3_ref_hashes.json"
YAML_LOCAL = ROOT / "configs" / "dataset_v3.yaml"
YAML_COLAB = ROOT / "configs" / "data_colab_v3.yaml"

REMAP = {0: 1, 1: 1, 2: 2, 3: 0, 4: 0}   # Zenodo id -> v3 id
NAMES = ["gun", "knife", "bat"]
HAMMING_THRESHOLD = 5
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ahash_img(im: Image.Image) -> int:
    g = im.convert("L").resize((8, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def ahash_path(path: Path):
    try:
        with Image.open(path) as im:
            return ahash_img(im)
    except Exception:
        return None


def reference_hashes() -> np.ndarray:
    """Average hashes of every existing image (curated train+val + gun2)."""
    if REF_CACHE.exists():
        vals = json.loads(REF_CACHE.read_text())
        print(f"Reference hashes: loaded {len(vals)} from cache.")
        return np.array(vals, dtype=np.uint64)

    paths = []
    for split in ("train", "val"):
        paths += [p for p in (CURATED / split / "images").iterdir()
                  if p.suffix.lower() in IMG_EXTS]
    paths += [p for p in (GUN2_OUT / "images").iterdir()
              if p.suffix.lower() in IMG_EXTS]

    print(f"Hashing existing set: {len(paths)} images ...")
    hashes = []
    for i, p in enumerate(paths, 1):
        h = ahash_path(p)
        if h is not None:
            hashes.append(h)
        if i % 3000 == 0:
            print(f"  {i}/{len(paths)}")
    REF_CACHE.write_text(json.dumps(hashes))
    print(f"  done: {len(hashes)} hashes (cached -> {REF_CACHE.name})")
    return np.array(hashes, dtype=np.uint64)


def min_distance(q: int, ref: np.ndarray) -> int:
    """Smallest Hamming distance from hash q to any reference hash (numpy popcount)."""
    x = ref ^ np.uint64(q)
    d = POP[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)
    return int(d.min())


def main():
    ref = reference_hashes()

    if ZEN_OUT.exists():
        shutil.rmtree(ZEN_OUT)
    for split in ("train", "val"):
        (ZEN_OUT / split / "images").mkdir(parents=True)
        (ZEN_OUT / split / "labels").mkdir(parents=True)

    kept = {"train": 0, "val": 0}
    boxes = {0: 0, 1: 0, 2: 0}
    dropped_dup = 0
    bg = 0

    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        label_entries = sorted(n for n in names
                               if "/labels/" in n and n.endswith(".txt"))
        print(f"Zenodo label files: {len(label_entries)}")

        for i, lbl in enumerate(label_entries, 1):
            src_split = lbl.split("/labels/")[1].split("/")[0]   # train/val/test
            dest = "val" if src_split == "val" else "train"      # train+test -> train
            stem = Path(lbl).stem

            # locate the matching image entry
            img_entry = None
            base = lbl.replace("/labels/", "/images/")[:-4]
            for ext in IMG_EXTS:
                if base + ext in names:
                    img_entry = base + ext
                    break
            if img_entry is None:
                continue

            img_bytes = z.read(img_entry)
            try:
                with Image.open(io.BytesIO(img_bytes)) as im:
                    h = ahash_img(im)
            except Exception:
                continue

            if h is not None and ref.size and min_distance(h, ref) <= HAMMING_THRESHOLD:
                dropped_dup += 1
                continue

            # remap label lines
            out_lines = []
            for row in z.read(lbl).decode("utf-8", "ignore").splitlines():
                parts = row.split()
                if len(parts) < 5:
                    continue
                new_c = REMAP.get(int(parts[0]))
                if new_c is None:
                    continue
                out_lines.append(f"{new_c} " + " ".join(parts[1:]))
                boxes[new_c] += 1

            ext = Path(img_entry).suffix.lower()
            (ZEN_OUT / dest / "images" / f"{stem}{ext}").write_bytes(img_bytes)
            (ZEN_OUT / dest / "labels" / f"{stem}.txt").write_text(
                ("\n".join(out_lines) + "\n") if out_lines else "")
            kept[dest] += 1
            if not out_lines:
                bg += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(label_entries)}  kept={sum(kept.values())} dup={dropped_dup}")

    YAML_LOCAL.parent.mkdir(exist_ok=True)
    for yaml_path, base in ((YAML_LOCAL, str(DATA)), (YAML_COLAB, "/content/dataset")):
        yaml_path.write_text(
            f"# v3: 3 classes (gun, knife, bat). curated+gun2 in place, Zenodo remapped+merged.\n"
            f"path: {base}\n"
            "train:\n"
            "  - haris-weapon-detection-dataset-curated/train/images\n"
            "  - gun2_yolo/images\n"
            "  - zenodo_yolo/train/images\n"
            "val:\n"
            "  - haris-weapon-detection-dataset-curated/val/images\n"
            "  - zenodo_yolo/val/images\n"
            "nc: 3\n"
            f"names: {NAMES}\n"
        )

    print("\n=== summary ===")
    print(f"Zenodo kept   : train={kept['train']}  val={kept['val']}  "
          f"(dropped {dropped_dup} duplicates of existing set, {bg} backgrounds)")
    print(f"Zenodo boxes  : gun={boxes[0]}  knife={boxes[1]}  bat={boxes[2]}")
    print(f"classes       : 0=gun, 1=knife, 2=bat")
    print(f"wrote         : {YAML_LOCAL.name}, {YAML_COLAB.name}")
    print("NOTE: bat exists ONLY in Zenodo -> verify bat count in val is healthy above.")


if __name__ == "__main__":
    main()
