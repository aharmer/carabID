"""
upload_to_roboflow.py

Upload images (optionally with auto-generated YOLO boxes) to a Roboflow project.

The API key is read from the ROBOFLOW_API_KEY environment variable and is never
printed or written to disk.  Set it yourself before running, e.g. in PowerShell:

    $env:ROBOFLOW_API_KEY = "<your key>"

Uploading sends images to a third-party service.  Where the source images carry
CC licences (the iNaturalist set does), attribution travels in the matching
attribution.csv, not in the upload itself - Roboflow does not preserve it, so
keep that CSV with the project.

Run from the carabID root, e.g.:
    conda run -n ultralytics-env python scripts/upload_to_roboflow.py \
        --workspace rainna --project extra_nzac \
        --images imgs/nzac_annotated/images --labels imgs/nzac_annotated/labels \
        --batch nzac_specimens --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--images", required=True,
                    help="directory of images (searched recursively)")
    ap.add_argument("--labels", default=None,
                    help="directory of matching YOLO .txt labels (optional)")
    ap.add_argument("--batch", default=None,
                    help="Roboflow batch name, to keep uploads separable")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be uploaded, send nothing")
    args = ap.parse_args()

    img_dir = (ROOT / args.images) if not Path(args.images).is_absolute() else Path(args.images)
    lbl_dir = None
    if args.labels:
        lbl_dir = (ROOT / args.labels) if not Path(args.labels).is_absolute() else Path(args.labels)

    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no images found under {img_dir}")

    def label_for(f):
        """Match a label to an image, allowing for the genus-prefixed names
        auto_annotate.py writes (<genus>_<stem>.txt)."""
        if not lbl_dir:
            return None
        for cand in (lbl_dir / f"{f.stem}.txt",
                     lbl_dir / f"{f.parent.name}_{f.stem}.txt"):
            if cand.exists():
                return str(cand)
        return None

    paired = sum(1 for f in files if label_for(f))

    print(f"images to upload : {len(files)}")
    print(f"  with boxes     : {paired}")
    print(f"  image only     : {len(files) - paired}")
    print(f"destination      : {args.workspace}/{args.project}  (split={args.split}"
          + (f", batch={args.batch}" if args.batch else "") + ")")

    if args.dry_run:
        print("\nDRY RUN - nothing uploaded.")
        for f in files[:5]:
            print(f"  {f.name}{'  [+box]' if label_for(f) else ''}")
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more")
        return

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set in the environment.\n"
                 "Set it yourself (it must not be pasted into a chat), e.g.:\n"
                 '    $env:ROBOFLOW_API_KEY = "<your key>"')

    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    project = rf.workspace(args.workspace).project(args.project)

    ok = fail = 0
    for i, f in enumerate(files, 1):
        ann = label_for(f)
        try:
            project.single_upload(
                image_path=str(f),
                annotation_path=ann,
                batch_name=args.batch,
                split=args.split,
                num_retry_uploads=2,
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  failed {f.name}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(files)}  (ok={ok} failed={fail})", flush=True)

    print(f"\nuploaded {ok}/{len(files)}  (failed {fail})")
    print("Remember to keep the matching attribution.csv with the project: "
          "the CC licences on the iNaturalist images require attribution.")


if __name__ == "__main__":
    main()
