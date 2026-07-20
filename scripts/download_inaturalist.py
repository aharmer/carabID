"""
download_inaturalist.py

Download candidate specimen photos from iNaturalist for the project's genera.

Defaults are deliberately conservative: New Zealand only (several genera are
cosmopolitan, and their overseas species are not what the model identifies),
research-grade only, reusable licences only, and the first photo of each
observation (the later ones are usually habitat shots, alternate angles or
labels, and near-duplicates of one individual can leak across a train/test
split).

Attribution is mandatory under every licence accepted here, so photographer,
licence and observation URL are written to a CSV alongside the images.  Nothing
downloaded here is ready for training: each image still needs a pronotum+elytra
box drawn to match the lab convention.

Run from the carabID root, e.g.:
    conda run -n ultralytics-env python scripts/download_inaturalist.py \
        --cap 20 --out imgs/inat_candidates
"""
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

API      = "https://api.inaturalist.org/v1/observations"
NZ_PLACE = 6803
REUSE    = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"
UA       = {"User-Agent": "carabID-research/1.0 (+https://github.com/aharmer/carabID)"}
ROOT     = Path(__file__).resolve().parent.parent


def api_get(**params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                print(f"    API failed: {e}")
                return {"results": []}
            time.sleep(2 * (attempt + 1))


def fetch_observations(genus, cap, place, licences):
    """Return up to `cap` observations, newest-first paging."""
    out, page = [], 1
    while len(out) < cap:
        params = dict(taxon_name=genus, quality_grade="research", photos="true",
                      per_page=min(200, cap - len(out)), page=page,
                      order_by="votes")          # best-voted first: better photos
        if place:
            params["place_id"] = place
        if licences:
            params["photo_license"] = licences
        data = api_get(**params)
        res = data.get("results", [])
        if not res:
            break
        out.extend(res)
        if len(res) < params["per_page"]:
            break
        page += 1
        time.sleep(1.1)
    return out[:cap]


def photo_url(photo, size="large"):
    """iNat serves sizes by filename suffix: square/small/medium/large/original."""
    url = photo.get("url", "")
    for s in ("square", "small", "medium", "large", "original"):
        url = url.replace(f"/{s}.", f"/{size}.")
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=20,
                    help="max observations per genus")
    ap.add_argument("--out", default="imgs/inat_candidates")
    ap.add_argument("--size", default="large",
                    choices=["medium", "large", "original"])
    ap.add_argument("--all-photos", action="store_true",
                    help="download every photo, not just the first")
    ap.add_argument("--global", dest="global_", action="store_true",
                    help="do not restrict to New Zealand (not recommended)")
    ap.add_argument("--genera", default=None,
                    help="comma-separated subset; default is all in data.yaml")
    args = ap.parse_args()

    names = yaml.safe_load(open(ROOT / "imgs/detection_set/data.yaml"))["names"]
    if args.genera:
        wanted = {g.strip().lower() for g in args.genera.split(",")}
        names = [n for n in names if n.lower() in wanted]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "attribution.csv"
    place = None if args.global_ else NZ_PLACE

    seen = set()
    if meta_path.exists():                       # resume support
        with open(meta_path, encoding="utf-8") as f:
            seen = {r["filename"] for r in csv.DictReader(f)}
        print(f"resuming: {len(seen)} already downloaded")

    new_file = not meta_path.exists()
    fh = open(meta_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["filename", "genus", "observation_id", "photo_id",
                         "licence", "attribution", "observed_on",
                         "observation_url"])

    total = 0
    for genus in names:
        obs = fetch_observations(genus, args.cap, place, REUSE)
        if not obs:
            print(f"{genus:<18} none")
            continue
        gdir = out_dir / genus
        gdir.mkdir(exist_ok=True)
        n = 0
        for o in obs:
            photos = o.get("photos", [])
            if not args.all_photos:
                photos = photos[:1]
            for idx, p in enumerate(photos):
                fname = f"{genus}_{o['id']}_{p['id']}.jpg"
                if fname in seen or (gdir / fname).exists():
                    continue
                url = photo_url(p, args.size)
                if not url:
                    continue
                try:
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        (gdir / fname).write_bytes(r.read())
                except Exception as e:
                    print(f"    download failed {fname}: {e}")
                    continue
                writer.writerow([fname, genus, o["id"], p["id"],
                                 p.get("license_code", ""),
                                 p.get("attribution", ""),
                                 o.get("observed_on", ""),
                                 f"https://www.inaturalist.org/observations/{o['id']}"])
                n += 1; total += 1
                time.sleep(0.35)          # be polite to the CDN
        fh.flush()
        print(f"{genus:<18} {n:>4} images", flush=True)
        time.sleep(1.1)

    fh.close()
    print(f"\n{total} images -> {out_dir}")
    print(f"attribution: {meta_path}")
    print("\nNOTE: these still need pronotum+elytra boxes before training use.")


if __name__ == "__main__":
    main()
