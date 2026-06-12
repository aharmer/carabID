"""
zenodo_deposit.py
-----------------
Upload an image dataset to Zenodo (or the sandbox for testing).

Usage:
    python zenodo_deposit.py

Set ZENODO_TOKEN in your environment before running.
Switch ZENODO_URL to the sandbox URL while testing.
"""

import os
import sys
import zipfile
import yaml
import requests
from pathlib import Path

# ── Configuration ───────────────────────

# Switch to sandbox.zenodo.org for testing; change to zenodo.org when ready
# ZENODO_URL = "https://sandbox.zenodo.org/api"
ZENODO_URL = "https://zenodo.org/api"

ACCESS_TOKEN = os.environ.get("ZENODO_TOKEN")
if not ACCESS_TOKEN:
    sys.exit("Error: ZENODO_TOKEN environment variable not set.")

# Root folder containing train/, valid/, test/ subdirectories
DATA_DIR = Path("D:/Dropbox/data/carabID/imgs/detection_set")

# Where to write the zip before uploading (deleted after upload)
OUTPUT_ZIP = Path("D:/Dropbox/data/carabID/imgs/dataset.zip")

# File extensions to exclude from the zip
EXCLUDE_EXTENSIONS = set()

# ── Metadata ───────────────────────
# See https://developers.zenodo.org/#representation for all available fields.

METADATA = {
    "title": "New Zealand carabid beetle images for genus-level identification",
    "upload_type": "dataset",
    "description": (
        "Lab images of New Zealand carabid beetles annotated for genus-level identification "
        "using a YOLO-based object detection and classification pipeline. "
        "Labels are provided in YOLO format (bounding box coordinates and genus "
        "class index per image). Dataset is pre-split into training, validation, "
        "and test sets, each containing an images/ and labels/ subdirectory."
    ),
    "creators": [
        {
            "name": "Harmer, Aaron",          # format: "Surname, Firstname"
            "affiliation": "New Zealand Institute of Bioeconomy Science",
            "orcid": "0000-0002-2244-5129",  # optional but recommended
        }
    ],
    "license": "cc-by-4.0",             # or "cc-by-nc-4.0", "cc0-1.0", etc.
    "access_right": "open",
    "keywords": [
        "Carabidae",
        "Coleoptera",
        "entomology",
        "object detection",
        "image classification",
        "YOLO",
        "New Zealand",
    ],
    # "related_identifiers": [           # link to your paper once published
    #     {
    #         "identifier": "10.xxxx/xxxxx",
    #         "relation": "isSupplementTo",
    #         "scheme": "doi",
    #     }
    # ],
    # "version": "1.0.0",
    # "language": "eng",
}

# ── Helpers ───────────────────────

def api(method, path, **kwargs):
    """Thin wrapper: always injects the access token and raises on error."""
    url = f"{ZENODO_URL}{path}"
    params = kwargs.pop("params", {})
    params["access_token"] = ACCESS_TOKEN
    r = requests.request(method, url, params=params, **kwargs)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print(f"HTTP {r.status_code} on {method.upper()} {path}")
        print(r.text)
        raise
    return r


def clean_yaml_for_deposit(file_path):
    """
    Read a YOLO dataset YAML and return cleaned content as bytes.
    - Removes the 'path' key (absolute local root path).
    - Converts any absolute train/val/test paths to relative ones
      by keeping only the last two path components (e.g. 'train/images').
    """
    with open(file_path) as f:
        data = yaml.safe_load(f)

    data.pop("path", None)

    for key in ("train", "val", "test"):
        if key in data and isinstance(data[key], str):
            p = Path(data[key])
            if p.is_absolute():
                data[key] = str(Path(*p.parts[-2:]))

    return yaml.dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")


def zip_dataset(source_dir, output_zip, exclude_extensions):
    """
    Zip source_dir into output_zip, preserving the folder structure.
    Skips files whose extension is in exclude_extensions.
    YAML files are cleaned of hardcoded paths before being added.
    """
    source_dir = Path(source_dir)
    files = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() not in exclude_extensions
    ]
    print(f"  Found {len(files)} files to zip (excluding {exclude_extensions}).")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, file in enumerate(files, 1):
            arcname = file.relative_to(source_dir.parent)  # keeps e.g. data/train/images/img.jpg
            if file.suffix.lower() in {".yaml", ".yml"}:
                cleaned = clean_yaml_for_deposit(file)
                zf.writestr(str(arcname), cleaned)
                print(f"  Cleaned and added {file.name}.")
            else:
                zf.write(file, arcname)
            if i % 500 == 0:
                print(f"    {i}/{len(files)} files zipped...")
    size_mb = output_zip.stat().st_size / 1e6
    print(f"  Created {output_zip} ({size_mb:.1f} MB).\n")


def upload_file(bucket_url, file_path):
    """Upload a single file to the deposition bucket."""
    path = Path(file_path)
    size = path.stat().st_size
    print(f"  Uploading {path.name} ({size / 1e6:.1f} MB) ...", end=" ", flush=True)
    with open(path, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{path.name}",
            data=f,
            params={"access_token": ACCESS_TOKEN},
        )
    r.raise_for_status()
    print("done.")
    return r.json()


# ── Main workflow ───────────────────────

def main():
    # 1. Create a new (empty) deposition
    print("Creating deposition...")
    r = api("post", "/deposit/depositions", json={}, headers={"Content-Type": "application/json"})
    deposition = r.json()
    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    print(f"  Deposition ID: {deposition_id}")
    print(f"  Draft URL: {deposition['links']['html']}\n")

    # 2. Zip the dataset
    print(f"Zipping {DATA_DIR} -> {OUTPUT_ZIP} ...")
    if not DATA_DIR.exists():
        sys.exit(f"Error: DATA_DIR '{DATA_DIR}' not found.")
    zip_dataset(DATA_DIR, OUTPUT_ZIP, EXCLUDE_EXTENSIONS)

    # 3. Upload the zip
    print("Uploading dataset.zip...")
    upload_file(bucket_url, OUTPUT_ZIP)
    print()

    # 4. Clean up local zip
    OUTPUT_ZIP.unlink()
    print(f"  Removed local {OUTPUT_ZIP}.\n")

    # 5. Set metadata
    print("Setting metadata...")
    api(
        "put",
        f"/deposit/depositions/{deposition_id}",
        json={"metadata": METADATA},
        headers={"Content-Type": "application/json"},
    )
    print("  Metadata saved.\n")

    # 6. Summary — do NOT publish automatically
    print("=" * 60)
    print("Deposit ready for review.")
    print(f"  View/edit/publish at: {deposition['links']['html']}")
    print()
    print("When you're happy with everything, either:")
    print("  a) Publish via the Zenodo web UI (recommended for first time), or")
    print("  b) Uncomment the publish block in this script and re-run.")
    print()
    print("NOTE: Publishing is PERMANENT on the real site.")
    print("=" * 60)

    # 7. Publish (uncomment when ready — irreversible on zenodo.org!)
    print("Publishing...")
    r = api("post", f"/deposit/depositions/{deposition_id}/actions/publish")
    doi = r.json()["doi"]
    print(f"  Published! DOI: {doi}")


if __name__ == "__main__":
    main()
