"""
download_dataset.py — Download Flickr8k from Kaggle
====================================================
Run this script once to download and extract the Flickr8k dataset
into the correct folder expected by config.py:

    dataset/Flickr8k/
        Images/          ← ~8 000 .jpg files
        captions.txt     ← image–caption pairs

Requirements
------------
    pip install kaggle

Kaggle API credentials
----------------------
1. Go to https://www.kaggle.com/settings/account  → "Create New Token"
2. A file  kaggle.json  is downloaded — it contains:
       {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
3. Either:
   (a) Place kaggle.json at  ~/.kaggle/kaggle.json   (Linux/Mac)
                          or  C:/Users/<you>/.kaggle/kaggle.json  (Windows)
   (b) OR set the two environment variables below before running this script.
"""

import os
import zipfile
from pathlib import Path

# ── Optional: hard-code credentials here if you cannot use kaggle.json ──────
# Leave as empty strings to rely on ~/.kaggle/kaggle.json instead.
KAGGLE_USERNAME = "Royalline Technology"   # e.g. "johndoe"
KAGGLE_KEY      = "KGAT_a865840f21541ee79e752925518d9b4f"   # e.g. "abc123..."


# ── Destination (must match config.py) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEST_DIR     = PROJECT_ROOT / "dataset" / "Flickr8k"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# ── Kaggle dataset slug ──────────────────────────────────────────────────────
DATASET_SLUG = "adityajn105/flickr8k"


def main() -> None:
    # Inject credentials into env if provided above
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"]      = KAGGLE_KEY

    # Validate that credentials are available (env var or kaggle.json)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env     = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    if not has_env and not kaggle_json.exists():
        raise EnvironmentError(
            "Kaggle credentials not found.\n"
            "Either fill in KAGGLE_USERNAME / KAGGLE_KEY above,\n"
            f"or place your kaggle.json at: {kaggle_json}"
        )

    try:
        import kaggle  # noqa: F401 — verify package is installed
    except ImportError:
        raise ImportError(
            "'kaggle' package not installed.\n"
            "Run:  pip install kaggle"
        )

    import subprocess, shutil

    kaggle_bin = shutil.which("kaggle")

    print(f"Downloading '{DATASET_SLUG}' → {DEST_DIR} …")

    if kaggle_bin:
        # Use the installed kaggle CLI binary
        subprocess.run(
            [kaggle_bin, "datasets", "download",
             "-d", DATASET_SLUG, "-p", str(DEST_DIR)],
            check=True,
        )
    else:
        # Fall back to the Python API (works for older kaggle versions)
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_SLUG, path=str(DEST_DIR), unzip=False, quiet=False
        )

    # ── Unzip ────────────────────────────────────────────────────────────────
    zip_files = list(DEST_DIR.glob("*.zip"))
    if not zip_files:
        print("No zip file found — dataset may already be extracted.")
        return

    for zf_path in zip_files:
        print(f"Extracting {zf_path.name} …")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(DEST_DIR)
        zf_path.unlink()   # remove zip to save space
        print(f"  Deleted {zf_path.name}")

    # ── Verify expected structure ────────────────────────────────────────────
    images_dir   = DEST_DIR / "Images"
    captions_file = DEST_DIR / "captions.txt"

    img_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    print()
    print("Dataset ready:")
    print(f"  Images dir   : {images_dir}  ({img_count} .jpg files)")
    print(f"  Captions file: {captions_file}  (exists={captions_file.exists()})")

    if img_count == 0 or not captions_file.exists():
        print()
        print("WARNING: Expected files not found in the standard location.")
        print("The Kaggle zip may have a sub-folder. Contents of DEST_DIR:")
        for item in sorted(DEST_DIR.iterdir()):
            print(f"  {item}")


if __name__ == "__main__":
    main()
