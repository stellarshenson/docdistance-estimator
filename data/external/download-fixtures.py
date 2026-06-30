#!/usr/bin/env python3
"""Download the external fixture PDFs for docdistance.

The PDFs under ``data/external/`` are gitignored (binaries); only their ``*_origin.md``
provenance companions and this script are tracked. Run this to fetch the PDFs into
``data/external/`` from their recorded source URLs. Source, licence and purpose for each
file live in the matching ``*_origin.md`` beside it - the PDFs are retained solely for
calibration and test-fixture construction, not redistributed and not used for training.

Usage:
    python data/external/download-fixtures.py           # download any missing file
    python data/external/download-fixtures.py --force    # re-download everything
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Each fixture: target filename, source URL, and a provenance note. Keep the filename
# byte-identical to what the rest of the pipeline expects.
FIXTURES = [
    {
        "filename": "FINAL-Impact-of-AI-on-society.pdf",
        "url": "https://wergelandcentre.org/content/uploads/2023/01/FINAL-Impact-of-AI-on-society.pdf",
        "note": "Direct PDF, hosted by The European Wergeland Centre (wergelandcentre.org).",
    },
    {
        "filename": "Data Suggests Growth in Enterprise Adoption of AI is Due to Widespread Deploymen.pdf",
        "url": "https://newsroom.ibm.com/2024-01-10-Data-Suggests-Growth-in-Enterprise-Adoption-of-AI-is-Due-to-Widespread-Deployment-by-Early-Adopters",
        "note": ("IBM Newsroom article page (HTML), not a direct PDF link. The local fixture PDF is a "
                 "print-to-PDF render of this page; this URL will return HTML, so save the page as PDF "
                 "from a browser to reproduce the exact fixture."),
    },
]

# Some hosts (newsroom CDNs) reject the default urllib User-Agent with a 403.
UA = "Mozilla/5.0 (X11; Linux x86_64) docdistance-fixture-downloader"


def download(url: str, dest: Path) -> int:
    """Stream a URL to dest in chunks; return the byte count written."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    total = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
            total += len(chunk)
    return total


def is_pdf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download docdistance external fixture PDFs.")
    ap.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = ap.parse_args()

    rc = 0
    for fx in FIXTURES:
        dest = HERE / fx["filename"]
        if dest.exists() and not args.force:
            print(f"[skip] {fx['filename']} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[get ] {fx['filename']}")
        print(f"       {fx['url']}")
        try:
            n = download(fx["url"], dest)
        except Exception as e:  # network / 403 / 404 - report and move on
            print(f"[FAIL] {fx['filename']}: {e}")
            rc = 1
            continue
        if is_pdf(dest):
            print(f"[ok  ] {fx['filename']} ({n:,} bytes)")
        else:
            print(f"[warn] {fx['filename']}: downloaded {n:,} bytes but it is not a PDF")
            print(f"       {fx['note']}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
