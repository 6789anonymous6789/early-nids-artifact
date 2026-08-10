#!/usr/bin/env python3
"""Download the CIC-IoT2023 dataset (PCAPs + CSVs) from the CIC portal.

Access model (verified 2026-05-31)
----------------------------------
CIC-IoT2023 lives behind the same PHP gateway as CIC-IDS2017, NOT on the
open S3 bucket used for CIC-IDS2018. The gateway is:

    https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/

Flow:
  1. GET  <root>/                      -> opens a PHP session (Token cookie)
  2. POST <root>/insert.php            -> registration form; AUTHORISES the
     (first_name,last_name,email,         calling IP for this session
      institution,job_title,country)
  3. GET  <root>/browse.php?p=<path>   -> directory listing (JSON-ish HTML)
  4. GET  <root>/download.php?file=<p> -> file bytes

IP BINDING & TOKEN TTL
----------------------
The Token cookie is bound to the client IP and expires after 24h
(Max-Age=86400). Because we register *from this server*, the Token is
bound to the server's egress IP automatically -- no laptop / SSH tunnel
needed. For long downloads the script re-registers automatically whenever
a request is bounced (HTTP 403 from browse.php, or a 302 back to the
gateway from download.php).

NO RESUME
---------
download.php streams with chunked transfer and *ignores Range requests*
(returns 200, not 206; no Accept-Ranges; no Content-Length). There is no
partial-file resume. Resilience therefore comes from file granularity:
each .pcap (<=2 GB) / .csv is an atomic unit, downloaded to a .part file
and renamed on success. Already-present final files are skipped.

Usage
-----
    python scripts/cic_iot2023/download.py --pcap          # all 309 PCAPs
    python scripts/cic_iot2023/download.py --csv           # MERGED + per-class CSVs
    python scripts/cic_iot2023/download.py --pcap --csv    # everything
    python scripts/cic_iot2023/download.py --pcap --dry-run
    python scripts/cic_iot2023/download.py --only PCAP/Benign_Final   # one subtree

Registration identity (override via flags or env if needed):
    --first/--last/--email/--institution/--job/--country
"""
from __future__ import annotations

import argparse
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, quote_plus, unquote_plus

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "CIC-IoT2023" / "original"

ROOT = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/"
BROWSE = ROOT + "browse.php?p="
DOWNLOAD = ROOT + "download.php?file="
INSERT = ROOT + "insert.php"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36")

# The portal requires a registration form before it will issue a session
# cookie. Supply your own details through the environment rather than editing
# this file:  export CIC_REG_EMAIL=you@example.org  (and the rest as needed).
DEFAULT_REG = dict(
    first_name=os.environ.get("CIC_REG_FIRST_NAME", "Jane"),
    last_name=os.environ.get("CIC_REG_LAST_NAME", "Doe"),
    email=os.environ.get("CIC_REG_EMAIL", "jane.doe@example.org"),
    institution=os.environ.get("CIC_REG_INSTITUTION", "Example University"),
    job_title=os.environ.get("CIC_REG_JOB_TITLE", "Researcher"),
    country=os.environ.get("CIC_REG_COUNTRY", "United States"),
)


class _LinkParser(HTMLParser):
    """Collect browse.php?p= and download.php?file= hrefs from a listing."""

    def __init__(self) -> None:
        super().__init__()
        self.folders: list[str] = []   # decoded p= values
        self.files: list[str] = []     # decoded file= values

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        if "browse.php?p=" in href:
            self.folders.append(unquote_plus(href.split("browse.php?p=", 1)[1]))
        elif "download.php?file=" in href:
            self.files.append(unquote_plus(href.split("download.php?file=", 1)[1]))


class CICPortal:
    def __init__(self, reg: dict) -> None:
        self.reg = reg
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.registered = False

    def register(self) -> None:
        """Open a session and authorise this IP via the registration form."""
        self.s.get(ROOT, timeout=60)  # seed Token cookie
        r = self.s.post(INSERT, data=self.reg,
                        headers={"Referer": ROOT}, timeout=60)
        ok = r.headers.get("content-type", "").startswith("application/json")
        msg = r.text[:120].replace("\n", " ")
        if r.status_code != 200:
            raise RuntimeError(f"registration failed: HTTP {r.status_code} {msg}")
        print(f"[auth] registered as {self.reg['email']} -> {msg}")
        self.registered = True

    def _ensure(self) -> None:
        if not self.registered:
            self.register()

    def browse(self, path: str) -> _LinkParser:
        self._ensure()
        url = BROWSE + quote_plus(path)
        r = self.s.get(url, headers={"Referer": BROWSE}, timeout=60)
        if r.status_code in (401, 403):
            self.register()
            r = self.s.get(url, headers={"Referer": BROWSE}, timeout=60)
        r.raise_for_status()
        p = _LinkParser()
        p.feed(r.text)
        return p

    def crawl(self, root: str) -> list[str]:
        """Recursively collect every download.php file= path under `root`."""
        files: list[str] = []
        stack = [root]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            lp = self.browse(cur)
            files.extend(f for f in lp.files if f.startswith(root))
            stack.extend(f for f in lp.folders
                         if f not in seen and f.startswith(root))
            print(f"[crawl] {cur}: +{len(lp.files)} files, +{len(lp.folders)} dirs "
                  f"(total files {len(files)})")
        # de-dup, keep order
        # skip docs and the monolithic *.zip bundles (no-resume -> fetch
        # the individual files inside the tree instead)
        out, s = [], set()
        for f in files:
            if f not in s and not f.lower().endswith((".pdf", ".zip")):
                out.append(f); s.add(f)
        return out

    def download(self, file_path: str, dest: Path) -> bool:
        """Stream one file to dest atomically. Returns True if downloaded."""
        if dest.exists() and dest.stat().st_size > 0:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        url = DOWNLOAD + quote_plus(file_path)
        for attempt in range(1, 4):
            try:
                self._ensure()
                with self.s.get(url, headers={"Referer": BROWSE},
                                stream=True, timeout=(30, 600)) as r:
                    # bounced back to gateway -> token died, re-register
                    if r.status_code != 200 or "download.php" not in r.url:
                        self.register()
                        continue
                    n = 0
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk); n += 1
                if tmp.stat().st_size == 0:
                    raise RuntimeError("empty download")
                tmp.rename(dest)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"  ! attempt {attempt}/3 failed for {file_path}: {e}")
                time.sleep(5 * attempt)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"giving up on {file_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", action="store_true", help="download PCAP/ tree")
    ap.add_argument("--csv", action="store_true", help="download CSV/ tree")
    ap.add_argument("--only", help="restrict to a single subtree, e.g. PCAP/Benign_Final")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="list files, don't download")
    ap.add_argument("--first", default=DEFAULT_REG["first_name"])
    ap.add_argument("--last", default=DEFAULT_REG["last_name"])
    ap.add_argument("--email", default=DEFAULT_REG["email"])
    ap.add_argument("--institution", default=DEFAULT_REG["institution"])
    ap.add_argument("--job", default=DEFAULT_REG["job_title"])
    ap.add_argument("--country", default=DEFAULT_REG["country"])
    args = ap.parse_args()

    roots: list[str] = []
    if args.only:
        roots = [args.only.strip("/")]
    else:
        if args.pcap:
            roots.append("PCAP")
        if args.csv:
            roots.append("CSV")
    if not roots:
        ap.error("choose at least one of --pcap / --csv / --only")

    reg = dict(first_name=args.first, last_name=args.last, email=args.email,
               institution=args.institution, job_title=args.job, country=args.country)
    portal = CICPortal(reg)

    all_files: list[str] = []
    for r in roots:
        all_files.extend(portal.crawl(r))
    print(f"\n[plan] {len(all_files)} files to fetch into {args.out}\n")

    if args.dry_run:
        for f in all_files:
            print("   ", f)
        return 0

    done = skipped = 0
    t0 = time.time()
    for i, f in enumerate(all_files, 1):
        dest = args.out / f  # f is a relative path like PCAP/<class>/<file>.pcap
        if portal.download(f, dest):
            done += 1
            print(f"[{i}/{len(all_files)}] OK  {f}  ({dest.stat().st_size/1e6:.1f} MB)")
        else:
            skipped += 1
            if skipped % 25 == 0 or i == len(all_files):
                print(f"[{i}/{len(all_files)}] skip (exists)  {f}")
    dt = time.time() - t0
    print(f"\n[done] downloaded {done}, skipped {skipped}, in {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
