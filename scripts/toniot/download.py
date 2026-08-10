#!/usr/bin/env python3
"""Download the full ToN_IoT dataset from the UNSW SharePoint share.

ToN_IoT is distributed by the same UNSW Canberra group (and from the same
``z5025758`` OneDrive-for-Business account) as UNSW-NB15, behind an anonymous
SharePoint/OneDrive shared folder. The official project page
(https://research.unsw.edu.au/projects/toniot-datasets) links to it via its
"download from HERE" anchor.

This mirrors the mechanics of ``scripts/unsw_nb15/download.py`` but, instead of
walking a fixed set of target subfolders, it recursively enumerates the ENTIRE
share tree (the full dataset: Raw_datasets pcap/Zeek/telemetry,
Processed_datasets, Train_Test_datasets, Description_stats_datasets and
Security_Events_GroundTruth_datasets) and downloads every file, preserving the
folder layout.

  1. Headless Chromium opens the share URL. Loading the page sets the
     anonymous-share cookies and lets the OneDrive SPA mint per-item
     access tokens for us.
  2. We harvest the root folder's ``spItemUrl`` from the SPA's
     ``RenderListDataAsStream`` response, then recursively walk
     ``_api/v2.0/drives/.../children`` capturing every file's full relative
     ``path``, ``size`` and a presigned ``@content.downloadUrl`` (~1h TTL).
  3. We close the browser and stream each file with ``aiohttp`` (4-way
     concurrency by default), with size-match resume, atomic ``.part``
     rename, pcap magic validation (only for ``*.pcap``), and
     exponential-backoff retry. Presigned URLs expire after ~1h, so a large
     transfer fans out across multiple enumerate+download rounds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import aiohttp
from playwright.sync_api import Response, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "ToN-IoT" / "original"

# Public anonymous share (the "HERE" link on
# https://research.unsw.edu.au/projects/toniot-datasets). Loading this in a
# browser issues the anonymous-share cookies / per-item access tokens we need;
# the authenticated ``_layouts/15/onedrive.aspx?id=...`` UI URL would redirect
# a headless browser to login.
DEFAULT_SHARE_URL = (
    "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
    "EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW"
)

# Pcap / pcapng file magic numbers, used to reject HTML error pages and
# placeholder text masquerading as pcaps.
PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1",  # classic pcap, little-endian
    b"\xa1\xb2\xc3\xd4",  # classic pcap, big-endian
    b"\x4d\x3c\xb2\xa1",  # nanosecond-precision pcap, little-endian
    b"\xa1\xb2\x3c\x4d",  # nanosecond-precision pcap, big-endian
    b"\x0a\x0d\x0d\x0a",  # pcapng (any endianness — block type 0x0A0D0D0A)
}

CHUNK_BYTES = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Phase 1: recursively enumerate downloadable items via headless browser
# ---------------------------------------------------------------------------


_RENDER_RE = re.compile(r"RenderListDataAsStream", re.IGNORECASE)


def _looks_like_render_response(resp: Response) -> bool:
    return bool(_RENDER_RE.search(resp.url)) and resp.ok


# JS executed inside the page to recursively walk the v2.0 driveItem tree
# using the anonymous-share session cookies SharePoint set during page load.
# Given the share root's ``spItemUrl``, depth-first walks the whole tree and
# returns one entry per file: {path (full relative path), size, url}.
_JS_WALK = r"""
async ({rootSpItemUrl, maxDepth}) => {
  // ``spItemUrl`` looks like:
  //   https://<host>:443/_api/v2.0/drives/<drive>/items/<item>?version=Published
  const m = rootSpItemUrl.match(
    /^(https?:\/\/[^/]+\/_api\/v2\.0\/drives\/[^/]+)\/items\/([^?]+)(\?.*)?$/
  );
  if (!m) throw new Error("could not parse spItemUrl: " + rootSpItemUrl);
  const drivePrefix = m[1];
  const rootId = m[2];
  const qs = m[3] || "?version=Published";

  function itemUrl(id, suffix) {
    return drivePrefix + "/items/" + id + (suffix || "") + qs + "&$top=1000";
  }

  async function children(id) {
    const r = await fetch(itemUrl(id, "/children"),
      {headers: {Accept: "application/json"}});
    if (!r.ok) throw new Error("children " + r.status + " for id=" + id);
    const data = await r.json();
    let value = data.value || [];
    let next = data["@odata.nextLink"];
    while (next) {
      const r2 = await fetch(next, {headers: {Accept: "application/json"}});
      if (!r2.ok) throw new Error("children-next " + r2.status);
      const d2 = await r2.json();
      value = value.concat(d2.value || []);
      next = d2["@odata.nextLink"];
    }
    return value;
  }

  const out = [];
  async function walk(id, prefix, depth) {
    if (depth > maxDepth) throw new Error("maxDepth exceeded at " + prefix);
    const kids = await children(id);
    for (const f of kids) {
      const rel = prefix ? (prefix + "/" + f.name) : f.name;
      if (f.folder) {
        await walk(f.id, rel, depth + 1);
      } else {
        const url = f["@content.downloadUrl"] || f["@microsoft.graph.downloadUrl"];
        if (!url) continue;
        out.push({path: rel, size: f.size, url: url});
      }
    }
  }
  await walk(rootId, "", 0);
  return out;
}
"""


def enumerate_via_browser(share_url: str, *, timeout_s: int = 90,
                          max_depth: int = 12) -> list[dict]:
    """Open the share in headless Chromium and return per-file download info
    for the ENTIRE tree.

    Returns a list of dicts ``{path, size, url}`` where ``path`` is the full
    path relative to the share root and ``url`` is a presigned
    ``@content.downloadUrl`` valid for ~1 hour.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
            )
        )
        page = ctx.new_page()

        with page.expect_response(
            _looks_like_render_response, timeout=timeout_s * 1000
        ) as resp_info:
            page.goto(share_url, wait_until="domcontentloaded",
                      timeout=timeout_s * 1000)

        if "login.microsoftonline.com" in page.url:
            raise SystemExit(
                "ERROR: SharePoint redirected to login. Pass a fresh public "
                "':f:/g/...' share URL via --share-url or TONIOT_SHARE_URL."
            )

        first = resp_info.value.json()
        list_data = first.get("ListData") or {}
        root_spitemurl = (
            list_data.get("CurrentFolderSpItemUrl")
            or (list_data.get("Row") or [{}])[0].get(".spItemUrl")
        )
        if not root_spitemurl:
            raise SystemExit(
                "could not extract root spItemUrl from RenderListDataAsStream "
                "response (SharePoint UI may have changed)."
            )
        print(f"[enum] root spItemUrl: {root_spitemurl}", file=sys.stderr, flush=True)

        items = page.evaluate(
            _JS_WALK,
            {"rootSpItemUrl": root_spitemurl, "maxDepth": max_depth},
        )

        # Summarize by top-level folder for a quick sanity read.
        by_top: dict[str, list[int]] = {}
        for it in items:
            top = it["path"].split("/", 1)[0]
            by_top.setdefault(top, []).append(it["size"])
        for top in sorted(by_top):
            sizes = by_top[top]
            gib = sum(sizes) / (1 << 30)
            print(f"[enum] {top!r}: {len(sizes)} files, {gib:.2f} GiB",
                  file=sys.stderr, flush=True)
        total_gib = sum(it["size"] for it in items) / (1 << 30)
        print(f"[enum] TOTAL: {len(items)} files, {total_gib:.2f} GiB",
              file=sys.stderr, flush=True)

        browser.close()

    return items


# ---------------------------------------------------------------------------
# Phase 2: parallel per-file downloads with resume + validation
# ---------------------------------------------------------------------------


async def _fetch_one(
    session: aiohttp.ClientSession,
    item: dict,
    out_dir: Path,
    *,
    max_attempts: int = 3,
) -> str:
    dest = out_dir / item["path"]
    if dest.exists() and dest.stat().st_size == item["size"]:
        return "skip"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            async with session.get(
                item["url"],
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120),
            ) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    async for chunk in r.content.iter_chunked(CHUNK_BYTES):
                        f.write(chunk)
            break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt + 1 == max_attempts:
                tmp.unlink(missing_ok=True)
                raise
            await asyncio.sleep(2 ** attempt)
    else:  # pragma: no cover - the for/else runs only if no break
        if last_err:
            raise last_err

    if dest.suffix.lower() == ".pcap":
        with tmp.open("rb") as f:
            magic = f.read(4)
        if magic not in PCAP_MAGICS:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"bad pcap magic for {item['path']!r}: {magic!r}"
            )

    if tmp.stat().st_size != item["size"]:
        actual = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"size mismatch for {item['path']!r}: got {actual}, expected {item['size']}"
        )

    os.replace(tmp, dest)
    return "ok"


async def _download_all(items: list[dict], out_dir: Path, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _go(it: dict) -> str:
            async with sem:
                try:
                    kind = await _fetch_one(session, it, out_dir)
                except Exception as e:
                    print(f"[fail] {it['path']}: {e}", file=sys.stderr, flush=True)
                    return "fail"
                size_mib = it["size"] / (1 << 20)
                tag = "skip" if kind == "skip" else "ok  "
                print(f"[{tag}] {it['path']:60s} {size_mib:8.1f} MiB",
                      flush=True)
                return kind

        results = await asyncio.gather(*(_go(it) for it in items))
        for r in results:
            counts[r] += 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--share-url",
        default=os.environ.get("TONIOT_SHARE_URL", DEFAULT_SHARE_URL),
        help="ToN_IoT SharePoint share URL "
        "(env: TONIOT_SHARE_URL; default: hardcoded UNSW share).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Local destination directory (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--concurrency", type=int, default=4,
        help="Parallel downloads (default: 4).",
    )
    p.add_argument(
        "--enumerate-only", action="store_true",
        help="Only run the recursive browser-enumeration phase and dump JSON "
             "to stdout. Useful for mapping the tree / sizes without starting "
             "a large transfer.",
    )
    p.add_argument(
        "--max-rounds", type=int, default=8,
        help="Maximum enumerate+download rounds before giving up (default: 8). "
             "Each round refreshes per-file presigned URLs (~1h TTL each), so a "
             "large run can fan out across multiple rounds without hitting "
             "expired tokens.",
    )
    args = p.parse_args()

    if not args.share_url:
        print("ERROR: --share-url or TONIOT_SHARE_URL is required.",
              file=sys.stderr)
        return 2

    if args.enumerate_only:
        print(f"[enum] opening {args.share_url}", file=sys.stderr, flush=True)
        items = enumerate_via_browser(args.share_url)
        if not items:
            print("ERROR: enumeration returned 0 files.", file=sys.stderr)
            return 3
        json.dump(items, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    last_remaining = None
    for round_no in range(1, args.max_rounds + 1):
        print(f"\n=== round {round_no}/{args.max_rounds}: enumerating ===",
              flush=True)
        items = enumerate_via_browser(args.share_url)
        if not items:
            print("ERROR: enumeration returned 0 files.", file=sys.stderr)
            return 3

        remaining = [
            it for it in items
            if not (
                (args.out_dir / it["path"]).exists()
                and (args.out_dir / it["path"]).stat().st_size == it["size"]
            )
        ]
        total_remaining = sum(it["size"] for it in remaining)
        print(
            f"[plan] {len(items)} files total, {len(remaining)} remaining, "
            f"{total_remaining / (1 << 30):.2f} GiB to fetch, "
            f"-> {args.out_dir}, concurrency={args.concurrency}",
            flush=True,
        )
        if not remaining:
            print("[done] nothing to do — all files present and sized correctly.",
                  flush=True)
            return 0

        counts = asyncio.run(
            _download_all(items, args.out_dir, args.concurrency)
        )
        print(
            f"[round {round_no}] OK: {counts['ok']}  SKIP: {counts['skip']}  "
            f"FAIL: {counts['fail']}",
            flush=True,
        )
        if counts["fail"] == 0:
            return 0
        if last_remaining is not None and len(remaining) >= last_remaining:
            print(
                f"[abort] no progress between rounds (still {len(remaining)} "
                "remaining). Inspect [fail] lines above and rerun.",
                file=sys.stderr, flush=True,
            )
            return 1
        last_remaining = len(remaining)

    print(f"[abort] hit --max-rounds={args.max_rounds} with files still missing.",
          file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
