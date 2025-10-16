#!/usr/bin/env python3
"""Repair STAC collection links to reference the shared catalog.

This utility inspects every `collection.json` living under the provided
catalog directory (except the root `catalog.json`). For each collection it:
  * Ensures `root` and `parent` links point back to the shared catalog.
  * Preserves existing non-root/parent links (e.g., item/self references).
  * Adds the collection as a `child` entry in the catalog with a stable title.

It mirrors the linking behaviour implemented in
`scripts/geo_utils/build_stac_catalog.sh` for cases where the catalog file did
not exist at generation time and needs to be linked afterwards.

Author: Juan Luis Herrera Cortijo <juan.luis.herrera.cortijo@gmail.com>
Created: 2025-10-16
Disclaimer: This script is provided as-is for research workflows and must be validated before any operational deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def ensure_relative(from_dir: Path, to_path: Path) -> str:
    """Return a POSIX-style relative href between two paths.

    The helper mirrors the logic used by STAC tools when writing link hrefs,
    prefixing dot paths so that catalog readers interpret them relative to the
    current object directory.
    """
    rel = os.path.relpath(to_path, start=from_dir)
    if not rel.startswith("."):
        rel = f"./{rel}"
    return rel.replace(os.sep, "/")


def load_json(path: Path) -> Dict:
    """Read a UTF-8 encoded JSON document into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Dict) -> None:
    """Persist a JSON payload with stable indentation and trailing newline."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_collection(collection_path: Path, catalog_path: Path) -> Tuple[bool, str]:
    """Ensure collection links reference the shared catalog; return modified flag.

    Parameters
    ----------
    collection_path:
        Location of the collection metadata that needs link repair.
    catalog_path:
        Path to the shared catalog.json acting as parent/root reference.
    """
    collection = load_json(collection_path)
    parent_href = ensure_relative(collection_path.parent, catalog_path)

    links: List[Dict] = collection.get("links", [])
    # Keep any existing links that are neither `root` nor `parent` so that item
    # pagination, self references, or custom relations survive the repair pass.
    existing_non_parent = [link for link in links if link.get("rel") not in {"root", "parent"}]

    new_links = existing_non_parent.copy()
    new_links.append({"rel": "parent", "href": parent_href, "type": "application/json"})
    new_links.append({"rel": "root", "href": parent_href, "type": "application/json"})

    changed = links != new_links
    if changed:
        collection["links"] = new_links
        dump_json(collection_path, collection)

    preferred_title = collection.get("title") or collection.get("id") or collection_path.parent.name
    return changed, preferred_title


def update_catalog_children(
    catalog_path: Path,
    additions: Iterable[Tuple[str, str]],
) -> bool:
    """Add or refresh child links for the provided (href, title) pairs.

    Parameters
    ----------
    catalog_path:
        Shared catalog whose `links` array is being reconciled.
    additions:
        Iterable of `(href, title)` tuples extracted from the collections.
    """
    catalog = load_json(catalog_path)
    links: List[Dict] = catalog.get("links", [])

    non_child: List[Dict] = []
    child_map: Dict[str, Dict] = {}

    # Separate child relations from the remaining links so we can rebuild the
    # child section deterministically without disturbing other metadata.
    for link in links:
        if link.get("rel") == "child":
            child_map[link.get("href")] = link
        else:
            non_child.append(link)

    mutated = False
    for child_href, child_title in additions:
        record = child_map.get(child_href, {"rel": "child", "href": child_href, "type": "application/json"})
        if record.get("title") != child_title:
            record["title"] = child_title
            mutated = True
        if child_href not in child_map:
            child_map[child_href] = record
            mutated = True

    ordered_children = [child_map[href] for href in sorted(child_map)]
    if links != non_child + ordered_children:
        catalog["links"] = non_child + ordered_children
        dump_json(catalog_path, catalog)
        mutated = True

    return mutated


def iter_collections(catalog_root: Path) -> Iterable[Path]:
    """Yield paths to collection.json files directly under the catalog root."""
    for path in catalog_root.glob("*/collection.json"):
        if path.is_file():
            yield path


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse CLI parameters and default to the repository-level catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog_dir",
        nargs="?",
        default="catalogs",
        help="Directory containing the shared catalog.json (default: catalogs)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    """CLI entry point returning zero on success."""
    args = parse_args(argv)
    root_dir = Path(args.catalog_dir).resolve()
    catalog_path = root_dir / "catalog.json"

    if not catalog_path.is_file():
        print(f"[ERROR] No catalog.json found at {catalog_path}", file=sys.stderr)
        return 1

    child_pairs: List[Tuple[str, str]] = []
    collection_changes = 0

    # Iterate over every collection, repairing links and caching titles so that
    # the parent catalog can reference them consistently.
    for collection_path in iter_collections(root_dir):
        changed, title = update_collection(collection_path, catalog_path)
        if changed:
            collection_changes += 1
        child_href = ensure_relative(root_dir, collection_path)
        child_pairs.append((child_href, title))

    # Refresh the catalog children in a deterministic order to keep diffs tidy.
    catalog_changed = update_catalog_children(catalog_path, child_pairs)

    print(
        f"Updated {collection_changes} collection(s); "
        f"{'modified' if catalog_changed else 'no changes to'} catalog.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
