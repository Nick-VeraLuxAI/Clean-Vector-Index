 #!/usr/bin/env python3
"""
One-shot cleaner for FAISS + JSON hybrid memory (type-safe, dedupe + prune + cap).
Also:
- Drops JSON entries with invalid/missing vector_id
- Verifies JSON vector_ids exist in FAISS; drops JSON orphans
- Removes FAISS vectors not present in final JSON (two-way sync)

Usage example (preview):
python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json "/path/to/longterm.json" \
  --subject-cap 3 \
  --dry-run
"""

import argparse, json, os, re, time, shutil, math
from collections import defaultdict
from typing import List, Dict, Any, Set
from decimal import Decimal, InvalidOperation

import numpy as np
import faiss

# ---------- helpers ----------

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def to_float(x, default=0.0):
    try:
        if x is None: return default
        v = float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except Exception:
        return default

def to_int(x, default=0):
    """Safe int64 parse without float() to avoid precision loss on large IDs."""
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            s = x.strip()
            # integer string
            if re.fullmatch(r'[+-]?\d+', s):
                return int(s)
            # decimal string – truncate toward zero
            if re.fullmatch(r'[+-]?\d+\.\d+', s):
                try:
                    return int(Decimal(s).to_integral_value(rounding="ROUND_DOWN"))
                except InvalidOperation:
                    return default
            return default
        # last resort (e.g., numpy int types)
        return int(x)
    except Exception:
        return default

def is_valid_vector_id(x) -> bool:
    """Valid if parseable to non-zero int64."""
    try:
        v = to_int(x, 0)
    except Exception:
        return False
    if v == 0:
        return False
    return -(1 << 63) <= v <= (1 << 63) - 1

def choose_better(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick the better memory between a and b.
    Preference: decided=True > newer timestamp > higher confidence > longer original
    """
    a_dec = bool(a.get("decided", True))
    b_dec = bool(b.get("decided", True))
    if a_dec != b_dec:
        return a if a_dec else b

    ta = to_float(a.get("timestamp"), 0.0)
    tb = to_float(b.get("timestamp"), 0.0)
    if ta != tb:
        return a if ta > tb else b

    ca = to_float(a.get("confidence"), 0.0)
    cb = to_float(b.get("confidence"), 0.0)
    if ca != cb:
        return a if ca > cb else b

    return a if len((a.get("original") or "")) >= len((b.get("original") or "")) else b

def backup_file(path: str):
    if not os.path.exists(path):
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    bpath = f"{path}.bak-{ts}"
    shutil.copy2(path, bpath)
    return bpath

def load_faiss_ids(index_path: str) -> Set[int]:
    """Load FAISS index and return the set of stored IDs. Supports IndexIDMap / IndexIDMap2."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    idx = faiss.read_index(index_path)

    ids_set: Set[int] = set()
    try:
        # In many builds, IndexIDMap exposes id_map as a vector<idx_t>
        arr = faiss.vector_to_array(idx.id_map)
        arr = np.asarray(arr, dtype=np.int64)  # ensure int64
        ids_set = set(int(v) for v in arr.tolist())
    except Exception:
        # Not an IDMap; skip existence filtering
        pass
    return ids_set

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="One-shot FAISS/JSON cleaner (dedupe + prune + cap + two-way sync).")
    ap.add_argument("--index", required=True, help="Path to FAISS index (e.g., ./memory/vector.index)")
    ap.add_argument("--json", required=True, help="Path to long-term memory JSON array (e.g., ./memory/longterm.json)")
    ap.add_argument("--subject-cap", type=int, default=3, help="Max items to keep per subject (default: 3). Use 0 to disable.")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="Drop items below this confidence (default: 0.0)")
    ap.add_argument("--drop-exact", nargs="*", default=[], help='Exact strings to drop (case/whitespace-insensitive).')
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak files for index/JSON.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change but do not modify files.")
    args = ap.parse_args()

    # Load JSON
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON not found: {args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of memory objects.")

    total_before = len(data)

    # Load FAISS IDs (for existence check)
    faiss_ids = load_faiss_ids(args.index)
    faiss_id_check_enabled = len(faiss_ids) > 0  # only validate existence if we could read ids

    # Filter: min confidence, require valid vector_id, and (optionally) existence in FAISS
    filtered = []
    invalid_id_count = 0
    low_conf_count = 0
    orphan_json_count = 0  # JSON entries whose id not found in FAISS
    for m in data:
        conf = to_float(m.get("confidence"), 0.0)
        if conf < float(args.min_confidence):
            low_conf_count += 1
            continue
        vid_raw = m.get("vector_id", None)
        if not is_valid_vector_id(vid_raw):
            invalid_id_count += 1
            continue
        vid = to_int(vid_raw)
        if faiss_id_check_enabled and vid not in faiss_ids:
            orphan_json_count += 1
            continue
        filtered.append(m)

    # Drop exact boilerplate strings
    drop_norms = set(normalize_text(s) for s in args.drop_exact)
    if drop_norms:
        filtered = [m for m in filtered if normalize_text(m.get("original", "")) not in drop_norms]

    # Dedupe by normalized original text
    by_text: Dict[str, Dict[str, Any]] = {}
    for m in filtered:
        key = normalize_text(m.get("original", ""))
        if key == "":
            key = f"__empty_{m.get('vector_id')}"
        if key not in by_text:
            by_text[key] = m
        else:
            by_text[key] = choose_better(by_text[key], m)

    dedup_list = list(by_text.values())

    # Subject cap (facet diversification)
    if args.subject_cap and int(args.subject_cap) > 0:
        subject_counts = defaultdict(int)
        kept = []
        dedup_list.sort(
            key=lambda m: (to_float(m.get("timestamp"), 0.0), to_float(m.get("confidence"), 0.0)),
            reverse=True
        )
        for m in dedup_list:
            subj = (m.get("subject") or "").strip().lower()
            if subject_counts[subj] < int(args.subject_cap):
                kept.append(m)
                subject_counts[subj] += 1
        dedup_list = kept

    # Determine IDs to keep/remove (type-safe)
    keep_ids = set(to_int(m.get("vector_id")) for m in dedup_list if "vector_id" in m)
    json_valid_ids = set(
        to_int(m.get("vector_id"))
        for m in data
        if "vector_id" in m and is_valid_vector_id(m.get("vector_id"))
    )
    # IDs to remove from FAISS are those valid JSON IDs not kept
    rm_ids = sorted(list(json_valid_ids - keep_ids))

    # Also remove FAISS vectors that aren't in final JSON (two-way sync)
    orphan_faiss_ids = []
    if faiss_id_check_enabled:
        orphan_faiss_ids = sorted(list(faiss_ids - keep_ids))

    # Stats
    print("=== Cleaner Summary ===")
    print(f"JSON entries before: {total_before}")
    print(f"Below min-confidence dropped: {low_conf_count}")
    print(f"Invalid/missing vector_id dropped: {invalid_id_count}")
    if faiss_id_check_enabled:
        print(f"JSON entries dropped (no matching vector in FAISS): {orphan_json_count}")
    else:
        print("FAISS id listing not available; skipping JSON→FAISS existence filter.")
    print(f"After min-confidence/drop-exact/valid-id/exists: {len(filtered)}")
    print(f"After dedupe: {len(by_text)}")
    if args.subject_cap and int(args.subject_cap) > 0:
        print(f"After subject cap ({args.subject_cap}/subject): {len(dedup_list)}")
    if faiss_id_check_enabled:
        print(f"FAISS vectors present: {len(faiss_ids)}")
    print(f"Final Keep IDs: {len(keep_ids)}")
    print(f"Remove IDs from FAISS (not kept but were valid in JSON): {len(rm_ids)}")
    if faiss_id_check_enabled:
        print(f"Remove orphan vectors from FAISS (ids not in final JSON): {len(orphan_faiss_ids)}")

    if args.dry_run:
        print("\n-- DRY RUN: no changes written --")
        print("Sample rm_ids (first 10):", rm_ids[:10])
        if faiss_id_check_enabled:
            print("Sample orphan_faiss_ids (first 10):", orphan_faiss_ids[:10])
        return

    # Backups
    if not args.no_backup:
        jb = backup_file(args.json)
        if jb: print(f"Backed up JSON -> {jb}")

    # Write pruned JSON
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(dedup_list, f, indent=2, ensure_ascii=False)
    print(f"Wrote pruned JSON: {args.json}")

    # Back up index
    if not args.no_backup:
        ib = backup_file(args.index)
        if ib: print(f"Backed up Index -> {ib}")

    # Open FAISS and remove both sets of IDs (union)
    index = faiss.read_index(args.index)
    to_remove = set(rm_ids)
    if faiss_id_check_enabled:
        to_remove |= set(orphan_faiss_ids)

    removed = 0
    if len(to_remove) > 0:
        sel = faiss.IDSelectorBatch(np.array(sorted(list(to_remove)), dtype="int64"))
        removed = index.remove_ids(sel)
    print(f"Removed {removed} vectors from FAISS.")

    faiss.write_index(index, args.index)
    print(f"Wrote updated FAISS index: {args.index}")
    print("Done ✅")

if __name__ == "__main__":
    main()
