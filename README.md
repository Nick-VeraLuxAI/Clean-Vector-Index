Clean Vector Index — README

A small utility to clean and synchronize a FAISS vector index with its paired JSON memory store. It de-duplicates entries, optionally limits how many memories you keep per subject, and guarantees the JSON and FAISS index contain the same set of IDs (no orphans either way).
What it does

    Deduplicate memories by normalized original text (case/whitespace-insensitive), keeping the “best” one by:

        decided=True > 2) newer timestamp > 3) higher confidence > 4) longer original.

    Subject cap (optional): keep at most N items per subject (e.g., avoid 50 near-duplicates about the same topic).

    Type-safe filtering: robust parsing of confidence, timestamp, and 64-bit vector_id (no float precision loss).

    Two-way sync:

        Drops JSON rows whose vector_id does not exist in FAISS.

        Removes FAISS vectors that aren’t present in the final JSON.

    Safety: writes timestamped backups of both files unless --no-backup is specified.

    Dry run mode: see exactly what would change before applying.

Requirements

    Python 3.9+ recommended

    faiss (faiss-cpu or faiss-gpu)

    numpy

Install:

pip install faiss-cpu numpy
# or: pip install faiss-gpu numpy

Usage

Preview changes (recommended):

python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json  "/path/to/longterm.json" \
  --subject-cap 3 \
  --dry-run

Apply changes:

python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json  "/path/to/longterm.json" \
  --subject-cap 3

Disable per-subject pruning (pure de-dupe + two-way sync):

python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json  "/path/to/longterm.json" \
  --subject-cap 0

Remove boilerplate test phrases by exact match (normalized):

python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json  "/path/to/longterm.json" \
  --subject-cap 3 \
  --drop-exact "how do i feel about my wife?" \
  --drop-exact "have i shared any feelings with you that i have about my wife?"

Be stricter on signal:

python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json  "/path/to/longterm.json" \
  --subject-cap 3 \
  --min-confidence 0.5

Skip backups (not recommended):

python clean_vector_index.py \
  --index "/path/to/vector.index" \
  --json  "/path/to/longterm.json" \
  --subject-cap 3 \
  --no-backup

Options

    --index (required): Path to FAISS index file (e.g., vector.index).

    --json (required): Path to long-term JSON memory array.

    --subject-cap (int, default: 3): Max items per subject. Use 0 to disable.

    --min-confidence (float, default: 0.0): Drop items below this confidence before dedupe.

    --drop-exact (repeatable): Exact phrases to drop (case/whitespace-insensitive).

    --no-backup: Don’t create .bak-YYYYMMDD-HHMMSS backups.

    --dry-run: Show what would change but don’t modify files.

Typical workflow

    Dry run to preview:

        Confirm counts for “After dedupe”, “After subject cap”, “Remove orphan vectors”.

    Run for real (remove --dry-run).

    Restart whatever service loads vector.index so it picks up changes.

    (Optional) Commit the updated JSON and index to version control/backups.

Example dry-run output (interpretation)

=== Cleaner Summary ===
JSON entries before: 216
Below min-confidence dropped: 0
Invalid/missing vector_id dropped: 0
JSON entries dropped (no matching vector in FAISS): 0
After min-confidence/drop-exact/valid-id/exists: 60
After dedupe: 60
After subject cap (3/subject): 60
FAISS vectors present: 215
Final Keep IDs: 60
Remove IDs from FAISS (not kept but were valid in JSON): 0
Remove orphan vectors from FAISS (ids not in final JSON): 155
-- DRY RUN: no changes written --

    You’ll keep 60 JSON memories and remove 155 orphan vectors from FAISS when you run for real.

Data model expectations

Each JSON memory is an object like:

{
  "action": "ask-question",
  "type": "emotion",
  "subject": "wife",
  "sentiment": "neutral",
  "confidence": 0.6,
  "original": "Have I shared any feelings with you that I have about my wife?",
  "summary": "…",
  "vector_id": 2566561915425579214,
  "decided": true,
  "timestamp": 1723333333
}

Required fields for keeping an entry:

    vector_id: valid non-zero 64-bit integer that exists in FAISS (unless the index cannot expose IDs).

    Optional but used for prioritization: confidence, timestamp, decided.

Troubleshooting

    “JSON entries dropped (no matching vector in FAISS)” is unexpectedly high
    Your vector.index may be out of sync; you might have re-built JSON or moved indexes. If intended, run without --dry-run to remove orphan vectors or re-embed/rebuild the index from JSON if you want the reverse.

    Large IDs “don’t match”
    This script safely parses 64-bit IDs (no float()), so mismatches usually mean real desync, not parse errors.

    Service still shows old results
    Restart the process that memory-maps or caches vector.index.

    Index isn’t an IDMap
    If the FAISS index can’t expose an id_map, the script will skip JSON→FAISS existence checks (it will still remove known IDs that you tell it to via JSON).

Safety & backup notes

    The script creates timestamped .bak-YYYYMMDD-HHMMSS backups for both JSON and index by default.

    To restore, just copy the .bak-* file back over the original path.

Recommendations

    Keep testing phrases out of production memory (--drop-exact is handy).

    Add timestamp when writing memories so dedupe chooses the freshest version.

    Consider MMR re-ranking at retrieval time for diversity, even with a clean index.

