import shutil
from path import MARKER_JSON_DIR, OUTPUT_JSON_DIR

"""
Copies per-document Marker JSON files from:
data/output/marker_json/<doc>/<doc>.json
into:
data/output/output_json/<doc>.json
"""

for doc_dir in MARKER_JSON_DIR.iterdir():
    if not doc_dir.is_dir():
        continue

    json_file = doc_dir / f"{doc_dir.name}.json"
    if json_file.exists():
        dest = OUTPUT_JSON_DIR / json_file.name
        shutil.copy2(json_file, dest)
        print(f"Collected: {json_file.name}")
