import json
from pathlib import Path
from path import OUTPUT_SCHEMA_DIR, OUTPUT_DIR

# =========================================================
# Output directory
# =========================================================

CHUNK_DIR = OUTPUT_DIR / "output_json_chunk"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Helpers
# =========================================================

def safe_filename(clause_id: str) -> str:
    """Make clause ID filesystem-safe"""
    return clause_id.replace("/", "_")

def write_clause_chunk(doc_id: str, clause: dict, out_root: Path):
    """
    Write one clause as a standalone JSON chunk.
    """
    doc_dir = out_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    chunk = {
        "chunk_id": clause["id"],
        "document_id": doc_id,
        "title": clause.get("title"),
        "parent_id": clause.get("id").rsplit(".", 1)[0] if "." in clause.get("id") else None,
        "content": clause.get("content", []),
        "tables": clause.get("tables", []),
        "figures": clause.get("figures", []),
        "requirements": clause.get("requirements", []),
        "references": clause.get("references", {}),
        "children_ids": [c["id"] for c in clause.get("children", [])]
    }

    out_path = doc_dir / f"{safe_filename(clause['id'])}.json"
    out_path.write_text(
        json.dumps(chunk, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def walk_clauses(doc_id: str, clauses: list, out_root: Path):
    """
    Depth-first traversal of clause tree.
    """
    for clause in clauses:
        write_clause_chunk(doc_id, clause, out_root)
        walk_clauses(doc_id, clause.get("children", []), out_root)

# =========================================================
# Main
# =========================================================

def main():
    schema_files = list(OUTPUT_SCHEMA_DIR.glob("*_final_schema.json"))

    if not schema_files:
        print(f"No schema files found in {OUTPUT_SCHEMA_DIR}")
        return

    for schema_file in schema_files:
        print(f"Chunking {schema_file.name}")

        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        doc_id = schema.get("document_id")

        if not doc_id:
            print(f"  Skipping {schema_file.name}: missing document_id")
            continue

        walk_clauses(doc_id, schema.get("clauses", []), CHUNK_DIR)

        print(f"  [OK] Chunks written for document: {doc_id}\n")

if __name__ == "__main__":
    main()