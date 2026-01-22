import json
import re
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from path import OUTPUT_JSON_DIR, OUTPUT_SCHEMA_DIR, OUTPUT_DIR

# =========================================================
# REGEX PATTERNS - For extracting structured information
# =========================================================

# Extract clause numbers and titles (e.g., "1.2.3 Safety Requirements")
CLAUSE_WITH_TITLE_RE = re.compile(r'^([A-Z]|\d+)(?:\.(\d+))*\s+(.+)$', re.IGNORECASE)
# Extract clause numbers only (e.g., "1.2.3")
CLAUSE_NUM_ONLY_RE = re.compile(r'^([A-Z]|\d+)(?:\.(\d+))*\s*$', re.IGNORECASE)
# Remove HTML tags
HTML_TAG_RE = re.compile(r'<[^>]+>')
# Extract normative keywords (shall, should, may)
REQ_RE = re.compile(r'\b(shall not|shall|should|may)\b', re.IGNORECASE)
# Extract table/figure numbers from captions
TABLE_REF_RE = re.compile(r'\btable\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)
FIGURE_REF_RE = re.compile(r'\b(?:figure|fig\.?)\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def strip_html(html: str) -> str:
    """Remove HTML tags and clean whitespace from text."""
    if not html:
        return ""
    return HTML_TAG_RE.sub('', html).strip()

def detect_image_format(data: bytes) -> str:
    """Detect image format from binary data header and return file extension."""
    if not data:
        return ".bin"
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"

def extract_clause_info(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extract clause ID and title from text.
    Examples:
        "1.2 Safety" -> ("1.2", "Safety")
        "A.3" -> ("A.3", None)
    """
    if not text:
        return None
    
    # Try matching clause with title
    m = CLAUSE_WITH_TITLE_RE.match(text)
    if m:
        clause_id = text.split()[0]
        title = text[len(clause_id):].strip()
        return (clause_id, title if title else None)
    
    # Try matching clause number only
    m = CLAUSE_NUM_ONLY_RE.match(text)
    if m:
        return (m.group(0).strip(), None)
    
    return None

def extract_table_number(caption: str) -> Optional[str]:
    """Extract table number from caption (e.g., 'Table 1.2 — Description' -> '1.2')."""
    if not caption:
        return None
    match = TABLE_REF_RE.search(caption)
    return match.group(1) if match else None

def extract_figure_number(caption: str) -> Optional[str]:
    """Extract figure number from caption (e.g., 'Figure 3.1 — Diagram' -> '3.1')."""
    if not caption:
        return None
    match = FIGURE_REF_RE.search(caption)
    return match.group(1) if match else None

def extract_requirements(text: str) -> List[Dict[str, str]]:
    """
    Extract normative requirements from text based on keywords.
    Returns list with requirement type (mandatory, recommendation, etc.)
    """
    if not text:
        return []
    
    requirements = []
    for match in REQ_RE.finditer(text):
        keyword = match.group(1).lower()
        req_type = {
            "shall not": "prohibition",
            "shall": "mandatory",
            "should": "recommendation",
            "may": "permission"
        }.get(keyword)
        
        if req_type:
            requirements.append({
                "type": req_type,
                "keyword": keyword,
                "text": text
            })
            break  # Only capture first requirement per text block
    
    return requirements

# =========================================================
# PROCESSING CONTEXT - Tracks state during document parsing
# =========================================================

class ProcessingContext:
    """
    Maintains state while processing the document tree.
    Tracks current clause, pending captions, etc.
    """
    def __init__(self):
        self.current_clause_id: Optional[str] = None  # Current clause being processed
        self.pending_number: Optional[str] = None      # Clause number waiting for title
        self.pending_caption: Optional[str] = None     # Caption waiting for table/figure
    
    def reset_pending(self):
        """Clear pending items when starting new clause."""
        self.pending_caption = None

# =========================================================
# MAIN PROCESSING LOGIC
# =========================================================

def process_block(
    block: Dict,
    clauses: Dict[str, Dict],
    context: ProcessingContext,
    counters: Dict,
    img_root: Path,
    misc_img_dir: Path
):
    """
    Recursively process a block from Marker JSON.
    
    Flow:
    1. Identify block type (SectionHeader, Text, Table, Picture, etc.)
    2. Extract content and metadata
    3. Associate with current clause
    4. Recursively process children
    
    Key associations:
    - Captions are captured and associated with next table/figure
    - All content is linked to the current clause context
    - Images and tables are numbered within each clause
    """
    
    if not isinstance(block, dict):
        return
    
    btype = block.get("block_type")
    
    # Skip headers/footers - they're not content
    if btype in ("PageHeader", "PageFooter"):
        return
    
    # ==================== SECTION HEADER - Creates or updates clauses ====================
    if btype == "SectionHeader":
        text = strip_html(block.get("html", ""))
        
        if text:
            info = extract_clause_info(text)
            
            if info:
                clause_id, title = info
                
                # Create new clause if we have both ID and title
                if title:
                    if clause_id not in clauses:
                        clauses[clause_id] = {
                            "id": clause_id,
                            "title": title,
                            "children": [],      # Sub-clauses
                            "content": [],       # Text paragraphs, lists
                            "tables": [],        # Tables in this clause
                            "figures": [],       # Figures in this clause
                            "requirements": []   # Normative requirements
                        }
                    context.current_clause_id = clause_id
                    context.pending_number = None
                    context.reset_pending()
                else:
                    # Only have number, wait for title in next block
                    context.pending_number = clause_id
            
            # This text is the title for pending clause number
            elif context.pending_number:
                clause_id = context.pending_number
                if clause_id not in clauses:
                    clauses[clause_id] = {
                        "id": clause_id,
                        "title": text,
                        "children": [],
                        "content": [],
                        "tables": [],
                        "figures": [],
                        "requirements": []
                    }
                context.current_clause_id = clause_id
                context.pending_number = None
                context.reset_pending()
    
    # ==================== CAPTION - Store for next table/figure ====================
    elif btype == "Caption":
        text = strip_html(block.get("html", ""))
        if text:
            # Check if this is a table/figure caption
            if extract_table_number(text) or extract_figure_number(text):
                context.pending_caption = text
            else:
                # Generic caption, add to clause content
                if context.current_clause_id and context.current_clause_id in clauses:
                    clauses[context.current_clause_id]["content"].append({
                        "type": "caption",
                        "text": text
                    })
    
    # ==================== TABLE - Extract table with metadata ====================
    elif btype == "Table":
        html = block.get("html", "")
        
        if html:
            counters["total_tables"] += 1
            
            # Get caption (from pending or block)
            caption = context.pending_caption or strip_html(block.get("caption", ""))
            table_number = extract_table_number(caption) if caption else None
            
            # Create table entry with all data
            table_entry = {
                "html": html,
                "number": table_number
            }
            
            if caption:
                table_entry["caption"] = caption
            
            # Include row data if available (structured data)
            rows = block.get("rows")
            if rows:
                table_entry["rows"] = rows
            
            # Add to current clause
            if context.current_clause_id and context.current_clause_id in clauses:
                clauses[context.current_clause_id]["tables"].append(table_entry)
            
            context.pending_caption = None
    
    # ==================== PICTURE - Extract and save images ====================
    elif btype == "Picture":
        images = block.get("images", {})
        
        if images:
            for img_key, b64_data in images.items():
                try:
                    # Skip empty images
                    if not b64_data:
                        print(f"    Warning: Empty image data for {img_key}")
                        continue
                    
                    # Decode base64 image
                    data = base64.b64decode(b64_data)
                    ext = detect_image_format(data)
                    counters["total_images"] += 1
                    
                    # Get caption and figure number
                    caption = context.pending_caption or strip_html(block.get("caption", ""))
                    figure_number = extract_figure_number(caption) if caption else None
                    
                    # Create clean filename from image key (/page/100/Picture/0 -> page_100_picture_0)
                    img_ref = img_key.replace("/", "_").strip("_").lower()
                    
                    # Save to clause folder if we have a current clause
                    if context.current_clause_id and context.current_clause_id in clauses:
                        cid = context.current_clause_id
                        
                        # Initialize counter for this clause
                        if cid not in counters["figure_counters"]:
                            counters["figure_counters"][cid] = 0
                        counters["figure_counters"][cid] += 1
                        
                        # Create clause directory
                        clause_dir = img_root / cid.replace(".", "_")
                        clause_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save image with descriptive filename
                        fname = f"figure_{counters['figure_counters'][cid]}_{img_ref}{ext}"
                        fpath = clause_dir / fname
                        fpath.write_bytes(data)
                        
                        # Add figure metadata to clause
                        figure_entry = {
                            "number": figure_number or counters["figure_counters"][cid],
                            "path": str(fpath.relative_to(OUTPUT_DIR)),
                            "format": ext.lstrip("."),
                            "original_key": img_key,
                            "size_bytes": len(data)
                        }
                        
                        if caption:
                            figure_entry["caption"] = caption
                        
                        clauses[context.current_clause_id]["figures"].append(figure_entry)
                        counters["clause_images"] += 1
                    
                    else:
                        # No clause context - save to misc folder
                        counters["misc_image_counter"] += 1
                        fname = f"misc_{counters['misc_image_counter']}_{img_ref}{ext}"
                        fpath = misc_img_dir / fname
                        fpath.write_bytes(data)
                        counters["misc_images"] += 1
                        
                        # Track misc image metadata
                        if "misc_image_metadata" not in counters:
                            counters["misc_image_metadata"] = []
                        counters["misc_image_metadata"].append({
                            "path": str(fpath.relative_to(OUTPUT_DIR)),
                            "caption": caption if caption else None,
                            "format": ext.lstrip("."),
                            "original_key": img_key,
                            "size_bytes": len(data)
                        })
                
                except Exception as e:
                    print(f"    Warning: Failed to process image {img_key}: {e}")
            
            context.pending_caption = None
    
    # ==================== TEXT - Regular paragraph content ====================
    elif btype == "Text":
        text = strip_html(block.get("html", ""))
        
        if text and context.current_clause_id and context.current_clause_id in clauses:
            clause = clauses[context.current_clause_id]
            
            # Add paragraph to content
            clause["content"].append({
                "type": "paragraph",
                "text": text
            })
            
            # Extract any requirements from this text
            reqs = extract_requirements(text)
            clause["requirements"].extend(reqs)
    
    # ==================== FOOTNOTE - Special notes ====================
    elif btype == "Footnote":
        text = strip_html(block.get("html", ""))
        
        if text and context.current_clause_id and context.current_clause_id in clauses:
            clauses[context.current_clause_id]["content"].append({
                "type": "footnote",
                "text": text
            })
    
    # ==================== LIST ITEM - Bullet/numbered lists ====================
    elif btype == "ListItem":
        text = strip_html(block.get("html", ""))
        
        if text and context.current_clause_id and context.current_clause_id in clauses:
            clause = clauses[context.current_clause_id]
            
            # Add list item to content
            clause["content"].append({
                "type": "list_item",
                "text": text
            })
            
            # Extract requirements
            reqs = extract_requirements(text)
            clause["requirements"].extend(reqs)
    
    # ==================== RECURSION - Process all children ====================
    children = block.get("children")
    if children and isinstance(children, list):
        for child in children:
            process_block(child, clauses, context, counters, img_root, misc_img_dir)

# =========================================================
# FILE CONVERSION - Main entry point
# =========================================================

def convert_file(path: Path) -> Dict:
    """
    Convert a Marker JSON file to structured schema.
    
    Process:
    1. Load JSON file
    2. Create output directories
    3. Process all blocks recursively
    4. Build clause hierarchy (1.1 is child of 1, A.1 is child of A)
    5. Generate statistics
    6. Return complete schema
    """
    
    # Load source JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    
    # Setup output directories
    doc_id = path.stem
    img_root = OUTPUT_DIR / "output_images" / doc_id
    img_root.mkdir(parents=True, exist_ok=True)
    
    misc_img_dir = img_root / "misc"
    misc_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data structures
    clauses: Dict[str, Dict] = OrderedDict()  # All clauses, keyed by ID
    context = ProcessingContext()             # Processing state
    
    # Counters for statistics
    counters = {
        "total_images": 0,
        "clause_images": 0,
        "misc_images": 0,
        "misc_image_counter": 0,
        "total_tables": 0,
        "figure_counters": {},
        "misc_image_metadata": []
    }
    
    # Process all blocks in document
    children = raw.get("children", [])
    if children:
        for child in children:
            process_block(child, clauses, context, counters, img_root, misc_img_dir)
    
    # Build clause hierarchy (1.1 becomes child of 1)
    for cid, clause in clauses.items():
        if cid[0].isdigit():
            # Numeric clause: 1.1.2 -> parent is 1.1
            parts = cid.split(".")
            if len(parts) > 1:
                parent_id = ".".join(parts[:-1])
                if parent_id in clauses:
                    parent = clauses[parent_id]
                    if clause not in parent["children"]:
                        parent["children"].append(clause)
        
        elif "." in cid:
            # Annex sub-clause: A.1 -> parent is A
            parts = cid.split(".")
            parent_id = parts[0]
            if parent_id in clauses:
                parent = clauses[parent_id]
                if clause not in parent["children"]:
                    parent["children"].append(clause)
    
    # Find root clauses (top-level, not children of others)
    child_ids = {c["id"] for cl in clauses.values() for c in cl["children"]}
    roots = [c for cid, c in clauses.items() if cid not in child_ids]
    
    # Sort: numeric clauses first (1, 2, 3), then annexes (A, B, C)
    def sort_key(clause):
        cid = clause["id"]
        if cid[0].isdigit():
            try:
                return (0, [int(n) for n in cid.split(".")])
            except:
                return (0, [0])
        else:
            return (1, cid)
    
    roots.sort(key=sort_key)
    
    # Build final result
    result = {
        "document_id": doc_id,
        "statistics": {
            "total_images": counters["total_images"],
            "images_in_clauses": counters["clause_images"],
            "images_in_misc": counters["misc_images"],
            "total_tables": counters["total_tables"],
            "total_clauses": len(clauses)
        },
        "clauses": roots
    }
    
    # Add misc images info if any
    if counters["misc_images"] > 0:
        result["misc_images"] = {
            "count": counters["misc_images"],
            "path": str(misc_img_dir.relative_to(OUTPUT_DIR)),
            "images": counters.get("misc_image_metadata", [])
        }
    
    return result

# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    """
    Process all JSON files in input directory.
    For each file:
    1. Convert to schema
    2. Save output JSON
    3. Print statistics
    """
    
    OUTPUT_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    
    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {OUTPUT_JSON_DIR}")
        return
    
    print(f"Found {len(json_files)} file(s) to process\n")
    
    for file in json_files:
        print(f"Processing: {file.name}")
        
        try:
            # Convert file
            schema = convert_file(file)
            
            # Save output
            out_path = OUTPUT_SCHEMA_DIR / f"{file.stem}_final_schema.json"
            out_path.write_text(
                json.dumps(schema, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            
            # Print statistics
            print(f"  [OK] Extracted {schema['statistics']['total_images']} images")
            print(f"    - {schema['statistics']['images_in_clauses']} in clauses")
            print(f"    - {schema['statistics']['images_in_misc']} in misc folder")
            print(f"  [OK] Extracted {schema['statistics']['total_tables']} tables")
            print(f"  [OK] Processed {schema['statistics']['total_clauses']} clauses")
            print(f"  [OK] Saved to: {out_path.name}\n")
            
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            print()

if __name__ == "__main__":
    main()