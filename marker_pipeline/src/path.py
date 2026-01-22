"""
path definitions for marker_pdf pipeline.
all scripts should import paths from this file.
"""
from pathlib import Path

# Project root (marker_pdf/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All Data directories
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PDFS_DIR = DATA_DIR / "input_pdfs"
OUTPUT_DIR = DATA_DIR / "output"
MARKER_JSON_DIR = OUTPUT_DIR / "marker_json"
MARKER_MD_DIR = OUTPUT_DIR / "marker_md"
OUTPUT_JSON_DIR = OUTPUT_DIR / "output_json"
OUTPUT_SCHEMA_DIR = OUTPUT_DIR / "output_schema"