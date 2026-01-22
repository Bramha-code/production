import subprocess
import sys
import logging
from pathlib import Path
from typing import Optional

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# =========================================================
# Resolve project paths (ABSOLUTE, SAFE)
# =========================================================

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

# Ensure src is importable
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =========================================================
# Project paths (from path.py)
# =========================================================

from path import (
    INPUT_PDFS_DIR,
    MARKER_JSON_DIR,
    MARKER_MD_DIR,
    OUTPUT_JSON_DIR,
    OUTPUT_SCHEMA_DIR,
    OUTPUT_DIR,
)

# =========================================================
# Configuration
# =========================================================

MARKER_WORKERS = "1"
MARKER_DISABLE_MP = True  # deterministic runs
MARKER_OUTPUT_FORMAT = 'json'

# =========================================================
# Helpers
# =========================================================

def run_cmd(cmd: list, cwd: Optional[Path] = None, stage_name: str = "") -> None:
    """Run a shell command and fail fast if it errors."""
    cmd = [str(c) for c in cmd]
    prefix = f"[{stage_name}] " if stage_name else ""

    logger.info(f"{prefix}Running: {' '.join(cmd)}")

    env = None
    if sys.platform == "win32":
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(f"{prefix}{line}")

    if result.stderr:
        for line in result.stderr.splitlines():
            logger.warning(f"{prefix}{line}")

    if result.returncode != 0:
        raise RuntimeError(f"{prefix}Command failed with exit code {result.returncode}")

    logger.info(f"{prefix}Completed successfully")


def ensure_dirs() -> None:
    """Create all required directories."""
    dirs = [
        INPUT_PDFS_DIR,
        MARKER_JSON_DIR,
        MARKER_MD_DIR,
        OUTPUT_JSON_DIR,
        OUTPUT_SCHEMA_DIR,
        OUTPUT_DIR / "output_json_chunk",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")


def assert_non_empty(dir_path: Path, pattern: str, stage: str) -> None:
    """Fail fast if expected outputs are missing."""
    files = list(dir_path.glob(pattern))
    if not files:
        raise RuntimeError(
            f"[{stage}] Expected files not found in {dir_path} (pattern={pattern})"
        )
    logger.info(f"[{stage}] Found {len(files)} file(s)")


def validate_input_pdfs() -> int:
    """Ensure PDFs exist."""
    pdfs = list(INPUT_PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {INPUT_PDFS_DIR}")
        return 0

    logger.info(f"Found {len(pdfs)} PDF(s)")
    for pdf in pdfs:
        logger.info(f"  - {pdf.name}")
    return len(pdfs)

# =========================================================
# Script paths
# =========================================================

COLLECT_JSON_SCRIPT = SRC_DIR / "collect_json.py"
JSON_TO_SCHEMA_SCRIPT = SRC_DIR / "json_to_schema_v5.py"
SCHEMA_TO_CHUNKS_SCRIPT = SRC_DIR / "schema_to_chunks.py"

# =========================================================
# Pipeline Stages
# =========================================================

def stage_1_run_marker() -> None:
    """Run Marker with a deterministic output directory."""
    logger.info("=" * 50)
    logger.info("STAGE 1: Marker PDF Processing")
    logger.info("=" * 50)

    cmd = [
        "marker",
        str(INPUT_PDFS_DIR),
        "--output_dir", str(MARKER_JSON_DIR),
        "--workers", MARKER_WORKERS,
        "--ouput_format", MARKER_OUTPUT_FORMAT,
    ]

    if MARKER_DISABLE_MP:
        cmd.append("--disable_multiprocessing")

    run_cmd(cmd, cwd=PROJECT_ROOT, stage_name="Marker")
    assert_non_empty(MARKER_JSON_DIR, "*/*.json", "Marker Output")


def stage_2_collect_marker_json() -> None:
    logger.info("=" * 50)
    logger.info("STAGE 2: Collect Marker JSON")
    logger.info("=" * 50)

    run_cmd([sys.executable, COLLECT_JSON_SCRIPT], cwd=SRC_DIR, stage_name="Collect JSON")
    assert_non_empty(OUTPUT_JSON_DIR, "*.json", "Collected JSON")


def stage_3_json_to_schema() -> None:
    logger.info("=" * 50)
    logger.info("STAGE 3: JSON → Schema")
    logger.info("=" * 50)

    run_cmd([sys.executable, JSON_TO_SCHEMA_SCRIPT], cwd=SRC_DIR, stage_name="JSON to Schema")
    assert_non_empty(OUTPUT_SCHEMA_DIR, "*_final_schema.json", "Schema Output")


def stage_4_schema_to_chunks() -> None:
    logger.info("=" * 50)
    logger.info("STAGE 4: Schema → Chunks")
    logger.info("=" * 50)

    run_cmd([sys.executable, SCHEMA_TO_CHUNKS_SCRIPT], cwd=SRC_DIR, stage_name="Schema to Chunks")
    assert_non_empty(OUTPUT_DIR / "output_json_chunk", "*/*.json", "Chunk Output")

# =========================================================
# Main
# =========================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("PDF PIPELINE START")
    logger.info("=" * 60)

    ensure_dirs()
    pdf_count = validate_input_pdfs()

    if pdf_count == 0:
        logger.warning("Nothing to process. Exiting.")
        return

    stage_1_run_marker()
    stage_2_collect_marker_json()
    stage_3_json_to_schema()
    stage_4_schema_to_chunks()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Schemas → {OUTPUT_SCHEMA_DIR}")
    logger.info(f"Chunks  → {OUTPUT_DIR / 'output_json_chunk'}")


if __name__ == "__main__":
    main()