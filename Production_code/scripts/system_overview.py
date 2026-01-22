#!/usr/bin/env python3
"""
Production Pipeline System Overview

This script provides a comprehensive view of your production pipeline
architecture, showing how your existing scripts map to production services.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.tree import Tree
from rich import box
from rich.text import Text

console = Console()


def show_architecture():
    """Display the production architecture diagram"""
    
    console.print("\n")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]        EMC DOCUMENT PROCESSING - PRODUCTION ARCHITECTURE        [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    arch = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                  📄 DOCUMENT INGESTION SERVICE                   │
    │     FastAPI • SHA-256 Dedup • PostgreSQL • S3 Storage          │
    │                    http://localhost:8000                        │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
                    ╔═══════════▼═══════════╗
                    ║   📨 MESSAGE BROKER    ║
                    ║  RabbitMQ/Kafka       ║
                    ║ localhost:5672/15672  ║
                    ╚═══╦═══════════╦═══════╝
                        │           │
        ┌───────────────┴─┐   ┌─────┴──────────────┐   ┌──────────────┐
        │  🔧 MARKER      │   │  📋 SCHEMA          │   │  📦 CHUNKING │
        │    WORKER       │   │    WORKER           │   │    WORKER    │
        │  GPU-Accelerated│   │  json_to_schema     │   │  Neo4j + Vec │
        │  PDF → JSON     │   │  Hierarchy Build    │   │  Graph Build │
        └────────┬────────┘   └──────┬──────────────┘   └──────┬───────┘
                 │                   │                          │
                 │                   │                          │
    ┌────────────▼───────────────────▼──────────────────────────▼─────────┐
    │                    💾 STORAGE & DATABASES                            │
    │  S3 (MinIO) • PostgreSQL • Neo4j • Redis • Vector DB                │
    │  9000/9001  •   5432      •  7474 • 6379 •  Pinecone                │
    └──────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │                   📊 OBSERVABILITY STACK                              │
    │  Jaeger (16686) • Prometheus (9090) • Grafana (3000)                │
    └──────────────────────────────────────────────────────────────────────┘
    """
    
    console.print(arch, style="cyan")


def show_script_mapping():
    """Show how existing scripts map to production services"""
    
    console.print("\n[bold yellow]📝 Script Migration Mapping[/bold yellow]\n")
    
    table = Table(title="Your Scripts → Production Services", box=box.ROUNDED)
    table.add_column("Your Script", style="cyan", no_wrap=True)
    table.add_column("Production Service", style="green")
    table.add_column("Enhancements", style="yellow")
    table.add_column("Status", style="magenta")
    
    table.add_row(
        "collect_json.py",
        "Storage Service + Events",
        "• S3 uploads\n• Event-driven\n• Deduplication",
        "✓ Ready"
    )
    
    table.add_row(
        "json_to_schema_v4.py",
        "Schema Worker",
        "• Pydantic validation\n• OpenTelemetry traces\n• Error recovery",
        "✓ Ready"
    )
    
    table.add_row(
        "schema_to_chunks.py",
        "Chunking Worker",
        "• Deterministic IDs\n• Neo4j ingestion\n• Vector embeddings",
        "✓ Ready"
    )
    
    console.print(table)


def show_event_flow():
    """Show the event-driven flow"""
    
    console.print("\n[bold green]📨 Event-Driven Processing Flow[/bold green]\n")
    
    tree = Tree("🎯 [bold]Document Upload[/bold]")
    
    branch1 = tree.add("📄 [cyan]DOCUMENT_UPLOADED[/cyan] event published")
    branch1.add("→ Marker Worker consumes event")
    branch1.add("→ Runs Marker extraction (GPU)")
    branch1.add("→ Uploads JSON to S3")
    
    branch2 = tree.add("📋 [cyan]EXTRACTION_COMPLETED[/cyan] event published")
    branch2.add("→ Schema Worker consumes event")
    branch2.add("→ Builds hierarchical structure")
    branch2.add("→ Validates with Pydantic")
    
    branch3 = tree.add("🏗️ [cyan]SCHEMA_READY[/cyan] event published")
    branch3.add("→ Chunking Worker consumes event")
    branch3.add("→ Generates chunks with deterministic IDs")
    branch3.add("→ Creates Neo4j nodes & relationships")
    branch3.add("→ Generates vector embeddings")
    
    branch4 = tree.add("✅ [green]CHUNKING_COMPLETED[/green] event published")
    branch4.add("→ Document status: COMPLETED")
    branch4.add("→ Ready for RAG queries")
    
    console.print(tree)


def show_chunk_schema():
    """Show the production chunk schema"""
    
    console.print("\n[bold blue]📦 Production Chunk Schema[/bold blue]\n")
    
    schema = """
{
  "chunk_id": "ISO_9001_2015:4.4.1",  ← Deterministic ID
  "document_metadata": {
    "id": "ISO_9001_2015",
    "hash": "sha256_abc123...",
    "processed_at": "2024-12-23T10:30:00Z"
  },
  "hierarchy": {
    "parent_id": "ISO_9001_2015:4.4",  ← Parent chunk
    "children_ids": ["ISO_9001_2015:4.4.1.a"],  ← Child chunks
    "level": 3  ← Nesting depth
  },
  "content": [
    {"type": "paragraph", "text": "The organization shall..."}
  ],
  "tables": [...],
  "figures": [...],
  "enrichment": {
    "requirements": [
      {"type": "mandatory", "keyword": "shall", "text": "..."}
    ],
    "external_refs": ["ISO 9000", "ISO 14001"]
  },
  "created_at": "2024-12-23T10:30:00Z",
  "version": 1  ← Version for updates
}
    """
    
    console.print(Panel(schema, title="Production Chunk JSON", border_style="blue"))


def show_advantages():
    """Show advantages of production system"""
    
    console.print("\n[bold magenta]🚀 Production System Advantages[/bold magenta]\n")
    
    table = Table(box=box.SIMPLE)
    table.add_column("Feature", style="cyan")
    table.add_column("Script Approach", style="yellow")
    table.add_column("Production Approach", style="green")
    
    table.add_row(
        "Deduplication",
        "Manual checks",
        "Automatic SHA-256 hash checking"
    )
    
    table.add_row(
        "Error Handling",
        "Script crashes",
        "Retry logic + Dead Letter Queue"
    )
    
    table.add_row(
        "Observability",
        "Print statements",
        "OpenTelemetry traces + Jaeger UI"
    )
    
    table.add_row(
        "Scaling",
        "Run multiple times",
        "Auto-scaling workers + GPU pool"
    )
    
    table.add_row(
        "Updates",
        "Reprocess everything",
        "Deterministic IDs = idempotent updates"
    )
    
    table.add_row(
        "Monitoring",
        "Manual log checking",
        "Grafana dashboards + Prometheus alerts"
    )
    
    console.print(table)


def show_quick_start():
    """Show quick start commands"""
    
    console.print("\n[bold cyan]⚡ Quick Start Commands[/bold cyan]\n")
    
    commands = """
# 1. Start all services
docker-compose -f docker/docker-compose.yml up -d

# 2. Check status
docker-compose ps

# 3. Upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \\
  -F "file=@/path/to/standard.pdf"

# 4. Monitor processing
curl http://localhost:8000/api/v1/documents/{document_id}/status

# 5. View traces in Jaeger
open http://localhost:16686

# 6. View dashboards in Grafana
open http://localhost:3000  # admin/admin

# 7. Check RabbitMQ queue
open http://localhost:15672  # emc/changeme
    """
    
    console.print(Panel(commands, title="Commands", border_style="cyan"))


def show_migration_path():
    """Show the migration path"""
    
    console.print("\n[bold yellow]🛤️  Migration Path[/bold yellow]\n")
    
    tree = Tree("📍 [bold]Migration Journey[/bold]")
    
    phase1 = tree.add("Phase 1: [cyan]Keep Scripts[/cyan] (Current)")
    phase1.add("✓ Your scripts work as-is")
    phase1.add("✓ Add observability (optional)")
    phase1.add("✓ No changes required")
    
    phase2 = tree.add("Phase 2: [yellow]Hybrid Approach[/yellow] (Recommended)")
    phase2.add("• Use production API for uploads")
    phase2.add("• Marker worker processes automatically")
    phase2.add("• Keep your schema script if needed")
    phase2.add("• Production chunking for graph")
    
    phase3 = tree.add("Phase 3: [green]Full Production[/green] (Goal)")
    phase3.add("• All stages automated")
    phase3.add("• Event-driven processing")
    phase3.add("• Auto-scaling workers")
    phase3.add("• Complete observability")
    
    console.print(tree)


def main():
    """Main entry point"""
    
    console.clear()
    
    # Show each section
    show_architecture()
    show_script_mapping()
    show_event_flow()
    show_chunk_schema()
    show_advantages()
    show_migration_path()
    show_quick_start()
    
    # Final message
    console.print("\n")
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]  Production pipeline ready! Start with: docker-compose up -d  [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]\n")


if __name__ == "__main__":
    main()
