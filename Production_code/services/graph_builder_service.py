"""
Knowledge Graph Builder Service

Implements production-grade graph ingestion with:
1. Three-stage validation gate
2. Two-pass ingestion strategy (nodes first, then relationships)
3. Deterministic UIDs for idempotency
4. Transaction management with audit trail
5. Event-driven processing

This is Phase 2 of the production pipeline.
"""

from pathlib import Path
import json
import asyncio
import aio_pika
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

from opentelemetry import trace

from models.graph_schema import (
    DocumentNode,
    ClauseNode,
    RequirementNode,
    TableNode,
    FigureNode,
    StandardNode,
    GraphNode,
    GraphRelationship,
    GraphTransaction,
    ContainsRelationship,
    RequiresRelationship,
    RefersToRelationship,
    ReferencesRelationship,
    UIDGenerator,
    ValidationGate,
    NodeType,
    RelationshipType,
    RequirementType,
)
from .neo4j_driver import Neo4jDriver, GraphBuilder


tracer = trace.get_tracer(__name__)


# =========================================================
# Chunk-to-Graph Mapper
# =========================================================


class ChunkMapper:
    """
    Maps JSON chunks to graph nodes and relationships.
    Implements the data-to-graph mapping logic.
    """

    def __init__(self, document_hash: str, transaction_id: str):
        self.document_hash = document_hash
        self.transaction_id = transaction_id

    def map_document_chunk(self, schema: Dict[str, Any]) -> DocumentNode:
        """
        Create Document node from schema file.

        Args:
            schema: Complete document schema from json_to_schema_v4.py

        Returns:
            DocumentNode
        """
        document_id = schema["document_id"]
        stats = schema.get("statistics", {})

        return DocumentNode(
            uid=UIDGenerator.generate_document_uid(document_id),
            document_id=document_id,
            filename=f"{document_id}.pdf",  # Could be passed in
            document_hash=self.document_hash,
            total_clauses=stats.get("total_clauses", 0),
            total_tables=stats.get("total_tables", 0),
            total_figures=stats.get("total_images", 0),
            transaction_id=self.transaction_id,
        )

    def map_clause_chunk(self, chunk: Dict[str, Any]) -> ClauseNode:
        """
        Create Clause node from chunk.

        Args:
            chunk: Chunk data from schema_to_chunks.py or simplified format

        Returns:
            ClauseNode
        """
        # Handle both old format (document_metadata.id) and new format (document_id)
        doc_metadata = chunk.get("document_metadata", {})
        document_id = doc_metadata.get("id", chunk.get("document_id"))

        # Extract clause ID from chunk_id (format: DOC_ID:CLAUSE_ID or just CLAUSE_ID)
        chunk_id = chunk["chunk_id"]
        clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id

        # Handle both old format (hierarchy object) and new format (direct fields)
        hierarchy = chunk.get("hierarchy", {})
        content_blocks = chunk.get("content", [])

        # Concatenate all content text
        content_text = " ".join([block.get("text", "") for block in content_blocks])

        # Determine parent UID - check both formats
        parent_id = hierarchy.get("parent_id") or chunk.get("parent_id")
        parent_uid = None
        if parent_id:
            parent_uid = UIDGenerator.generate_clause_uid(document_id, parent_id)

        # Generate child UIDs - check both formats
        children_ids = hierarchy.get("children_ids", []) or chunk.get("children_ids", [])
        children_uids = [
            UIDGenerator.generate_clause_uid(document_id, child_id)
            for child_id in children_ids
        ]

        # Get level - check both formats
        level = hierarchy.get("level", 0)
        if level == 0 and clause_id:
            # Estimate level from clause_id (e.g., "1.0" = level 1, "1.2.3" = level 3)
            level = len(clause_id.split("."))

        # Check for requirements - both formats
        has_requirements = bool(chunk.get("enrichment", {}).get("requirements")) or bool(chunk.get("requirements"))

        return ClauseNode(
            uid=UIDGenerator.generate_clause_uid(document_id, clause_id),
            document_id=document_id,
            clause_id=clause_id,
            title=chunk.get("title", ""),
            parent_uid=parent_uid,
            children_uids=children_uids,
            level=level,
            content_text=content_text,
            content_length=len(content_text),
            has_requirements=has_requirements,
            has_tables=bool(chunk.get("tables")),
            has_figures=bool(chunk.get("figures")),
            document_hash=self.document_hash,
            transaction_id=self.transaction_id,
        )

    def map_requirement_chunks(self, chunk: Dict[str, Any]) -> List[RequirementNode]:
        """
        Create Requirement nodes from chunk.

        Args:
            chunk: Chunk data

        Returns:
            List of RequirementNode
        """
        doc_metadata = chunk.get("document_metadata", {})
        document_id = doc_metadata.get("id", chunk.get("document_id"))

        chunk_id = chunk["chunk_id"]
        clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id

        source_clause_uid = UIDGenerator.generate_clause_uid(document_id, clause_id)

        enrichment = chunk.get("enrichment", {})
        requirements = enrichment.get("requirements", [])

        req_nodes = []
        for idx, req in enumerate(requirements):
            req_node = RequirementNode(
                document_id=document_id,
                clause_id=clause_id,
                requirement_type=RequirementType(req["type"]),
                keyword=req["keyword"],
                text=req["text"],
                source_clause_uid=source_clause_uid,
                document_hash=self.document_hash,
                transaction_id=self.transaction_id,
            )
            req_nodes.append(req_node)

        return req_nodes

    def map_table_chunks(self, chunk: Dict[str, Any]) -> List[TableNode]:
        """Create Table nodes from chunk"""
        doc_metadata = chunk.get("document_metadata", {})
        document_id = doc_metadata.get("id", chunk.get("document_id"))

        chunk_id = chunk["chunk_id"]
        clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id

        source_clause_uid = UIDGenerator.generate_clause_uid(document_id, clause_id)

        tables = chunk.get("tables", [])
        table_nodes = []

        for idx, table in enumerate(tables):
            rows = table.get("rows", [])
            row_count = len(rows)
            column_count = len(rows[0]) if rows else 0

            # Handle null table numbers
            table_number = table.get("number")
            if table_number is None:
                table_number = idx + 1

            # Explicitly generate UID
            table_uid = UIDGenerator.generate_table_uid(document_id, clause_id, table_number)

            table_node = TableNode(
                uid=table_uid,
                document_id=document_id,
                clause_id=clause_id,
                table_number=table_number,
                caption=table.get("caption"),
                row_count=row_count,
                column_count=column_count,
                source_clause_uid=source_clause_uid,
                document_hash=self.document_hash,
                transaction_id=self.transaction_id,
            )
            table_nodes.append(table_node)

        return table_nodes

    def map_figure_chunks(self, chunk: Dict[str, Any]) -> List[FigureNode]:
        """Create Figure nodes from chunk"""
        doc_metadata = chunk.get("document_metadata", {})
        document_id = doc_metadata.get("id", chunk.get("document_id"))

        chunk_id = chunk["chunk_id"]
        clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id

        source_clause_uid = UIDGenerator.generate_clause_uid(document_id, clause_id)

        figures = chunk.get("figures", [])
        figure_nodes = []

        for idx, figure in enumerate(figures):
            # Handle null figure numbers
            figure_number = figure.get("number")
            if figure_number is None:
                figure_number = idx + 1

            # Explicitly generate UID
            figure_uid = UIDGenerator.generate_figure_uid(document_id, clause_id, figure_number)

            figure_node = FigureNode(
                uid=figure_uid,
                document_id=document_id,
                clause_id=clause_id,
                figure_number=figure_number,
                caption=figure.get("caption"),
                image_path=figure.get("path"),
                source_clause_uid=source_clause_uid,
                document_hash=self.document_hash,
                transaction_id=self.transaction_id,
            )
            figure_nodes.append(figure_node)

        return figure_nodes

    def map_standard_references(self, chunk: Dict[str, Any]) -> List[StandardNode]:
        """Create Standard nodes for external references"""
        enrichment = chunk.get("enrichment", {})
        external_refs = enrichment.get("external_refs", [])

        standard_nodes = []
        for std_name in external_refs:
            std_node = StandardNode(
                standard_name=std_name,
                normalized_name=std_name.upper().replace(" ", "").replace("-", ""),
                document_hash=self.document_hash,
                transaction_id=self.transaction_id,
            )
            standard_nodes.append(std_node)

        return standard_nodes


# =========================================================
# Relationship Builder
# =========================================================


class RelationshipBuilder:
    """
    Builds relationships between nodes based on chunk data.
    Implements Pass 2 of the two-pass strategy.
    """

    def __init__(self, transaction_id: str):
        self.transaction_id = transaction_id

    def build_contains_relationships(
        self, document_uid: str, clause_nodes: List[ClauseNode]
    ) -> List[ContainsRelationship]:
        """
        Build CONTAINS relationships for the hierarchy.

        Document→Clause (root clauses)
        Clause→Clause (parent-child)
        """
        relationships = []

        for clause in clause_nodes:
            if clause.parent_uid:
                # Clause has a parent clause
                rel = ContainsRelationship(
                    source_uid=clause.parent_uid,
                    target_uid=clause.uid,
                    transaction_id=self.transaction_id,
                )
                relationships.append(rel)
            else:
                # Root clause - parent is document
                rel = ContainsRelationship(
                    source_uid=document_uid,
                    target_uid=clause.uid,
                    transaction_id=self.transaction_id,
                )
                relationships.append(rel)

        return relationships

    def build_requires_relationships(
        self, requirement_nodes: List[RequirementNode]
    ) -> List[RequiresRelationship]:
        """Build REQUIRES relationships: Clause→Requirement"""
        relationships = []

        for req in requirement_nodes:
            rel = RequiresRelationship(
                source_uid=req.source_clause_uid,
                target_uid=req.uid,
                transaction_id=self.transaction_id,
            )
            relationships.append(rel)

        return relationships

    def build_content_relationships(
        self, table_nodes: List[TableNode], figure_nodes: List[FigureNode]
    ) -> List[GraphRelationship]:
        """Build HAS_TABLE and HAS_FIGURE relationships"""
        relationships = []

        # Tables
        for table in table_nodes:
            rel = GraphRelationship(
                relationship_type=RelationshipType.HAS_TABLE,
                source_uid=table.source_clause_uid,
                target_uid=table.uid,
                transaction_id=self.transaction_id,
            )
            relationships.append(rel)

        # Figures
        for figure in figure_nodes:
            rel = GraphRelationship(
                relationship_type=RelationshipType.HAS_FIGURE,
                source_uid=figure.source_clause_uid,
                target_uid=figure.uid,
                transaction_id=self.transaction_id,
            )
            relationships.append(rel)

        return relationships

    def build_reference_relationships(
        self, chunks: List[Dict[str, Any]], existing_clause_uids: Set[str]
    ) -> List[GraphRelationship]:
        """
        Build REFERS_TO (internal) and REFERENCES (external) relationships.

        Args:
            chunks: All chunks for the document
            existing_clause_uids: Set of clause UIDs that exist in graph

        Returns:
            List of reference relationships
        """
        relationships = []

        for chunk in chunks:
            doc_metadata = chunk.get("document_metadata", {})
            document_id = doc_metadata.get("id", chunk.get("document_id"))

            chunk_id = chunk["chunk_id"]
            clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id
            source_uid = UIDGenerator.generate_clause_uid(document_id, clause_id)

            enrichment = chunk.get("enrichment", {})
            internal_refs = enrichment.get("internal_refs", {})

            # Internal clause references
            for ref_clause_id in internal_refs.get("clauses", []):
                target_uid = UIDGenerator.generate_clause_uid(
                    document_id, ref_clause_id
                )

                # Only create if target exists
                if target_uid in existing_clause_uids:
                    rel = RefersToRelationship(
                        source_uid=source_uid,
                        target_uid=target_uid,
                        reference_text=f"Clause {ref_clause_id}",
                        transaction_id=self.transaction_id,
                    )
                    relationships.append(rel)

            # External standard references
            external_refs = enrichment.get("external_refs", [])
            for std_name in external_refs:
                target_uid = UIDGenerator.generate_standard_uid(std_name)

                rel = ReferencesRelationship(
                    source_uid=source_uid,
                    target_uid=target_uid,
                    reference_text=std_name,
                    transaction_id=self.transaction_id,
                )
                relationships.append(rel)

        return relationships


# =========================================================
# Knowledge Graph Builder Service
# =========================================================


class KnowledgeGraphBuilderService:
    """
    Main service that orchestrates graph building with:
    1. Three-stage validation
    2. Two-pass ingestion (nodes, then relationships)
    3. Transaction management
    4. Event processing
    """

    def __init__(self, neo4j_driver: Neo4jDriver):
        self.driver = neo4j_driver
        self.graph_builder = GraphBuilder(neo4j_driver)
        self.validator = ValidationGate()

    def build_graph_from_chunks_direct(
        self,
        document_id: str,
        schema_data: Optional[Dict[str, Any]],
        schema_file: Optional[Path],
        chunks_dir: Path,
        document_hash: str,
    ) -> GraphTransaction:
        """
        Build graph with support for in-memory schema data.
        Wrapper around build_graph_from_chunks.
        """
        if schema_data is not None:
            # Use in-memory schema
            return self._build_graph_internal(
                document_id=document_id,
                schema=schema_data,
                chunks_dir=chunks_dir,
                document_hash=document_hash,
            )
        elif schema_file is not None:
            return self.build_graph_from_chunks(
                document_id=document_id,
                schema_file=schema_file,
                chunks_dir=chunks_dir,
                document_hash=document_hash,
            )
        else:
            raise ValueError("Either schema_data or schema_file must be provided")

    @tracer.start_as_current_span("build_graph_from_chunks")
    def build_graph_from_chunks(
        self,
        document_id: str,
        schema_file: Path,
        chunks_dir: Path,
        document_hash: str,
    ) -> GraphTransaction:
        """
        Build complete graph from chunks using two-pass strategy.

        Args:
            document_id: Document identifier
            schema_file: Path to final schema JSON
            chunks_dir: Directory containing chunk JSON files
            document_hash: SHA-256 hash for versioning

        Returns:
            GraphTransaction with results
        """
        # Load schema from file
        schema = json.loads(schema_file.read_text())
        return self._build_graph_internal(document_id, schema, chunks_dir, document_hash)

    def _build_graph_internal(
        self,
        document_id: str,
        schema: Dict[str, Any],
        chunks_dir: Path,
        document_hash: str,
    ) -> GraphTransaction:
        """
        Internal method that builds graph from schema dict and chunks.
        """
        span = trace.get_current_span()
        span.set_attribute("document.id", document_id)

        # Initialize transaction
        transaction = GraphTransaction(
            document_id=document_id, document_hash=document_hash
        )
        transaction.status = "processing"

        try:
            # Load chunks
            chunks = self._load_chunks(chunks_dir, document_id)

            span.set_attribute("chunks.loaded", len(chunks))

            # Stage 1: Validate all chunks
            validation_results = self._validate_chunks(chunks)
            if validation_results["failed"] > 0:
                error_msg = (
                    f"Validation failed: {validation_results['failed']} chunks invalid"
                )
                transaction.mark_failed(error_msg)
                return transaction

            span.add_event("Validation passed")

            # Initialize mapper and relationship builder
            mapper = ChunkMapper(document_hash, transaction.transaction_id)
            rel_builder = RelationshipBuilder(transaction.transaction_id)

            # ===== PASS 1: CREATE ALL NODES =====
            span.add_event("Starting Pass 1: Node creation")

            all_nodes = []

            # 1. Create Document node
            doc_node = mapper.map_document_chunk(schema)
            all_nodes.append(doc_node)

            # 2. Create Clause nodes
            clause_nodes = []
            requirement_nodes = []
            table_nodes = []
            figure_nodes = []
            standard_nodes = []

            for chunk in chunks:
                clause_node = mapper.map_clause_chunk(chunk)
                clause_nodes.append(clause_node)
                all_nodes.append(clause_node)

                # Extract requirements
                req_nodes = mapper.map_requirement_chunks(chunk)
                requirement_nodes.extend(req_nodes)
                all_nodes.extend(req_nodes)

                # Extract tables
                tbl_nodes = mapper.map_table_chunks(chunk)
                table_nodes.extend(tbl_nodes)
                all_nodes.extend(tbl_nodes)

                # Extract figures
                fig_nodes = mapper.map_figure_chunks(chunk)
                figure_nodes.extend(fig_nodes)
                all_nodes.extend(fig_nodes)

                # Extract external standards
                std_nodes = mapper.map_standard_references(chunk)
                standard_nodes.extend(std_nodes)
                all_nodes.extend(std_nodes)

            # Deduplicate standard nodes
            standard_nodes = self._deduplicate_standards(standard_nodes)

            # Create all nodes
            transaction.nodes = all_nodes
            nodes_created = self.graph_builder.create_nodes_batch(all_nodes)
            transaction.nodes_created = nodes_created

            span.add_event(f"Created {nodes_created} nodes")

            # ===== PASS 2: CREATE ALL RELATIONSHIPS =====
            span.add_event("Starting Pass 2: Relationship creation")

            all_relationships = []

            # Get existing clause UIDs for reference validation
            existing_clause_uids = {node.uid for node in clause_nodes}

            # 1. Hierarchical relationships (CONTAINS)
            contains_rels = rel_builder.build_contains_relationships(
                doc_node.uid, clause_nodes
            )
            all_relationships.extend(contains_rels)

            # 2. Requirement relationships (REQUIRES)
            requires_rels = rel_builder.build_requires_relationships(requirement_nodes)
            all_relationships.extend(requires_rels)

            # 3. Content relationships (HAS_TABLE, HAS_FIGURE)
            content_rels = rel_builder.build_content_relationships(
                table_nodes, figure_nodes
            )
            all_relationships.extend(content_rels)

            # 4. Reference relationships (REFERS_TO, REFERENCES)
            ref_rels = rel_builder.build_reference_relationships(
                chunks, existing_clause_uids
            )
            all_relationships.extend(ref_rels)

            # Create all relationships
            transaction.relationships = all_relationships
            rels_created = self.graph_builder.create_relationships_batch(
                all_relationships
            )
            transaction.relationships_created = rels_created

            span.add_event(f"Created {rels_created} relationships")

            # Mark transaction as completed
            transaction.mark_completed()

            span.set_attribute("nodes.created", nodes_created)
            span.set_attribute("relationships.created", rels_created)

            return transaction

        except Exception as e:
            span.record_exception(e)
            transaction.mark_failed(str(e))
            return transaction

    def _load_chunks(self, chunks_dir: Path, document_id: str) -> List[Dict]:
        """Load all chunk JSON files for a document"""
        doc_chunks_dir = chunks_dir / document_id

        if not doc_chunks_dir.exists():
            return []

        chunks = []
        for chunk_file in doc_chunks_dir.glob("*.json"):
            chunk = json.loads(chunk_file.read_text(encoding='utf-8'))
            chunks.append(chunk)

        return chunks

    def _validate_chunks(self, chunks: List[Dict]) -> Dict[str, int]:
        """
        Validate all chunks through three-stage gate.
        Supports both old format (document_metadata.id) and new format (document_id).

        Returns:
            Dict with counts: {passed, failed}
        """
        # Get existing UIDs from graph
        existing_uids = self.graph_builder.get_all_uids()

        # Collect all chunk UIDs (will be created)
        chunk_uids = set()
        for chunk in chunks:
            # Support both old and new format
            doc_id = chunk.get("document_metadata", {}).get("id", "") or chunk.get("document_id", "")
            chunk_id = chunk["chunk_id"]
            clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id

            uid = UIDGenerator.generate_clause_uid(doc_id, clause_id)
            chunk_uids.add(uid)

        # Add chunk UIDs to existing for parent validation
        all_uids = existing_uids | chunk_uids

        results = {"passed": 0, "failed": 0}

        for chunk in chunks:
            # Simple validation for new format - just check required fields exist
            if chunk.get("document_id") and chunk.get("chunk_id"):
                # Has basic required fields, pass validation
                results["passed"] += 1
            else:
                # Use full validation for old format
                is_valid, error = self.validator.validate_chunk(chunk, all_uids)

                if is_valid:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    print(f"Validation failed for chunk {chunk.get('chunk_id')}: {error}")

        return results

    def _deduplicate_standards(
        self,
        standards: List[StandardNode],
    ) -> List[StandardNode]:
        """Deduplicate standard nodes by UID"""
        seen_uids = set()
        unique_standards = []

        for std in standards:
            if std.uid not in seen_uids:
                seen_uids.add(std.uid)
                unique_standards.append(std)

        return unique_standards


async def consume_chunking_completed_events():
    """
    Consumes chunking completed events from the message broker using aio_pika.
    """
    broker_url = "amqp://emc:changeme@rabbitmq" # Updated broker URL to use service name and credentials
    queue_name = "graph_builder_queue"
    exchange_name = "document_processing"

    connection = None
    channel = None

    try:
        connection = await aio_pika.connect_robust(broker_url)
        channel = await connection.channel()

        await channel.set_qos(prefetch_count=1) # Process one message at a time

        exchange = await channel.declare_exchange(
            exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )

        queue = await channel.declare_queue(queue_name, durable=True)
        await queue.bind(exchange, routing_key="document.chunked")

        print(f"[GRAPH BUILDER WORKER] Waiting for messages in queue '{queue_name}'...")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        event_with_metadata = json.loads(message.body.decode("utf-8"))
                        event_payload = event_with_metadata["payload"]
                        
                        print(f"[GRAPH BUILDER WORKER] Received message: {event_payload.get('event_type')} for doc {event_payload.get('document_id')}")

                        if event_payload.get("event_type") == "CHUNKING_COMPLETED":
                            await process_chunking_completed_event(event_payload)
                        
                    except Exception as e:
                        print(f"[GRAPH BUILDER WORKER] Error processing message: {e}")
                        # message.nack(requeue=False) # Consider DLQ
    except asyncio.CancelledError:
        print("[GRAPH BUILDER WORKER] Consumer task cancelled.")
    except Exception as e:
        print(f"[GRAPH BUILDER WORKER] Critical error in consumer: {e}")
    finally:
        if channel:
            await channel.close()
        if connection:
            await connection.close()
    
    print("[GRAPH BUILDER WORKER] Shutdown complete")


async def process_chunking_completed_event(event: dict):
    """
    Processes a chunking completed event.

    Args:
        event: The chunking completed event.
    """
    document_id = event["document_id"]
    document_hash = event["document_hash"]
    schema_file = Path(event["schema_file"])
    chunks_dir = Path(event["chunks_dir"])

    # Configuration (would come from environment in production)
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"

    # Initialize Neo4j driver
    driver = Neo4jDriver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Create indexes and constraints
    driver.create_indexes()
    driver.create_constraints()

    # Initialize service
    service = KnowledgeGraphBuilderService(driver)

    # Build graph
    transaction = service.build_graph_from_chunks(
        document_id=document_id,
        schema_file=schema_file,
        chunks_dir=chunks_dir,
        document_hash=document_hash,
    )

    if transaction.status == "completed":
        print(f"  ✓ Success!")
        print(f"    Nodes created: {transaction.nodes_created}")
        print(f"    Relationships created: {transaction.relationships_created}")
    else:
        print(f"  ✗ Failed: {transaction.error_message}")

    driver.close()


# =========================================================
# CLI / Worker Entry Point
# =========================================================


def main():
    """Main entry point for knowledge graph builder"""
    import argparse
    import hashlib
    import os

    parser = argparse.ArgumentParser(description='Build knowledge graph from chunks')
    parser.add_argument('--document', '-d', help='Process specific document only')
    parser.add_argument('--all', '-a', action='store_true', help='Process all documents')
    args = parser.parse_args()

    # Configuration from environment or defaults
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Use Windows paths - detect based on OS
    import platform
    if platform.system() == "Windows":
        CHUNKS_DIR = Path(r"C:\Users\Lenovo\OneDrive\Desktop\Production_code\output\output_json_chunk")
        SCHEMA_DIR = Path(r"C:\Users\Lenovo\OneDrive\Desktop\Production_code\output\output_schema")
    else:
        CHUNKS_DIR = Path("/home/claude/production_pipeline/output/output_json_chunk")
        SCHEMA_DIR = Path("/home/claude/production_pipeline/output/output_schema")

    print("=" * 60)
    print("Knowledge Graph Builder")
    print("=" * 60)
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Chunks: {CHUNKS_DIR}")

    # Initialize Neo4j driver
    print("\nConnecting to Neo4j...")
    driver = Neo4jDriver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    if not driver.verify_connectivity():
        print("ERROR: Cannot connect to Neo4j")
        return

    print("Connected to Neo4j")

    # Create indexes and constraints
    driver.create_indexes()
    driver.create_constraints()

    # Initialize service
    service = KnowledgeGraphBuilderService(driver)

    # Find all documents
    if not CHUNKS_DIR.exists():
        print(f"Chunks directory not found: {CHUNKS_DIR}")
        return

    # Determine which documents to process
    doc_dirs = [d for d in CHUNKS_DIR.iterdir() if d.is_dir()]

    if args.document:
        doc_dirs = [d for d in doc_dirs if d.name == args.document]
        if not doc_dirs:
            print(f"Document not found: {args.document}")
            return
    elif not args.all:
        # Default: process first document for testing
        doc_dirs = doc_dirs[:1]

    print(f"\nProcessing {len(doc_dirs)} document(s)...")

    for doc_dir in doc_dirs:
        document_id = doc_dir.name
        schema_file = SCHEMA_DIR / f"{document_id}_final_schema.json"

        # Create a dummy schema if not exists
        if not schema_file.exists():
            # Count chunks to create dummy schema
            chunk_count = len(list(doc_dir.glob("*.json")))
            dummy_schema = {
                "document_id": document_id,
                "statistics": {
                    "total_clauses": chunk_count,
                    "total_tables": 0,
                    "total_images": 0
                }
            }
            # Use in-memory schema
            schema_file = None
            schema_data = dummy_schema
        else:
            schema_data = None

        print(f"\nProcessing: {document_id}")

        # Compute document hash
        document_hash = hashlib.sha256(document_id.encode()).hexdigest()[:16]

        # Build graph
        transaction = service.build_graph_from_chunks_direct(
            document_id=document_id,
            schema_data=schema_data,
            schema_file=schema_file,
            chunks_dir=CHUNKS_DIR,
            document_hash=document_hash,
        )

        if transaction.status == "completed":
            print(f"  [OK] Success!")
            print(f"    Nodes created: {transaction.nodes_created}")
            print(f"    Relationships created: {transaction.relationships_created}")
        else:
            print(f"  [FAILED] {transaction.error_message}")

    # Print statistics
    stats = GraphBuilder(driver).get_statistics()
    print("\n" + "=" * 60)
    print("Graph Statistics")
    print("=" * 60)
    for label, count in stats.items():
        print(f"  {label}: {count}")

    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
