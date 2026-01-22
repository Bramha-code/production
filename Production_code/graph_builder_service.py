
import json
import asyncio
from pathlib import Path
async def consume_chunking_completed_events():
    """Consumes chunking completed events from the message broker."""
    event_log = Path("/home/claude/production_pipeline/events.jsonl")
    processed_events = set()

    while True:
        try:
            if event_log.exists():
                with open(event_log, "r") as f:
                    for line in f:
                        event = json.loads(line)
                        event_id = (
                            event.get("timestamp")
                            + event["payload"].get("document_id", "")
                        )

                        # Skip already processed
                        if event_id in processed_events:
                            continue

                        # Only process CHUNKING_COMPLETED events
                        if event["payload"].get("event_type") == "CHUNKING_COMPLETED":
                            # Create task
                            task = asyncio.create_task(
                                process_chunking_completed_event(event["payload"])
                            )
                            processed_events.add(event_id)

            # Wait before checking for new events
            await asyncio.sleep(2)

        except Exception as e:
            print(f"Error consuming events: {e}")
            await asyncio.sleep(5)


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