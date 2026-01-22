"""
Neo4j Graph Database Driver

Production-grade Neo4j driver with:
- Transaction management
- Batch operations
- Idempotent writes (MERGE instead of CREATE)
- Connection pooling
- Retry logic
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Transaction, Session
from neo4j.exceptions import ServiceUnavailable, TransientError
import time
from datetime import datetime

from opentelemetry import trace

from models.graph_schema import (
    GraphNode, GraphRelationship, GraphTransaction,
    NodeType, RelationshipType
)


tracer = trace.get_tracer(__name__)


class Neo4jDriver:
    """
    Production Neo4j driver with connection pooling and retry logic.
    """
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50
    ):
        self.uri = uri
        self.database = database
        
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size
        )
    
    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
    
    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j"""
        try:
            self.driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            return False
    
    @tracer.start_as_current_span("neo4j_execute_query")
    def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        retries: int = 3
    ) -> List[Dict]:
        """
        Execute a Cypher query with retry logic.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            retries: Number of retry attempts
        
        Returns:
            List of result records as dictionaries
        """
        span = trace.get_current_span()
        span.set_attribute("neo4j.query", query[:100])  # First 100 chars
        
        for attempt in range(retries):
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(query, parameters or {})
                    records = [record.data() for record in result]
                    
                    span.set_attribute("neo4j.records_returned", len(records))
                    return records
            
            except TransientError as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    span.record_exception(e)
                    raise
            
            except Exception as e:
                span.record_exception(e)
                raise
        
        return []
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        indexes = [
            # Primary UID indexes
            "CREATE INDEX node_uid IF NOT EXISTS FOR (n:Document) ON (n.uid)",
            "CREATE INDEX clause_uid IF NOT EXISTS FOR (n:Clause) ON (n.uid)",
            "CREATE INDEX requirement_uid IF NOT EXISTS FOR (n:Requirement) ON (n.uid)",
            "CREATE INDEX table_uid IF NOT EXISTS FOR (n:Table) ON (n.uid)",
            "CREATE INDEX figure_uid IF NOT EXISTS FOR (n:Figure) ON (n.uid)",
            "CREATE INDEX standard_uid IF NOT EXISTS FOR (n:Standard) ON (n.uid)",
            
            # Secondary indexes for queries
            "CREATE INDEX document_id IF NOT EXISTS FOR (n:Document) ON (n.document_id)",
            "CREATE INDEX clause_document_id IF NOT EXISTS FOR (n:Clause) ON (n.document_id)",
            "CREATE INDEX clause_id IF NOT EXISTS FOR (n:Clause) ON (n.clause_id)",
            "CREATE INDEX document_hash IF NOT EXISTS FOR (n:Document) ON (n.document_hash)",
            
            # Composite indexes
            "CREATE INDEX clause_doc_clause IF NOT EXISTS FOR (n:Clause) ON (n.document_id, n.clause_id)",
        ]
        
        for index_query in indexes:
            try:
                self.execute_query(index_query)
            except Exception as e:
                print(f"Index creation warning: {e}")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        constraints = [
            "CREATE CONSTRAINT document_uid_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT clause_uid_unique IF NOT EXISTS FOR (n:Clause) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT requirement_uid_unique IF NOT EXISTS FOR (n:Requirement) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT table_uid_unique IF NOT EXISTS FOR (n:Table) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT figure_uid_unique IF NOT EXISTS FOR (n:Figure) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT standard_uid_unique IF NOT EXISTS FOR (n:Standard) REQUIRE n.uid IS UNIQUE",
        ]
        
        for constraint_query in constraints:
            try:
                self.execute_query(constraint_query)
            except Exception as e:
                print(f"Constraint creation warning: {e}")


class CypherBuilder:
    """
    Builds Cypher queries for idempotent graph operations.
    
    Uses MERGE instead of CREATE to ensure idempotency:
    - If node exists: update properties
    - If node doesn't exist: create it
    """
    
    @staticmethod
    def build_merge_node_query(node: GraphNode) -> tuple[str, Dict]:
        """
        Build MERGE query for a node.

        Returns:
            (cypher_query, parameters)
        """
        # Handle both enum and string types
        node_label = node.node_type
        if hasattr(node_label, 'value'):
            node_label = node_label.value
        
        # Start with MERGE on UID
        query = f"MERGE (n:{node_label} {{uid: $uid}})\n"
        
        # Set all properties
        set_clauses = []
        params = {"uid": node.uid}
        
        # Add all fields as properties
        for field, value in node.dict(exclude={'node_type', 'properties'}).items():
            if value is not None:
                param_name = f"prop_{field}"
                set_clauses.append(f"n.{field} = ${param_name}")
                
                # Handle datetime serialization
                if isinstance(value, datetime):
                    params[param_name] = value.isoformat()
                else:
                    params[param_name] = value
        
        # Add custom properties
        for key, value in node.properties.items():
            param_name = f"custom_{key}"
            set_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value
        
        if set_clauses:
            query += "SET " + ",\n    ".join(set_clauses) + "\n"
        
        query += "RETURN n"
        
        return query, params
    
    @staticmethod
    def build_merge_relationship_query(relationship: GraphRelationship) -> tuple[str, Dict]:
        """
        Build MERGE query for a relationship.

        Returns:
            (cypher_query, parameters)
        """
        # Handle both enum and string types
        rel_type = relationship.relationship_type
        if hasattr(rel_type, 'value'):
            rel_type = rel_type.value
        
        query = f"""
        MATCH (source {{uid: $source_uid}})
        MATCH (target {{uid: $target_uid}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.created_at = $created_at
        """
        
        params = {
            "source_uid": relationship.source_uid,
            "target_uid": relationship.target_uid,
            "created_at": relationship.created_at.isoformat()
        }
        
        # Add custom properties
        for key, value in relationship.properties.items():
            query += f", r.{key} = ${key}\n"
            params[key] = value
        
        if relationship.transaction_id:
            query += ", r.transaction_id = $transaction_id\n"
            params["transaction_id"] = relationship.transaction_id
        
        query += "RETURN r"
        
        return query, params
    
    @staticmethod
    def build_batch_merge_nodes_query(nodes: List[GraphNode]) -> tuple[str, Dict]:
        """
        Build batch MERGE query for multiple nodes of the same type.
        More efficient than individual queries.

        Returns:
            (cypher_query, parameters)
        """
        if not nodes:
            return "", {}

        # Handle both enum and string types
        node_label = nodes[0].node_type
        if hasattr(node_label, 'value'):
            node_label = node_label.value
        
        # Build list of node dictionaries
        node_dicts = []
        for node in nodes:
            node_dict = node.dict(exclude={'node_type', 'properties'})
            # Handle datetime serialization
            for key, value in node_dict.items():
                if isinstance(value, datetime):
                    node_dict[key] = value.isoformat()
            node_dicts.append(node_dict)
        
        query = f"""
        UNWIND $nodes AS nodeData
        MERGE (n:{node_label} {{uid: nodeData.uid}})
        SET n += nodeData
        RETURN n
        """
        
        params = {"nodes": node_dicts}
        
        return query, params


class GraphBuilder:
    """
    High-level graph builder that orchestrates node and relationship creation.
    Implements the two-pass ingestion strategy.
    """
    
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver
        self.cypher = CypherBuilder()
    
    @tracer.start_as_current_span("create_node")
    def create_node(self, node: GraphNode) -> bool:
        """
        Create or update a single node.
        
        Args:
            node: Node to create/update
        
        Returns:
            True if successful
        """
        query, params = self.cypher.build_merge_node_query(node)
        
        try:
            self.driver.execute_query(query, params)
            return True
        except Exception as e:
            print(f"Error creating node {node.uid}: {e}")
            return False
    
    @tracer.start_as_current_span("create_nodes_batch")
    def create_nodes_batch(self, nodes: List[GraphNode]) -> int:
        """
        Create or update multiple nodes in batch.
        Groups nodes by type for efficient batch operations.
        
        Args:
            nodes: List of nodes to create/update
        
        Returns:
            Number of nodes successfully created
        """
        span = trace.get_current_span()
        span.set_attribute("nodes.count", len(nodes))
        
        # Group by node type
        nodes_by_type = {}
        for node in nodes:
            node_type = node.node_type.value
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        total_created = 0
        
        for node_type, type_nodes in nodes_by_type.items():
            span.add_event(f"Creating {len(type_nodes)} {node_type} nodes")
            
            # Batch size for efficient processing
            batch_size = 100
            for i in range(0, len(type_nodes), batch_size):
                batch = type_nodes[i:i+batch_size]
                
                query, params = self.cypher.build_batch_merge_nodes_query(batch)
                
                try:
                    self.driver.execute_query(query, params)
                    total_created += len(batch)
                except Exception as e:
                    print(f"Error creating batch of {node_type} nodes: {e}")
        
        span.set_attribute("nodes.created", total_created)
        return total_created
    
    @tracer.start_as_current_span("create_relationship")
    def create_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Create or update a single relationship.
        
        Args:
            relationship: Relationship to create/update
        
        Returns:
            True if successful
        """
        query, params = self.cypher.build_merge_relationship_query(relationship)
        
        try:
            self.driver.execute_query(query, params)
            return True
        except Exception as e:
            print(f"Error creating relationship {relationship.source_uid} -> {relationship.target_uid}: {e}")
            return False
    
    @tracer.start_as_current_span("create_relationships_batch")
    def create_relationships_batch(self, relationships: List[GraphRelationship]) -> int:
        """
        Create or update multiple relationships in batch.
        
        Args:
            relationships: List of relationships to create/update
        
        Returns:
            Number of relationships successfully created
        """
        span = trace.get_current_span()
        span.set_attribute("relationships.count", len(relationships))
        
        total_created = 0
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            
            for rel in batch:
                if self.create_relationship(rel):
                    total_created += 1
        
        span.set_attribute("relationships.created", total_created)
        return total_created
    
    def node_exists(self, uid: str) -> bool:
        """Check if a node exists by UID"""
        query = "MATCH (n {uid: $uid}) RETURN count(n) as count"
        results = self.driver.execute_query(query, {"uid": uid})
        return results[0]["count"] > 0 if results else False
    
    def get_all_uids(self) -> set:
        """Get all existing UIDs in the graph"""
        query = "MATCH (n) RETURN n.uid as uid"
        results = self.driver.execute_query(query)
        return {r["uid"] for r in results if r.get("uid")}
    
    def delete_document_graph(self, document_id: str) -> int:
        """
        Delete entire graph for a document.
        Used when re-processing a document.
        
        Returns:
            Number of nodes deleted
        """
        query = """
        MATCH (n)
        WHERE n.document_id = $document_id
        DETACH DELETE n
        RETURN count(n) as count
        """
        
        results = self.driver.execute_query(query, {"document_id": document_id})
        return results[0]["count"] if results else 0
    
    def get_statistics(self) -> Dict[str, int]:
        """Get graph statistics"""
        stats_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        """
        
        results = self.driver.execute_query(stats_query)
        stats = {r["label"]: r["count"] for r in results}
        
        # Count relationships
        rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
        rel_results = self.driver.execute_query(rel_query)
        stats["relationships"] = rel_results[0]["count"] if rel_results else 0
        
        return stats
