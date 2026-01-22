"""
Neo4j Cypher Queries for EMC Test Plan Generation

Queries for extracting requirements, test limits, equipment,
and other test plan data from the knowledge graph.
"""


class TestPlanCypherQueries:
    """Collection of Cypher queries for test plan generation"""

    # =========================================================
    # Requirement Extraction Queries
    # =========================================================

    GET_ALL_REQUIREMENTS = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)
    WHERE d.document_id = $doc_id
    OPTIONAL MATCH (c)-[:HAS_TABLE]->(t:Table)
    OPTIONAL MATCH (c)-[:HAS_FIGURE]->(f:Figure)
    WITH c, collect(DISTINCT t) as tables, collect(DISTINCT f) as figures
    WHERE c.content_text IS NOT NULL
      AND (c.content_text CONTAINS 'shall'
           OR c.content_text CONTAINS 'must'
           OR c.content_text CONTAINS 'should'
           OR c.content_text CONTAINS 'may')
    RETURN c.uid as chunk_id,
           c.document_id as document_id,
           c.clause_id as clause_id,
           c.title as title,
           c.content_text as content,
           size(tables) as table_count,
           size(figures) as figure_count
    ORDER BY c.clause_id
    """

    GET_MANDATORY_REQUIREMENTS = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)
    WHERE d.document_id = $doc_id
      AND c.content_text IS NOT NULL
      AND (c.content_text CONTAINS 'shall' OR c.content_text CONTAINS 'must')
    OPTIONAL MATCH (c)-[:HAS_TABLE]->(t:Table)
    OPTIONAL MATCH (c)-[:HAS_FIGURE]->(f:Figure)
    RETURN c.uid as chunk_id,
           c.document_id as document_id,
           c.clause_id as clause_id,
           c.title as title,
           c.content_text as content,
           collect(DISTINCT t.caption) as tables,
           collect(DISTINCT f.caption) as figures
    ORDER BY c.clause_id
    """

    GET_CLAUSES_WITH_CONTENT = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)
    WHERE d.document_id = $doc_id
      AND c.content_text IS NOT NULL
      AND length(c.content_text) > 50
    RETURN c.uid as chunk_id,
           c.document_id as document_id,
           c.clause_id as clause_id,
           c.title as title,
           c.content_text as content
    ORDER BY c.clause_id
    """

    # =========================================================
    # Table & Figure Queries
    # =========================================================

    GET_TABLES_FOR_DOCUMENT = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)-[:HAS_TABLE]->(t:Table)
    WHERE d.document_id = $doc_id
    RETURN t.uid as table_id,
           t.table_number as table_number,
           t.caption as caption,
           c.uid as source_clause_id,
           c.clause_id as clause_id,
           c.title as clause_title
    ORDER BY t.table_number
    """

    GET_FIGURES_FOR_DOCUMENT = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)-[:HAS_FIGURE]->(f:Figure)
    WHERE d.document_id = $doc_id
    RETURN f.uid as figure_id,
           f.figure_number as figure_number,
           f.caption as caption,
           f.image_path as image_path,
           c.uid as source_clause_id,
           c.clause_id as clause_id,
           c.title as clause_title
    ORDER BY f.figure_number
    """

    GET_TEST_LIMIT_TABLES = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)-[:HAS_TABLE]->(t:Table)
    WHERE d.document_id = $doc_id
      AND t.caption IS NOT NULL
      AND (t.caption CONTAINS 'limit'
           OR t.caption CONTAINS 'Limit'
           OR t.caption CONTAINS 'level'
           OR t.caption CONTAINS 'Level'
           OR t.caption CONTAINS 'specification'
           OR t.caption CONTAINS 'requirement')
    RETURN t.uid as table_id,
           t.table_number as table_number,
           t.caption as caption,
           c.uid as source_clause_id,
           c.clause_id as clause_id,
           c.title as clause_title,
           c.content_text as context
    """

    GET_TEST_SETUP_FIGURES = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)-[:HAS_FIGURE]->(f:Figure)
    WHERE d.document_id = $doc_id
      AND f.caption IS NOT NULL
      AND (f.caption CONTAINS 'setup'
           OR f.caption CONTAINS 'Setup'
           OR f.caption CONTAINS 'arrangement'
           OR f.caption CONTAINS 'configuration'
           OR f.caption CONTAINS 'test'
           OR f.caption CONTAINS 'Test')
    RETURN f.uid as figure_id,
           f.figure_number as figure_number,
           f.caption as caption,
           f.image_path as image_path,
           c.uid as source_clause_id,
           c.clause_id as clause_id,
           c.title as clause_title
    """

    # =========================================================
    # External Reference Queries
    # =========================================================

    GET_EXTERNAL_REFERENCES = """
    MATCH (c:Clause)-[:REFERENCES]->(s:Standard)
    WHERE c.document_id = $doc_id
    RETURN c.uid as clause_id,
           c.clause_id as clause_number,
           collect(s.standard_name) as referenced_standards
    """

    # =========================================================
    # Equipment Mention Queries
    # =========================================================

    GET_EQUIPMENT_MENTIONS = """
    MATCH (d:Document)-[:CONTAINS*]->(c:Clause)
    WHERE d.document_id = $doc_id
      AND c.content_text IS NOT NULL
      AND (c.content_text CONTAINS 'equipment'
           OR c.content_text CONTAINS 'apparatus'
           OR c.content_text CONTAINS 'generator'
           OR c.content_text CONTAINS 'receiver'
           OR c.content_text CONTAINS 'antenna'
           OR c.content_text CONTAINS 'LISN'
           OR c.content_text CONTAINS 'amplifier'
           OR c.content_text CONTAINS 'meter')
    RETURN c.uid as chunk_id,
           c.clause_id as clause_id,
           c.title as title,
           c.content_text as content
    """

    # =========================================================
    # Document Statistics
    # =========================================================

    GET_DOCUMENT_STATS = """
    MATCH (d:Document {document_id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS*]->(c:Clause)
    OPTIONAL MATCH (d)-[:CONTAINS*]->(:Clause)-[:HAS_TABLE]->(t:Table)
    OPTIONAL MATCH (d)-[:CONTAINS*]->(:Clause)-[:HAS_FIGURE]->(f:Figure)
    RETURN d.document_id as document_id,
           count(DISTINCT c) as clause_count,
           count(DISTINCT t) as table_count,
           count(DISTINCT f) as figure_count
    """

    GET_ALL_DOCUMENTS = """
    MATCH (d:Document)
    RETURN d.document_id as document_id,
           d.total_clauses as clause_count
    ORDER BY d.document_id
    """
