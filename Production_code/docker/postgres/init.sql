-- PostgreSQL initialization script for EMC Document Registry

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types
CREATE TYPE document_status AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- Documents table - Core metadata registry
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 hash for deduplication
    filename VARCHAR(512) NOT NULL,
    upload_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    status document_status NOT NULL DEFAULT 'PENDING',
    
    -- Processing metadata
    marker_version VARCHAR(50),
    processing_started TIMESTAMP WITH TIME ZONE,
    processing_completed TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    
    -- Storage paths
    s3_raw_path TEXT,
    s3_json_path TEXT,
    s3_schema_path TEXT,
    
    -- Retry tracking
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    
    -- Statistics
    page_count INTEGER,
    clause_count INTEGER,
    chunk_count INTEGER,
    total_images INTEGER,
    total_tables INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX idx_documents_hash ON documents(hash);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_upload_timestamp ON documents(upload_timestamp DESC);
CREATE INDEX idx_documents_processing_completed ON documents(processing_completed DESC);

-- Processing events table - Audit trail
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,  -- DOCUMENT_UPLOADED, EXTRACTION_COMPLETED, etc.
    event_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_processing_events_document_id ON processing_events(document_id);
CREATE INDEX idx_processing_events_created_at ON processing_events(created_at DESC);

-- Chunks table - Track individual chunks for the knowledge graph
CREATE TABLE chunks (
    chunk_id VARCHAR(512) PRIMARY KEY,  -- Format: DOC_ID:CLAUSE_ID
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_id VARCHAR(255) NOT NULL,
    
    -- Hierarchy
    parent_chunk_id VARCHAR(512),
    level INTEGER NOT NULL,
    
    -- Content metadata
    content_length INTEGER,
    has_requirements BOOLEAN NOT NULL DEFAULT FALSE,
    has_tables BOOLEAN NOT NULL DEFAULT FALSE,
    has_figures BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Graph ingestion
    neo4j_node_id BIGINT,  -- Neo4j internal ID
    vector_embedding_id VARCHAR(255),  -- Vector DB ID
    
    -- Versioning
    version INTEGER NOT NULL DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_parent_chunk_id ON chunks(parent_chunk_id);
CREATE INDEX idx_chunks_clause_id ON chunks(clause_id);

-- External references table - Track cross-document references
CREATE TABLE external_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_chunk_id VARCHAR(512) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    target_standard VARCHAR(255) NOT NULL,  -- e.g., "ISO 9001", "IEC 61000-4-3"
    reference_type VARCHAR(50),  -- 'standard', 'clause', 'table', 'figure'
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_external_refs_source_chunk_id ON external_references(source_chunk_id);
CREATE INDEX idx_external_refs_target_standard ON external_references(target_standard);

-- Dead Letter Queue table - Failed documents requiring manual intervention
CREATE TABLE dead_letter_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    failure_stage VARCHAR(100) NOT NULL,  -- 'marker_extraction', 'schema_building', 'chunking'
    error_message TEXT NOT NULL,
    error_details JSONB,
    retry_count INTEGER NOT NULL,
    
    -- Manual intervention tracking
    reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dlq_document_id ON dead_letter_queue(document_id);
CREATE INDEX idx_dlq_reviewed ON dead_letter_queue(reviewed);
CREATE INDEX idx_dlq_created_at ON dead_letter_queue(created_at DESC);

-- Processing statistics table - Aggregated metrics
CREATE TABLE processing_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stat_date DATE NOT NULL,
    
    -- Document counts
    total_documents INTEGER NOT NULL DEFAULT 0,
    documents_completed INTEGER NOT NULL DEFAULT 0,
    documents_failed INTEGER NOT NULL DEFAULT 0,
    documents_pending INTEGER NOT NULL DEFAULT 0,
    
    -- Processing times (seconds)
    avg_marker_time NUMERIC(10, 2),
    avg_schema_time NUMERIC(10, 2),
    avg_chunking_time NUMERIC(10, 2),
    avg_total_time NUMERIC(10, 2),
    
    -- Content statistics
    total_pages INTEGER NOT NULL DEFAULT 0,
    total_clauses INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(stat_date)
);

CREATE INDEX idx_processing_stats_stat_date ON processing_statistics(stat_date DESC);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for auto-updating updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries

-- Active documents view
CREATE VIEW active_documents AS
SELECT 
    d.*,
    EXTRACT(EPOCH FROM (processing_completed - processing_started)) AS processing_duration_seconds
FROM documents d
WHERE status IN ('PENDING', 'PROCESSING');

-- Failed documents view
CREATE VIEW failed_documents AS
SELECT 
    d.*,
    dlq.failure_stage,
    dlq.reviewed,
    dlq.reviewed_by
FROM documents d
LEFT JOIN dead_letter_queue dlq ON d.id = dlq.document_id
WHERE d.status = 'FAILED';

-- Document processing summary view
CREATE VIEW document_processing_summary AS
SELECT 
    d.id,
    d.filename,
    d.status,
    d.upload_timestamp,
    d.processing_completed,
    EXTRACT(EPOCH FROM (d.processing_completed - d.upload_timestamp)) AS total_processing_seconds,
    d.page_count,
    d.clause_count,
    d.chunk_count,
    COUNT(DISTINCT c.chunk_id) AS actual_chunk_count
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id;

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO emc_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO emc_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO emc_user;

-- Insert initial statistics record
INSERT INTO processing_statistics (stat_date)
VALUES (CURRENT_DATE)
ON CONFLICT (stat_date) DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'EMC Document Registry schema initialized successfully';
END $$;
