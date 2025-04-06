-- Enable the pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the table if it exists (comment out if you don't want to reset)
DROP TABLE IF EXISTS stanford_law_contracts;

-- Create the table for storing document chunks and their embeddings
CREATE TABLE stanford_law_contracts (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT,              -- Original document identifier
    doc_chunk_id TEXT,        -- Identifier for the specific chunk within a document
    content TEXT,             -- The actual text content of the chunk
    embedding vector(768),    -- Vector embedding (768 dimensions for nomic-embed-text)
    metadata JSONB,           -- Additional metadata about the chunk
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for faster similarity search
CREATE INDEX ON stanford_law_contracts 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Number of lists can be adjusted based on your data size

-- Create an index on doc_id for faster lookups
CREATE INDEX ON stanford_law_contracts(doc_id);

-- Create an index on doc_chunk_id for faster lookups
CREATE INDEX ON stanford_law_contracts(doc_chunk_id);

-- Grant necessary permissions (adjust according to your needs)
-- GRANT SELECT, INSERT, UPDATE ON stanford_law_contracts TO your_user; 