import os
from dotenv import load_dotenv
import logging
import numpy as np
import psycopg2
import psycopg2.extras

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DirectVectorStore:
    """A simple vector store implementation using direct SQL queries."""

    def __init__(self, connection_string, table_name="stanford_law_contracts"):
        self.connection_string = connection_string
        self.table_name = table_name

    def add_vector(self, text, embedding, doc_id, doc_chunk_id, metadata=None):
        """Add a vector to the database."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Format vector - pgvector expects JSON array format with square brackets
            vector_str = str(
                embedding if isinstance(embedding, list) else embedding.tolist()
            )

            # Default metadata
            if metadata is None:
                metadata = {}

            # Convert metadata to JSON string
            metadata_str = str(metadata).replace("'", '"')

            # Insert vector
            cursor.execute(
                f"""
                INSERT INTO {self.table_name} (doc_id, doc_chunk_id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                RETURNING id;
            """,
                (doc_id, doc_chunk_id, text, vector_str, metadata_str),
            )

            inserted_id = cursor.fetchone()[0]
            conn.commit()

            cursor.close()
            conn.close()

            return inserted_id
        except Exception as e:
            logger.error(f"Error adding vector: {str(e)}")
            raise

    def search(self, query_embedding, top_k=5, filter_doc_id=None):
        """Search for similar vectors."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Format query embedding
            vector_str = str(
                query_embedding
                if isinstance(query_embedding, list)
                else query_embedding.tolist()
            )

            # Build query with optional filter
            query = f"""
                SELECT id, content, doc_id, doc_chunk_id, metadata,
                       embedding <-> %s::vector as distance
                FROM {self.table_name}
            """

            params = [vector_str]

            # Add filter if requested
            if filter_doc_id:
                query += " WHERE doc_id = %s"
                params.append(filter_doc_id)

            # Add ordering and limit
            query += f" ORDER BY distance LIMIT {top_k};"

            # Execute query
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise


def test_vector_store():
    """Test the DirectVectorStore."""
    try:
        # Get connection string
        connection_string = os.getenv("DB_URL")
        if not connection_string:
            raise ValueError("DB_URL environment variable is not set")

        # Initialize our DirectVectorStore
        vector_store = DirectVectorStore(connection_string)

        # Generate test data
        test_vector = np.random.rand(768).astype(np.float32)
        test_text = "This is a test vector for direct SQL"

        # Add a vector
        logger.info("Adding test vector...")
        inserted_id = vector_store.add_vector(
            text=test_text,
            embedding=test_vector,
            doc_id="test_doc_direct",
            doc_chunk_id=f"chunk_{int(np.random.rand(1)[0] * 10000)}",
            metadata={"source": "test", "timestamp": "2023-06-01"},
        )
        logger.info(f"Added vector with ID: {inserted_id}")

        # Search for similar vectors
        logger.info("Searching for similar vectors...")
        results = vector_store.search(test_vector, top_k=3)

        # Display results
        logger.info(f"Found {len(results)} results")
        for i, result in enumerate(results):
            logger.info(
                f"Result {i+1}: ID={result['id']}, Distance={result['distance']:.4f}"
            )
            logger.info(f"  Content: {result['content'][:50]}...")
            logger.info(
                f"  Doc ID: {result['doc_id']}, Chunk ID: {result['doc_chunk_id']}"
            )

        logger.info("Vector store test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Vector store test failed: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting direct SQL vector store test...")
    success = test_vector_store()
    if success:
        logger.info("✅ Direct SQL vector store is working correctly")
    else:
        logger.error("❌ Direct SQL vector store test failed")
