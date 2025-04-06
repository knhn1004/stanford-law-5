from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
from pathlib import Path
from llama_index.core import (
    Document,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PDFReader
from pypdf import PdfReader
from dotenv import load_dotenv
import logging
import time
import numpy as np
import psycopg2
import psycopg2.extras
from groq import Groq
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Initialize FastAPI app
app = FastAPI(title="Contract Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# DirectVectorStore implementation using direct SQL
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


# Initialize components
# We'll keep Ollama for embeddings but use Groq for LLM
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    request_timeout=30.0,
    embed_batch_size=1,
    additional_kwargs={
        "temperature": 0.0,
        "num_thread": 4,
    },
    dimensions=768,  # Explicitly set dimensions
)

# Set the default embedding model
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.num_output = 512  # Limit output size
Settings.context_window = 3072  # Set context window


# Initialize embedding model with retries
def get_embedding_with_retries(text: str, max_retries: int = 3) -> list[float]:
    """Get embeddings with retry logic."""
    for attempt in range(max_retries):
        try:
            # Ensure text is not empty
            if not text.strip():
                logger.warning("Empty text provided for embedding, using zero vector")
                return [0.0] * 768

            # Get embedding with explicit type conversion
            embedding = embed_model.get_text_embedding(text)

            # Handle None, empty, or string 'None' embedding
            if (
                embedding is None
                or len(embedding) == 0
                or any(x == "None" for x in embedding)
            ):
                logger.warning(
                    f"Attempt {attempt + 1}: Received invalid embedding (None values), retrying..."
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    logger.warning("All retries failed, using zero vector")
                    return [0.0] * 768

            # Convert string values to float, replacing 'None' with 0.0
            try:
                embedding = [float(x) if x != "None" else 0.0 for x in embedding]
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Attempt {attempt + 1}: Failed to convert embedding values: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    logger.warning("All retries failed, using zero vector")
                    return [0.0] * 768

            # Convert to numpy array for validation
            embedding_array = np.array(embedding, dtype=np.float64)

            # Replace any remaining None or nan values with 0.0
            embedding_array = np.nan_to_num(embedding_array, 0.0)

            # Check for NaN or infinite values after conversion
            if not np.all(np.isfinite(embedding_array)):
                logger.warning(
                    f"Attempt {attempt + 1}: Found non-finite values after conversion, retrying..."
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    logger.warning("All retries failed, using zero vector")
                    return [0.0] * 768

            # Ensure correct dimension
            if len(embedding_array) != 768:
                raise ValueError(f"Wrong embedding dimension: {len(embedding_array)}")

            # Convert back to list of floats with explicit conversion
            return [float(x) for x in embedding_array]

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.warning("All retries failed, using zero vector")
                return [0.0] * 768

    # This should never be reached due to the return in the last else block
    return [0.0] * 768


pdf_reader = PDFReader()


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    doc_id: Optional[str]


def get_vector_store():
    """Get the vector store instance."""
    try:
        connection_string = os.getenv("DB_URL")
        if not connection_string:
            raise ValueError("DB_URL environment variable is not set")

        logger.info("Initializing DirectVectorStore...")
        vector_store = DirectVectorStore(
            connection_string=connection_string, table_name="stanford_law_contracts"
        )

        logger.info("Vector store initialized successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise ValueError(f"Vector store initialization failed: {str(e)}")


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyPDF2 as a fallback method."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text with PyPDF2: {str(e)}")
        raise


def validate_embeddings(text: str) -> bool:
    """Validate that we can generate embeddings for the text."""
    try:
        # Try to generate an embedding for a small sample
        sample = text[:500]  # Take a smaller sample to test
        logger.info("Testing embedding generation with sample text...")

        # Get embedding with retries - this will now always return a valid float list
        embedding = get_embedding_with_retries(sample)

        # Convert to numpy array for validation
        embedding_array = np.array(embedding, dtype=np.float64)

        # Check dimensions
        if embedding_array.shape[0] != 768:
            logger.error(f"Wrong embedding dimension: {embedding_array.shape[0]}")
            return False

        logger.info("Embedding validation successful")
        return True

    except Exception as e:
        logger.error(f"Error validating embeddings: {str(e)}")
        return False


def process_pdf(file_path: str, doc_id: str):
    """Process a PDF file and store it in the vector store."""
    try:
        logger.info(f"Processing PDF: {file_path}")

        # Try primary PDF reader first
        try:
            logger.info("Attempting to read PDF with primary reader")
            documents = pdf_reader.load_data(file_path)
            text = "\n\n".join([d.text for d in documents])
        except Exception as e:
            logger.warning(f"Primary PDF reader failed: {str(e)}")
            logger.info("Falling back to PyPDF2")
            text = extract_text_from_pdf(file_path)

        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")

        logger.info(f"Successfully extracted {len(text)} characters of text")

        # Validate that we can generate embeddings
        if not validate_embeddings(text):
            raise ValueError("Failed to validate embeddings")

        # Clean and preprocess the text
        text = text.replace("\x00", " ")  # Remove null bytes
        text = " ".join(text.split())  # Normalize whitespace

        # Create document with metadata
        doc = Document(
            text=text,
            metadata={
                "doc_id": doc_id,
                "file_name": Path(file_path).name,
                "doc_hash": str(hash(text)),
            },
        )

        try:
            # Initialize vector store
            logger.info("Initializing vector store")
            vector_store = get_vector_store()

            # Create parser with smaller chunk size
            logger.info("Creating node parser")
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap
            )

            # Create nodes
            logger.info("Creating nodes from document")
            nodes = node_parser.get_nodes_from_documents([doc])

            if not nodes:
                raise ValueError("No nodes were created from the document")

            logger.info(f"Created {len(nodes)} nodes from document")

            # Store nodes directly using our DirectVectorStore
            logger.info("Storing nodes in vector store...")
            for i, node in enumerate(nodes):
                try:
                    # Generate embedding for the node
                    embedding = get_embedding_with_retries(node.text)

                    # Create unique chunk ID
                    chunk_id = f"{doc_id}_chunk_{i}"

                    # Extract metadata
                    metadata = node.metadata

                    # Store in vector store directly
                    vector_store.add_vector(
                        text=node.text,
                        embedding=embedding,
                        doc_id=doc_id,
                        doc_chunk_id=chunk_id,
                        metadata=metadata,
                    )

                    logger.info(f"Successfully stored node {i+1}/{len(nodes)}")
                except Exception as e:
                    logger.error(f"Failed to store node {i+1}: {str(e)}")
                    raise

            logger.info(f"Successfully stored all {len(nodes)} nodes")
            return True

        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            raise ValueError(f"Error during document indexing: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    try:
        # Validate using mimetype instead of file signature
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, detail="Invalid file type. Only PDF files are allowed."
            )

        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Saved uploaded file to temporary path: {tmp_path}")

        # Generate a unique doc_id (using the filename without extension)
        doc_id = Path(file.filename).stem

        # Process the PDF synchronously
        process_pdf(tmp_path, doc_id)

        # Clean up
        os.unlink(tmp_path)

        return {"message": "PDF processed successfully", "doc_id": doc_id}

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the document(s) with optional document filtering."""
    try:
        # Get vector store
        vector_store = get_vector_store()

        # Get embedding for the query
        query_embedding = get_embedding_with_retries(request.query)

        # Search for similar vectors
        filter_doc_id = request.doc_id if request.doc_id else None
        results = vector_store.search(
            query_embedding=query_embedding, top_k=5, filter_doc_id=filter_doc_id
        )

        if not results:
            return QueryResponse(
                response=json.dumps({
                    "contractName": f"Contract {request.doc_id}",
                    "description": "No relevant information found in the contract.",
                    "metrics": {
                        "overallFairnessScore": 50,
                        "potentialBiasIndicators": 0,
                        "highRiskClauses": 0,
                        "balancedClauses": 0
                    },
                    "sentimentDistribution": {
                        "vendorFavorable": 25,
                        "balanced": 25,
                        "customerFavorable": 25,
                        "neutral": 25
                    },
                    "notableClauses": [],
                    "summary": {
                        "title": "Analysis Summary",
                        "description": "No relevant contract sections found for analysis.",
                        "points": [],
                        "riskAssessment": {
                            "level": "neutral",
                            "label": "Unknown",
                            "description": "Insufficient information for risk assessment."
                        }
                    }
                }),
                doc_id=request.doc_id,
            )

        # Extract the content from the most relevant results
        contexts = [result["content"] for result in results]

        # Create messages for Groq chat completion with structured output instruction
        messages = [
            {
                "role": "system",
                "content": """You are a contract analysis assistant that provides structured analysis of legal documents.
                Your responses must be in valid JSON format following the exact schema provided.
                Do not include any explanatory text or markdown formatting outside the JSON structure.
                If you are unsure about any values, use reasonable defaults rather than omitting fields.
                Ensure all string values are properly escaped and all numbers are valid."""
            },
            {
                "role": "user",
                "content": f"""Analyze the following contract text for sentiment, bias, and fairness.
                Focus on identifying vendor-favorable vs customer-favorable clauses.
                Provide an overall fairness score out of 100, risk assessment, and recommendations.

                Contract text from relevant sections:
                {' '.join(contexts)}

                Return your analysis in this exact JSON structure, with no additional text or formatting:
                {{
                  "contractName": string,
                  "description": string,
                  "metrics": {{
                    "overallFairnessScore": number,
                    "potentialBiasIndicators": number,
                    "highRiskClauses": number,
                    "balancedClauses": number
                  }},
                  "sentimentDistribution": {{
                    "vendorFavorable": number,
                    "balanced": number,
                    "customerFavorable": number,
                    "neutral": number
                  }},
                  "notableClauses": [
                    {{
                      "type": string,
                      "sentiment": string,
                      "sentimentLabel": string,
                      "biasScore": number,
                      "riskLevel": string,
                      "riskLabel": string,
                      "text": string,
                      "analysis": string,
                      "biasIndicators": [
                        {{
                          "label": string,
                          "value": number
                        }}
                      ],
                      "industryComparison": string,
                      "recommendations": [string]
                    }}
                  ],
                  "summary": {{
                    "title": string,
                    "description": string,
                    "points": [
                      {{
                        "title": string,
                        "description": string
                      }}
                    ],
                    "riskAssessment": {{
                      "level": string,
                      "label": string,
                      "description": string
                    }}
                  }}
                }}"""
            }
        ]

        # Get response from Groq
        try:
            completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Parse the response to ensure it's valid JSON and properly formatted
            try:
                response_json = json.loads(completion.choices[0].message.content)
                response_text = json.dumps(response_json)  # Re-serialize to ensure proper formatting
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from LLM: {str(e)}")
                # Return a default response if JSON is invalid
                response_text = json.dumps({
                    "contractName": f"Contract {request.doc_id}",
                    "description": "Error processing contract analysis.",
                    "metrics": {
                        "overallFairnessScore": 50,
                        "potentialBiasIndicators": 0,
                        "highRiskClauses": 0,
                        "balancedClauses": 0
                    },
                    "sentimentDistribution": {
                        "vendorFavorable": 25,
                        "balanced": 25,
                        "customerFavorable": 25,
                        "neutral": 25
                    },
                    "notableClauses": [],
                    "summary": {
                        "title": "Analysis Error",
                        "description": "An error occurred while analyzing the contract.",
                        "points": [],
                        "riskAssessment": {
                            "level": "neutral",
                            "label": "Unknown",
                            "description": "Analysis failed due to technical issues."
                        }
                    }
                })
        except Exception as e:
            logger.error(f"Error getting completion from Groq: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get response from LLM")

        return QueryResponse(response=response_text, doc_id=request.doc_id)

    except Exception as e:
        logger.error(f"Error in query_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
