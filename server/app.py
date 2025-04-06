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
import requests
import html
import re

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


# Simple DuckDuckGo search function
def duckduckgo_search(query, max_results=5):
    """Perform a search using DuckDuckGo."""
    try:
        # Clean the query for URL
        query = query.replace(" ", "+")

        # Make the request to DuckDuckGo
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Extract search results using regex
        results = []
        pattern = r'<a class="result__a" href="(.*?)".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>'
        matches = re.findall(pattern, response.text, re.DOTALL)

        for i, match in enumerate(matches):
            if i >= max_results:
                break

            # Extract the raw URL and decode it
            raw_url = match[0]
            if "uddg=" in raw_url:
                # DuckDuckGo redirects with this parameter
                url_start = raw_url.find("uddg=") + 5
                decoded_url = raw_url[url_start:]
                if "&" in decoded_url:
                    decoded_url = decoded_url.split("&")[0]
                # URL decode
                try:
                    from urllib.parse import unquote

                    url = unquote(decoded_url)
                except:
                    url = decoded_url
            else:
                url = raw_url

            # Make sure URL has a protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url.lstrip("/")

            # Clean HTML from title and snippet
            title = html.unescape(re.sub(r"<.*?>", "", match[1]))
            snippet = html.unescape(re.sub(r"<.*?>", "", match[2]))

            results.append({"title": title, "url": url, "snippet": snippet})

        return results
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return []


# Define a function to get more info about a legal topic
def get_legal_info(topic):
    """Get information about a legal topic using DuckDuckGo search."""
    search_query = f"legal precedent {topic} law contract"
    results = duckduckgo_search(search_query)

    # Format the results
    formatted_results = []
    for result in results:
        formatted_results.append(
            {
                "title": result["title"],
                "description": result["snippet"],
                "url": result["url"],
            }
        )

    return formatted_results


# Replace Composio toolset and agent with direct implementation
# Remove this section:
# Initialize Composio toolset and Groq LLM
# toolset = ComposioToolSet()
# tools = toolset.get_tools(apps=[App.TAVILY])  # Add Tavily search tool

# Initialize Groq LLM
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define system message for the agent
system_message = """You are an expert legal contract analyzer with deep expertise in contract law and fairness assessment.
You have access to search capabilities that can help you find relevant legal information, precedents, and industry standards.
Use this information to enrich your analysis with:
1. Relevant legal precedents and case law
2. Industry standard practices and benchmarks
3. Recent regulatory changes or guidelines
4. Similar contract examples and their outcomes

For each clause you analyze:
1. Consider relevant legal precedents
2. Compare against industry standards
3. Check for recent regulatory changes
4. Identify potential compliance issues

Your analysis should be thorough, well-researched, and supported by concrete examples and references."""


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
                response=json.dumps(
                    {
                        "contractName": f"Contract {request.doc_id}",
                        "description": "No relevant information found in the contract.",
                        "metrics": {
                            "overallFairnessScore": 50,
                            "potentialBiasIndicators": 0,
                            "highRiskClauses": 0,
                            "balancedClauses": 0,
                        },
                        "sentimentDistribution": {
                            "vendorFavorable": 25,
                            "balanced": 25,
                            "customerFavorable": 25,
                            "neutral": 25,
                        },
                        "notableClauses": [],
                        "summary": {
                            "title": "Analysis Summary",
                            "description": "No relevant contract sections found for analysis.",
                            "points": [],
                            "riskAssessment": {
                                "level": "neutral",
                                "label": "Unknown",
                                "description": "Insufficient information for risk assessment.",
                            },
                        },
                        "legalReferences": [],
                        "industryStandards": [],
                        "regulatoryGuidelines": [],
                    }
                ),
                doc_id=request.doc_id,
            )

        # Extract the content from the most relevant results
        contexts = [result["content"] for result in results]

        # Get enrichment information using DuckDuckGo search
        contract_topic = " ".join(
            contexts[:100]
        )  # Use a portion of the contract to identify topic
        legal_info = get_legal_info(contract_topic)

        # Use Groq to analyze the contract with enriched information
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"""Analyze the following contract sections with focus on legal precedents, industry standards, and regulatory compliance:

                    Contract sections:
                    {' '.join(contexts)}
                    
                    Additional legal information:
                    {json.dumps(legal_info, indent=2)}

                    Provide a comprehensive analysis including:
                    1. Fairness and bias assessment
                    2. Relevant legal precedents
                    3. Industry standard comparisons
                    4. Regulatory compliance issues
                    5. Risk assessment
                    
                    Return the analysis in this exact JSON structure:
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
                                "recommendations": [string],
                                "legalPrecedents": [
                                    {{
                                        "case": string,
                                        "relevance": string,
                                        "implication": string
                                    }}
                                ]
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
                        }},
                        "legalReferences": [
                            {{
                                "title": string,
                                "description": string,
                                "url": string
                            }}
                        ],
                        "industryStandards": [
                            {{
                                "name": string,
                                "description": string,
                                "complianceStatus": string
                            }}
                        ],
                        "regulatoryGuidelines": [
                            {{
                                "regulation": string,
                                "relevance": string,
                                "complianceStatus": string
                            }}
                        ]
                    }}""",
                    },
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=4000,
            )

            response_text = response.choices[0].message.content

            # Clean up response text to ensure it's valid JSON
            # Remove markdown code blocks if present
            response_text = re.sub(r"^```json\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)

            # Try to parse and re-serialize to ensure valid JSON
            try:
                # Parse to check if it's valid JSON
                json_object = json.loads(response_text)
                # Re-serialize to ensure clean JSON
                response_text = json.dumps(json_object)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from Groq: {str(e)}")
                # Return a default response if JSON is invalid
                response_text = json.dumps(
                    {
                        "contractName": f"Contract {request.doc_id}",
                        "description": "Error processing contract analysis.",
                        "metrics": {
                            "overallFairnessScore": 50,
                            "potentialBiasIndicators": 0,
                            "highRiskClauses": 0,
                            "balancedClauses": 0,
                        },
                        "sentimentDistribution": {
                            "vendorFavorable": 25,
                            "balanced": 25,
                            "customerFavorable": 25,
                            "neutral": 25,
                        },
                        "notableClauses": [],
                        "summary": {
                            "title": "Analysis Error",
                            "description": "An error occurred while analyzing the contract.",
                            "points": [],
                            "riskAssessment": {
                                "level": "neutral",
                                "label": "Unknown",
                                "description": "Analysis failed due to technical issues.",
                            },
                        },
                        "legalReferences": [],
                        "industryStandards": [],
                        "regulatoryGuidelines": [],
                    }
                )

        except Exception as e:
            logger.error(f"Error getting response from Groq: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to get response from Groq LLM"
            )

        return QueryResponse(response=response_text, doc_id=request.doc_id)

    except Exception as e:
        logger.error(f"Error in query_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
