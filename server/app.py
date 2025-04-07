from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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
from google import genai
import json
import requests
import html
import re
from duckduckgo_search import DDGS
from google.genai.types import (
    GenerationConfig,
    Tool,
    GenerateContentConfig,
    GoogleSearch,
)
import enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Define enums for contract analysis
class SentimentLevel(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RiskLevel(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEUTRAL = "neutral"


# Define Pydantic models for structured output
class BiasIndicator(BaseModel):
    label: str
    value: float


class LegalPrecedent(BaseModel):
    case: str
    relevance: str
    implication: str


class NotableClause(BaseModel):
    type: str
    sentiment: str
    sentimentLabel: SentimentLevel
    biasScore: float
    riskLevel: str
    riskLabel: RiskLevel
    text: str
    analysis: str
    biasIndicators: List[BiasIndicator]
    industryComparison: str
    recommendations: List[str]
    legalPrecedents: List[LegalPrecedent]


class SummaryPoint(BaseModel):
    title: str
    description: str


class RiskAssessment(BaseModel):
    level: str
    label: RiskLevel
    description: str


class Summary(BaseModel):
    title: str
    description: str
    points: List[SummaryPoint]
    riskAssessment: RiskAssessment


class IndustryStandard(BaseModel):
    name: str
    description: str
    complianceStatus: str


class RegulatoryGuideline(BaseModel):
    regulation: str
    relevance: str
    complianceStatus: str


class Metrics(BaseModel):
    overallFairnessScore: float
    potentialBiasIndicators: int
    highRiskClauses: int
    balancedClauses: int


class SentimentDistribution(BaseModel):
    vendorFavorable: float
    balanced: float
    customerFavorable: float
    neutral: float


class LegalReference(BaseModel):
    """Legal reference from DuckDuckGo search."""

    title: str
    description: str
    url: str
    source: str
    type: str


class ContractAnalysis(BaseModel):
    contractName: str
    description: str
    metrics: Metrics
    sentimentDistribution: SentimentDistribution
    notableClauses: List[NotableClause]
    summary: Summary
    industryStandards: List[IndustryStandard]
    regulatoryGuidelines: List[RegulatoryGuideline]
    legalReferences: List[LegalReference]


# Initialize Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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


def generate_search_keywords(contract_text):
    """Use Gemini to generate relevant search keywords from contract text."""
    try:
        # Truncate contract text if too long
        max_length = 5000
        if len(contract_text) > max_length:
            contract_text = contract_text[:max_length] + "..."

        prompt = """Based on this contract text, generate 5 specific legal search queries.
        Each query should focus on finding relevant legal precedents, case law, or legal analysis.
        
        For example, if the contract mentions software licensing, generate queries like:
        - "software license agreement legal precedents"
        - "intellectual property licensing case law"
        - "software contract breach cases"
        
        Contract text:
        {text}
        
        Generate 5 specific search queries, focusing on the key legal concepts in this contract.
        Format each query as a plain text string, one per line.
        """.format(
            text=contract_text
        )

        # Get keywords from Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt
        )

        # Split response into lines and clean up
        search_terms = []
        for line in response.text.split("\n"):
            line = line.strip()
            # Remove bullet points, quotes, and other formatting
            line = line.lstrip("•-*\"' ").rstrip("\"' ")
            if line and not line.startswith(("Format", "Generate", "Based on")):
                search_terms.append(line + " legal precedent case law")

        logger.info(f"Generated search terms: {search_terms}")
        return search_terms

    except Exception as e:
        logger.error(f"Error generating search keywords: {str(e)}")
        return ["contract law precedents", "legal contract analysis"]


def get_legal_info(topic):
    """Get information about a legal topic using Gemini's built-in Google Search."""
    try:
        # Configure Google Search as a tool exactly as in the docs
        google_search_tool = Tool(google_search=GoogleSearch())

        # Create a simple, focused prompt that encourages search use
        prompt = f"""Find relevant legal precedents, case law, and analysis for this contract text.
        
        Contract text:
        {topic}
        
        Search for and provide:
        1. Recent court cases and legal precedents
        2. Legal analysis articles
        3. Industry standards
        4. Similar contract disputes
        
        For each source, include:
        - The title
        - Why it's relevant
        - Where it's from"""

        # Generate content with search capability using the exact configuration from docs
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Make sure to use the correct model
            contents=prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            ),
        )

        # Check if response is valid
        if not response or not response.candidates or not response.candidates[0]:
            logger.warning("Invalid response from Gemini")
            return get_fallback_references()

        # Get the response text
        candidate = response.candidates[0]
        if not candidate.content:
            logger.warning("No content in response")
            return get_fallback_references()

        # Log the raw response for debugging
        logger.info(f"Raw response: {candidate}")

        # Extract grounding metadata
        grounding_metadata = candidate.grounding_metadata
        if not grounding_metadata:
            logger.warning("No grounding metadata in response")
            return get_fallback_references()

        # Format the results using the grounding chunks
        formatted_results = []
        seen_urls = set()

        if grounding_metadata.grounding_chunks:
            logger.info(
                f"Found {len(grounding_metadata.grounding_chunks)} grounding chunks"
            )

            for chunk in grounding_metadata.grounding_chunks:
                if not chunk.web:
                    continue

                url = chunk.web.uri if hasattr(chunk.web, "uri") else None
                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                # Get the grounding supports for this chunk
                description = ""

                # Try to get description from grounding supports first
                if grounding_metadata.grounding_supports:
                    relevant_texts = []
                    for support in grounding_metadata.grounding_supports:
                        if hasattr(support, "groundingChunkIndices"):
                            chunk_index = grounding_metadata.grounding_chunks.index(
                                chunk
                            )
                            if (
                                chunk_index in support.groundingChunkIndices
                                and support.segment.text
                            ):
                                relevant_texts.append(support.segment.text)
                    if relevant_texts:
                        description = " ".join(relevant_texts)

                # Fallback to other sources if no grounding support
                if not description:
                    if hasattr(chunk.web, "snippet"):
                        description = chunk.web.snippet
                    elif hasattr(chunk, "segment") and hasattr(chunk.segment, "text"):
                        description = chunk.segment.text
                    else:
                        description = "Legal analysis and precedents"

                formatted_results.append(
                    {
                        "title": (
                            chunk.web.title
                            if hasattr(chunk.web, "title")
                            else "Legal Reference"
                        ),
                        "description": description,
                        "url": url,
                        "source": "Legal Database Search",
                        "type": "legal_reference",
                    }
                )

        # Log search queries used
        if hasattr(grounding_metadata, "webSearchQueries"):
            logger.info(f"Search queries used: {grounding_metadata.webSearchQueries}")

        # Add fallback references if we have too few results
        if len(formatted_results) < 3:
            logger.info("Adding fallback references")
            formatted_results.extend(get_fallback_references())

        logger.info(f"Found {len(formatted_results)} legal references")
        return formatted_results

    except Exception as e:
        logger.error(f"Error in legal information search: {str(e)}")
        return get_fallback_references()


def get_fallback_references():
    """Return fallback legal references when search fails."""
    return [
        {
            "title": "Contract Law Fundamentals",
            "description": "Overview of basic contract law principles and precedents",
            "url": "https://www.law.cornell.edu/wex/contract",
            "source": "Legal Information Institute",
            "type": "legal_reference",
        },
        {
            "title": "Commercial Contract Case Law",
            "description": "Collection of significant contract law cases and their implications",
            "url": "https://www.law.cornell.edu/wex/commercial_law",
            "source": "Legal Information Institute",
            "type": "legal_reference",
        },
        {
            "title": "Contract Interpretation Principles",
            "description": "Legal principles and precedents for interpreting contract terms",
            "url": "https://www.law.cornell.edu/wex/contract_interpretation",
            "source": "Legal Information Institute",
            "type": "legal_reference",
        },
    ]


def extract_legal_key_terms(text, max_terms=3):
    """Extract key legal terms from text to improve search quality."""
    # Common legal contract terms to look for
    legal_terms = [
        "indemnification",
        "liability",
        "warranty",
        "termination",
        "confidentiality",
        "intellectual property",
        "payment terms",
        "force majeure",
        "arbitration",
        "jurisdiction",
        "breach",
        "damages",
        "performance",
        "compliance",
        "default",
        "remedies",
        "assignment",
        "severability",
        "waiver",
        "governing law",
    ]

    # Lowercase the text for case-insensitive matching
    text_lower = text.lower()

    # Find terms present in the text
    found_terms = []
    for term in legal_terms:
        if term.lower() in text_lower:
            found_terms.append(term)
            if len(found_terms) >= max_terms:
                break

    # If no key terms found, return a general term
    if not found_terms:
        return "contract provisions"

    return " ".join(found_terms)


# Initialize Groq LLM
llm = client

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


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    # Ensure x is a numpy array and handle potential numerical instability
    x = np.array(x, dtype=np.float64)
    # Subtract the maximum value for numerical stability
    x = x - np.max(x)
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()) * 100  # Multiply by 100 to get percentages


def normalize_sentiment_distribution(sentiment_dist):
    """Normalize sentiment distribution using softmax."""
    # Extract values in a fixed order
    values = [
        sentiment_dist.get("vendorFavorable", 0.35),
        sentiment_dist.get("balanced", 0.4),
        sentiment_dist.get("customerFavorable", 0.15),
        sentiment_dist.get("neutral", 0.1),
    ]

    # Apply softmax normalization
    normalized = softmax(values)

    # Return normalized values in a dictionary
    return {
        "vendorFavorable": round(float(normalized[0]), 2),
        "balanced": round(float(normalized[1]), 2),
        "customerFavorable": round(float(normalized[2]), 2),
        "neutral": round(float(normalized[3]), 2),
    }


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
            default_dist = normalize_sentiment_distribution(
                {
                    "vendorFavorable": 0.35,
                    "balanced": 0.4,
                    "customerFavorable": 0.15,
                    "neutral": 0.1,
                }
            )
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
                        "sentimentDistribution": default_dist,
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

        # Use Groq to analyze the contract
        try:
            # Get contract topic for relevant search
            contract_topic = " ".join(contexts[:100])

            # Extract key legal terms from the contract
            legal_terms = extract_legal_key_terms(contract_topic)
            search_query = f"legal contract {legal_terms} clause precedent"

            # Get legal references first
            logger.info(f"Getting legal references for contract topic")
            legal_references = get_legal_info(contract_topic)

            # Prepare the prompt for structured output
            prompt = f"""Analyze the following contract sections and provide a structured analysis in JSON format.
            Focus on contract terms, fairness, and compliance.

            Contract sections to analyze:
            {' '.join(contexts)}

            Please provide a comprehensive analysis including:
            1. Overall fairness assessment
            2. Sentiment analysis of clauses
            3. Risk assessment
            4. Notable clauses and their implications
            5. Industry standards comparison
            6. Regulatory compliance evaluation
            7. Bias indicators

            Note: Legal references will be handled separately, focus on analyzing the contract content.
            Format the response as a structured JSON object following the specified schema.
            """

            # Configure the response format
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ContractAnalysis,
                },
            )

            # The response will be in JSON format
            try:
                # Parse response to JSON
                analysis_json = json.loads(response.text)

                # Normalize sentiment distribution
                if "sentimentDistribution" in analysis_json:
                    analysis_json["sentimentDistribution"] = (
                        normalize_sentiment_distribution(
                            analysis_json["sentimentDistribution"]
                        )
                    )
                else:
                    # Use default distribution if none provided
                    analysis_json["sentimentDistribution"] = (
                        normalize_sentiment_distribution(
                            {
                                "vendorFavorable": 0.35,
                                "balanced": 0.4,
                                "customerFavorable": 0.15,
                                "neutral": 0.1,
                            }
                        )
                    )

                # Format legal references with proper source and type
                formatted_legal_refs = []
                for ref in legal_references:
                    formatted_legal_refs.append(
                        {
                            "title": ref.get("title", ""),
                            "description": ref.get("description", ""),
                            "url": ref.get("url", ""),
                            "source": ref.get("source", "Legal Database Search"),
                            "type": ref.get("type", "legal_reference"),
                        }
                    )

                # Add the legal references from DuckDuckGo search
                analysis_json["legalReferences"] = formatted_legal_refs

                # Re-encode as JSON string
                response_text = json.dumps(analysis_json)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from Gemini: {str(e)}")
                # Return a default response if JSON is invalid
                default_dist = normalize_sentiment_distribution(
                    {
                        "vendorFavorable": 0.35,
                        "balanced": 0.4,
                        "customerFavorable": 0.15,
                        "neutral": 0.1,
                    }
                )
                response_text = json.dumps(
                    {
                        "contractName": f"Contract {request.doc_id}",
                        "description": "Error processing contract analysis.",
                        "metrics": {
                            "overallFairnessScore": 50.0,
                            "potentialBiasIndicators": 0,
                            "highRiskClauses": 0,
                            "balancedClauses": 0,
                        },
                        "sentimentDistribution": default_dist,
                        "notableClauses": [],
                        "summary": {
                            "title": "Analysis Error",
                            "description": "An error occurred while analyzing the contract.",
                            "points": [],
                            "riskAssessment": {
                                "level": "neutral",
                                "label": RiskLevel.NEUTRAL,
                                "description": "Analysis failed due to technical issues.",
                            },
                        },
                        "industryStandards": [],
                        "regulatoryGuidelines": [],
                        "legalReferences": (
                            formatted_legal_refs
                            if "formatted_legal_refs" in locals()
                            else []
                        ),
                    }
                )

        except Exception as e:
            logger.error(f"Error getting response from Gemini: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to get response from Gemini LLM"
            )

        return QueryResponse(response=response_text, doc_id=request.doc_id)

    except Exception as e:
        logger.error(f"Error in query_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
