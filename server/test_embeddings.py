import logging
from llama_index.embeddings.ollama import OllamaEmbedding
import numpy as np
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding():
    """Test the Ollama embedding model with various inputs."""
    try:
        # Initialize embedding model
        logger.info("Initializing Ollama embedding model...")
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            request_timeout=30.0,
            embed_batch_size=1,
            additional_kwargs={
                "temperature": 0.0,
                "num_thread": 4,
            },
            dimensions=768,
        )

        # Test cases
        test_texts = [
            "This is a simple test sentence.",
            "A longer test sentence with more words to process and analyze.",
            "A very long test sentence " * 10,  # Test with longer text
            "",  # Test empty string
            "Special characters: !@#$%^&*()",  # Test special characters
            "Numbers: 1234567890",  # Test numbers
            "Mixed content: Text with numbers 123 and special chars !@#",  # Test mixed content
        ]

        logger.info("Starting embedding tests...")
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\nTest case {i}: {text[:50]}...")
            
            try:
                # Get embedding
                start_time = time.time()
                embedding = embed_model.get_text_embedding(text)
                end_time = time.time()
                
                # Log basic info
                logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
                logger.info(f"Embedding type: {type(embedding)}")
                logger.info(f"Embedding length: {len(embedding) if embedding else 'None'}")
                
                if embedding is None:
                    logger.error("Embedding is None!")
                    continue
                    
                # Convert to numpy array for analysis
                embedding_array = np.array([
                    float(x) if x != 'None' and x is not None else 0.0 
                    for x in embedding
                ])
                
                # Analyze embedding
                logger.info(f"Shape: {embedding_array.shape}")
                logger.info(f"Contains NaN: {np.any(np.isnan(embedding_array))}")
                logger.info(f"Contains Inf: {np.any(np.isinf(embedding_array))}")
                logger.info(f"Min value: {np.min(embedding_array)}")
                logger.info(f"Max value: {np.max(embedding_array)}")
                logger.info(f"Mean value: {np.mean(embedding_array)}")
                logger.info(f"Number of zeros: {np.sum(embedding_array == 0)}")
                logger.info(f"Number of None/nan values: {np.sum(np.isnan(embedding_array))}")
                
                # Print first few values
                logger.info(f"First 5 values: {embedding[:5]}")
                
            except Exception as e:
                logger.error(f"Error processing test case {i}: {str(e)}")

    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting embedding test script...")
    test_embedding()
    logger.info("Embedding test script completed.") 