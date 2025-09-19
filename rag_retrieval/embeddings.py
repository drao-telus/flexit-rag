"""
Embedding Generation for Fuelix API
Provides text embedding functionality using the Fuelix embeddings endpoint.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "text-embedding-3-large"
    encoding_format: str = "float"
    dimensions: Optional[int] = None  # Use model default if None
    user: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embedding: List[float]
    model: str
    usage: Dict[str, int]
    dimensions: int
    input_text: str
    processing_time_ms: float


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding generation."""

    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]
    dimensions: int
    input_texts: List[str]
    processing_time_ms: float
    successful_count: int
    failed_count: int


class FuelixEmbeddingManager:
    """
    Manager class for generating embeddings using the Fuelix API.
    Handles single and batch embedding generation with error handling and retry logic.
    """

    def __init__(
        self, api_key: Optional[str] = None, base_url: str = "https://api.fuelix.ai/v1"
    ):
        """
        Initialize the Fuelix Embedding Manager.

        Args:
            api_key: API key for authentication
            base_url: Base URL for Fuelix API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings_endpoint = f"{base_url}/embeddings"

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate_embedding(
        self, text: str, config: Optional[EmbeddingConfig] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text input.

        Args:
            text: Input text to embed
            config: Optional embedding configuration

        Returns:
            EmbeddingResult with embedding vector and metadata

        Raises:
            requests.RequestException: If API request fails
        """
        if config is None:
            config = EmbeddingConfig()

        payload = {
            "input": text,
            "model": config.model,
            "encoding_format": config.encoding_format,
        }

        # Add optional parameters
        if config.dimensions is not None:
            payload["dimensions"] = config.dimensions
        if config.user is not None:
            payload["user"] = config.user

        logger.info(f"Generating embedding for text (length: {len(text)} chars)")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        start_time = time.time()

        try:
            response = requests.post(
                self.embeddings_endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Extract embedding data
            embedding_data = result["data"][0]
            embedding_vector = embedding_data["embedding"]

            logger.info(
                f"Embedding generated successfully. Dimensions: {len(embedding_vector)}"
            )

            return EmbeddingResult(
                embedding=embedding_vector,
                model=result["model"],
                usage=result["usage"],
                dimensions=len(embedding_vector),
                input_text=text,
                processing_time_ms=processing_time,
            )

        except requests.RequestException as e:
            logger.error(f"Failed to generate embedding: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise

    def generate_batch_embeddings(
        self,
        texts: List[str],
        config: Optional[EmbeddingConfig] = None,
        batch_size: int = 100,
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to embed
            config: Optional embedding configuration
            batch_size: Number of texts to process per batch

        Returns:
            BatchEmbeddingResult with all embeddings and metadata
        """
        if config is None:
            config = EmbeddingConfig()

        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        successful_count = 0
        failed_count = 0

        start_time = time.time()

        logger.info(
            f"Generating embeddings for {len(texts)} texts in batches of {batch_size}"
        )

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # For batch processing, we send multiple inputs
                payload = {
                    "input": batch_texts,
                    "model": config.model,
                    "encoding_format": config.encoding_format,
                }

                # Add optional parameters
                if config.dimensions is not None:
                    payload["dimensions"] = config.dimensions
                if config.user is not None:
                    payload["user"] = config.user

                logger.debug(
                    f"Processing batch {i // batch_size + 1}: {len(batch_texts)} texts"
                )

                response = requests.post(
                    self.embeddings_endpoint,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60,  # Longer timeout for batch requests
                )
                response.raise_for_status()

                result = response.json()

                # Extract embeddings from batch result
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)

                # Accumulate usage statistics
                if "usage" in result:
                    total_usage["prompt_tokens"] += result["usage"].get(
                        "prompt_tokens", 0
                    )
                    total_usage["total_tokens"] += result["usage"].get(
                        "total_tokens", 0
                    )

                successful_count += len(batch_texts)
                logger.info(
                    f"Batch {i // batch_size + 1} completed: {len(batch_texts)} embeddings generated"
                )

            except requests.RequestException as e:
                logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                failed_count += len(batch_texts)
                # Add empty embeddings for failed texts to maintain alignment
                all_embeddings.extend([[] for _ in batch_texts])

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get dimensions from first successful embedding
        dimensions = (
            len(all_embeddings[0]) if all_embeddings and all_embeddings[0] else 0
        )

        logger.info(
            f"Batch embedding completed: {successful_count} successful, {failed_count} failed"
        )

        return BatchEmbeddingResult(
            embeddings=all_embeddings,
            model=config.model,
            usage=total_usage,
            dimensions=dimensions,
            input_texts=texts,
            processing_time_ms=processing_time,
            successful_count=successful_count,
            failed_count=failed_count,
        )

    def generate_embedding_with_retry(
        self,
        text: str,
        config: Optional[EmbeddingConfig] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> EmbeddingResult:
        """
        Generate embedding with retry logic for robustness.

        Args:
            text: Input text to embed
            config: Optional embedding configuration
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            EmbeddingResult with embedding vector and metadata
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return self.generate_embedding(text, config)

            except requests.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Embedding attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries + 1} embedding attempts failed")

        raise last_exception

    def test_embedding_api(self) -> Dict[str, Any]:
        """
        Test the embedding API with a simple request.

        Returns:
            Test result with status and performance metrics
        """
        test_text = "The quick brown fox jumped over the lazy dog"

        try:
            logger.info("Testing embedding API...")
            result = self.generate_embedding(test_text)

            return {
                "status": "success",
                "model": result.model,
                "dimensions": result.dimensions,
                "processing_time_ms": result.processing_time_ms,
                "usage": result.usage,
                "test_text": test_text,
            }

        except Exception as e:
            logger.error(f"Embedding API test failed: {e}")
            return {"status": "failed", "error": str(e), "test_text": test_text}


def create_optimized_embedding_config(
    model: str = "text-embedding-3-large",
    dimensions: Optional[int] = None,
    user_id: Optional[str] = None,
) -> EmbeddingConfig:
    """
    Create an optimized embedding configuration.

    Args:
        model: Embedding model to use
        dimensions: Optional dimension reduction (for text-embedding-3-large)
        user_id: Optional user identifier for tracking

    Returns:
        Optimized EmbeddingConfig
    """
    return EmbeddingConfig(
        model=model, encoding_format="float", dimensions=dimensions, user=user_id
    )


def demonstrate_embedding_functionality(api_key: Optional[str] = None):
    """Demonstrate embedding functionality with examples."""
    print("Fuelix Embedding API Demonstration")
    print("=" * 50)

    manager = FuelixEmbeddingManager(api_key=api_key)

    # Test 1: Basic embedding generation
    print("\n1. Basic Embedding Generation")
    print("-" * 30)

    test_text = "The quick brown fox jumped over the lazy dog"
    try:
        result = manager.generate_embedding(test_text)
        print(f"✅ Text: {test_text}")
        print(f"✅ Model: {result.model}")
        print(f"✅ Dimensions: {result.dimensions}")
        print(f"✅ Processing time: {result.processing_time_ms:.2f}ms")
        print(f"✅ Usage: {result.usage}")
        print(f"✅ Embedding preview: {result.embedding[:5]}...")

    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 2: Custom configuration
    print("\n2. Custom Configuration")
    print("-" * 30)

    try:
        config = EmbeddingConfig(
            model="text-embedding-3-large",
            dimensions=512,  # Reduced dimensions
            user="demo-user",
        )

        result = manager.generate_embedding(test_text, config)
        print(f"✅ Custom dimensions: {result.dimensions}")
        print(f"✅ Model: {result.model}")
        print(f"✅ Processing time: {result.processing_time_ms:.2f}ms")

    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 3: Batch embedding (small batch for demo)
    print("\n3. Batch Embedding")
    print("-" * 30)

    try:
        texts = [
            "How to enroll in benefits?",
            "Annual enrollment process",
            "Change beneficiary information",
            "Employee payroll setup",
        ]

        batch_result = manager.generate_batch_embeddings(texts, batch_size=2)
        print(f"✅ Processed {len(texts)} texts")
        print(f"✅ Successful: {batch_result.successful_count}")
        print(f"✅ Failed: {batch_result.failed_count}")
        print(f"✅ Total processing time: {batch_result.processing_time_ms:.2f}ms")
        print(
            f"✅ Average per text: {batch_result.processing_time_ms / len(texts):.2f}ms"
        )
        print(f"✅ Dimensions: {batch_result.dimensions}")

    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 4: API test
    print("\n4. API Test")
    print("-" * 30)

    test_result = manager.test_embedding_api()
    if test_result["status"] == "success":
        print("✅ API test successful")
        print(f"✅ Model: {test_result['model']}")
        print(f"✅ Dimensions: {test_result['dimensions']}")
        print(f"✅ Response time: {test_result['processing_time_ms']:.2f}ms")
    else:
        print(f"❌ API test failed: {test_result['error']}")
