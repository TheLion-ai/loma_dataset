"""
MIRIAD Dataset Processing Module
===============================

This module provides functionality to process the MIRIAD medical Q&A dataset
and convert it to the database format using medical text embeddings.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Iterator
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Import dataset and model libraries
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(f"Missing required dependencies: {e}. "
                     "Please install: pip install datasets transformers huggingface_hub onnxruntime tqdm")

# Optional sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .models import MedicalDocument, MedicalQA, ProcessingConfig
from .database import MedicalVectorDB
from .exceptions import ProcessingError, ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATASET_NAME = "miriad/miriad-5.8M"
DEFAULT_SPLIT = "train"
EMBEDDING_DIMENSION = 384


class MedicalEmbeddingGenerator:
    """
    Generates medical text embeddings using ONNX model or sentence-transformers.
    
    This class handles the loading and inference of medical text embedding models
    optimized for medical domain text processing. Supports both ONNX models for
    faster inference and sentence-transformers for easier model usage.
    """
    
    def __init__(self, model_name: str, use_quantized: bool = True, cache_dir: Optional[str] = None, 
                 model_type: str = "auto"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the Hugging Face model
            use_quantized: Whether to use quantized model for faster inference (ONNX only)
            cache_dir: Directory to cache downloaded models
            model_type: Type of model to use ("onnx", "sentence_transformers", or "auto")
                       "auto" will try sentence-transformers first, then fall back to ONNX
        """
        self.model_name = model_name
        self.use_quantized = use_quantized
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.tokenizer = None
        self.session = None
        self.sentence_model = None
        self.max_length = 512
        self._actual_model_type = None
        
    def initialize(self) -> None:
        """Initialize the embedding model (sentence-transformers or ONNX)."""
        try:
            if self.model_type == "sentence_transformers":
                self._initialize_sentence_transformers()
            elif self.model_type == "onnx":
                self._initialize_onnx()
            elif self.model_type == "auto":
                # Try sentence-transformers first, fall back to ONNX
                try:
                    self._initialize_sentence_transformers()
                except Exception as e:
                    logger.info(f"Sentence-transformers initialization failed, falling back to ONNX: {e}")
                    self._initialize_onnx()
            else:
                raise ConfigurationError(f"Unknown model_type: {self.model_type}. "
                                       "Use 'onnx', 'sentence_transformers', or 'auto'")
            
            logger.info(f"Medical embedding generator initialized successfully using {self._actual_model_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}")
            raise ProcessingError(f"Failed to initialize embedding generator: {e}")
    
    def _initialize_sentence_transformers(self) -> None:
        """Initialize using sentence-transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ProcessingError("sentence-transformers not available. "
                                "Install with: pip install sentence-transformers")
        
        logger.info(f"Loading sentence-transformers model: {self.model_name}")
        self.sentence_model = SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_dir
        )
        self._actual_model_type = "sentence_transformers"
    
    def _initialize_onnx(self) -> None:
        """Initialize using ONNX model."""
        logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Download and load ONNX model
        model_filename = "model_quantized.onnx" if self.use_quantized else "model.onnx"
        model_path = hf_hub_download(
            repo_id=self.model_name,
            filename=model_filename,
            cache_dir=self.cache_dir
        )
        
        logger.info(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        self._actual_model_type = "onnx"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProcessingError: If embedding generation fails
        """
        if self._actual_model_type is None:
            raise ProcessingError("Embedding generator not initialized")
        
        try:
            if self._actual_model_type == "sentence_transformers":
                return self._generate_embeddings_sentence_transformers(texts)
            elif self._actual_model_type == "onnx":
                return self._generate_embeddings_onnx(texts)
            else:
                raise ProcessingError(f"Unknown model type: {self._actual_model_type}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ProcessingError(f"Failed to generate embeddings: {e}")
    
    def _generate_embeddings_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        if not self.sentence_model:
            raise ProcessingError("Sentence-transformers model not initialized")
        
        # Generate embeddings - sentence-transformers handles normalization by default
        embeddings = self.sentence_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # Ensure L2 normalization
        )
        
        return embeddings.tolist()
    
    def _generate_embeddings_onnx(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using ONNX model."""
        if not self.tokenizer or not self.session:
            raise ProcessingError("ONNX model not initialized")
        
        # Tokenize the batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        
        # Run inference
        outputs = self.session.run(None, dict(inputs))
        embeddings = outputs[0]
        
        # Apply mean pooling if needed
        if len(embeddings.shape) == 3:  # [batch, seq_len, hidden_dim]
            attention_mask = inputs.get('attention_mask', np.ones(embeddings.shape[:2]))
            attention_mask_expanded = np.expand_dims(attention_mask, -1)
            embeddings = np.sum(embeddings * attention_mask_expanded, axis=1) / np.sum(attention_mask_expanded, axis=1)
        
        # L2 normalize embeddings to unit vectors for optimal cosine similarity
        # This ensures all vectors have magnitude 1, making cosine similarity 
        # equivalent to dot product and improving search consistency
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)  # Add epsilon to prevent division by zero
        
        return embeddings.tolist()


class MiriadProcessor:
    """
    Processes MIRIAD dataset and populates the medical database.
    
    This class handles the complete pipeline from dataset loading to
    database population with vector embeddings.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.db = MedicalVectorDB(config.db_path)
        self.embedding_generator = MedicalEmbeddingGenerator(
            config.model_name,
            config.use_quantized,
            config.cache_dir,
            config.model_type
        )
        self.processed_count = 0
        self.error_count = 0
        
    def initialize(self) -> None:
        """Initialize database and embedding generator."""
        logger.info("Initializing MIRIAD processor...")
        
        try:
            # Initialize database
            self.db.initialize()
            
            # Initialize embedding generator
            self.embedding_generator.initialize()
            
            logger.info("MIRIAD processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            raise ProcessingError(f"Failed to initialize processor: {e}")
    
    def load_dataset(self) -> Any:
        """
        Load the MIRIAD dataset from Hugging Face.
        
        Returns:
            Loaded dataset
            
        Raises:
            ProcessingError: If dataset loading fails
        """
        logger.info(f"Loading dataset {DEFAULT_DATASET_NAME}...")
        
        try:
            dataset = load_dataset(
                DEFAULT_DATASET_NAME,
                split=DEFAULT_SPLIT
            )
            
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
            
            logger.info(f"Loaded {len(dataset)} samples from dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise ProcessingError(f"Failed to load dataset: {e}")
    
    def process_dataset(self) -> None:
        """Process the complete dataset."""
        try:
            dataset = self.load_dataset()
            
            # Process in batches
            total_batches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
            
            with tqdm(total=len(dataset), desc="Processing samples") as pbar:
                for i in range(0, len(dataset), self.config.batch_size):
                    batch = dataset[i:i + self.config.batch_size]
                    self._process_batch(batch)
                    pbar.update(len(batch) if isinstance(batch, list) else 1)
            
            logger.info(f"Processing completed. Processed: {self.processed_count}, Errors: {self.error_count}")
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise ProcessingError(f"Dataset processing failed: {e}")
    
    def populate_database(self, db: 'MedicalVectorDB', dataset: Any) -> None:
        """
        Populate database with processed dataset.
        
        Args:
            db: Database instance to populate
            dataset: Dataset to process
        """
        # Set the database instance
        self.db = db
        
        # Process in batches
        with tqdm(total=len(dataset), desc="Processing samples") as pbar:
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[i:i + self.config.batch_size]
                self._process_batch(batch)
                # Update progress bar with actual number of samples processed
                batch_size = min(self.config.batch_size, len(dataset) - i)
                pbar.update(batch_size)
        
        logger.info(f"Processing completed. Processed: {self.processed_count}, Errors: {self.error_count}")
    
    def _process_batch(self, batch: Any) -> None:
        """Process a batch of samples."""
        try:
            # Handle HuggingFace dataset batch format (dict with lists) vs individual samples
            if isinstance(batch, dict) and isinstance(batch.get('question'), list):
                # This is a HuggingFace batch format: {'question': [q1, q2, ...], 'answer': [a1, a2, ...], ...}
                batch_size = len(batch['question'])
                samples = []
                for i in range(batch_size):
                    sample = {key: values[i] for key, values in batch.items()}
                    samples.append(sample)
                batch = samples
            elif not isinstance(batch, list):
                batch = [batch]
            
            # Extract Q&A data
            qa_data = []
            doc_data = []
            
            for sample in batch:
                try:
                    qa_entry, doc_entry = self._extract_qa_data(sample)
                    if qa_entry and doc_entry:
                        qa_data.append(qa_entry)
                        doc_data.append(doc_entry)
                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    self.error_count += 1
                    continue
            
            if not qa_data:
                return
            
            # Generate embeddings
            qa_texts = [qa['question'] for qa in qa_data]  # Use only the question for Q&A embeddings
            doc_texts = [doc['content'] for doc in doc_data]
            
            qa_embeddings = self.embedding_generator.generate_embeddings(qa_texts)
            doc_embeddings = self.embedding_generator.generate_embeddings(doc_texts)
            
            # Store in database
            # First, collect unique documents to avoid duplicates
            unique_docs = {}
            doc_vectors_map = {}
            
            for i, (qa_entry, doc_entry) in enumerate(zip(qa_data, doc_data)):
                doc_id = doc_entry['id']
                if doc_id not in unique_docs:
                    # Ensure vectors are lists of floats
                    doc_vector = doc_embeddings[i]
                    if not isinstance(doc_vector, list):
                        doc_vector = doc_vector.tolist() if hasattr(doc_vector, 'tolist') else list(doc_vector)
                    
                    unique_docs[doc_id] = doc_entry
                    doc_vectors_map[doc_id] = doc_vector
            
            # Add unique documents first
            for doc_id, doc_entry in unique_docs.items():
                try:
                    document = MedicalDocument(
                        id=doc_entry['id'],
                        title=doc_entry['title'],
                        content=doc_entry['content'],
                        vector=doc_vectors_map[doc_id],
                        created_at=datetime.now(),
                        url=doc_entry.get('url'),
                        specialty=doc_entry.get('specialty'),
                        year=doc_entry.get('year')
                    )
                    self.db.add_document(document)
                except Exception as e:
                    logger.warning(f"Error storing document {doc_id}: {e}")
                    continue
            
            # Then add all Q&As
            for i, (qa_entry, doc_entry) in enumerate(zip(qa_data, doc_data)):
                try:
                    # Ensure vectors are lists of floats
                    qa_vector = qa_embeddings[i]
                    if not isinstance(qa_vector, list):
                        qa_vector = qa_vector.tolist() if hasattr(qa_vector, 'tolist') else list(qa_vector)
                    
                    # Create Q&A
                    qa = MedicalQA(
                        id=qa_entry['id'],
                        question=qa_entry['question'],
                        answer=qa_entry['answer'],
                        vector=qa_vector,
                        document_id=doc_entry['id']
                    )
                    
                    # Add Q&A to database
                    if not self.config.skip_existing or not self._entry_exists(qa.id):
                        self.db.add_qa(qa)
                        self.processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error storing Q&A entry: {e}")
                    self.error_count += 1
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.error_count += len(batch) if isinstance(batch, list) else 1
    
    def _extract_qa_data(self, sample: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Extract Q&A and document data from a sample."""
        try:
            # Helper function to safely get string values
            def safe_get_str(key: str, default: str = '') -> str:
                value = sample.get(key, default)
                if isinstance(value, list):
                    # This shouldn't happen anymore with the fixed batch processing,
                    # but handle it just in case
                    logger.warning(f"Unexpected list value for key '{key}': {value}")
                    if key == 'specialty' and value:
                        return str(value[0])
                    return str(value[0]) if value else default
                return str(value) if value is not None else default
            
            # Generate unique IDs
            question = safe_get_str('question')
            answer = safe_get_str('answer')
            url = safe_get_str('paper_url')
            paper_title = safe_get_str('paper_title', 'Unknown')
            
            qa_id = self._generate_id(question + answer)
            doc_id = self._generate_id(url + paper_title)
            
            qa_entry = {
                'id': qa_id,
                'question': question,
                'answer': answer
            }
            
            # Handle year field - convert to int if possible
            year_value = sample.get('year')
            if isinstance(year_value, list) and year_value:
                year_value = year_value[0]
            if year_value is not None:
                try:
                    year_value = int(year_value)
                except (ValueError, TypeError):
                    year_value = None
            
            doc_entry = {
                'id': doc_id,
                'title': paper_title,
                'content': safe_get_str('passage_text', answer),
                'url': url or None,
                'specialty': safe_get_str('specialty') or None,
                'year': year_value
            }
            
            return qa_entry, doc_entry
            
        except Exception as e:
            logger.warning(f"Error extracting data from sample: {e}")
            return None, None
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID from text."""
        # Handle cases where text might be a list or other non-string type
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _entry_exists(self, qa_id: str) -> bool:
        """Check if a Q&A entry already exists in the database."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT 1 FROM medical_qa WHERE id = ?", [qa_id])
            return cursor.fetchone() is not None
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "database_stats": self.db.get_stats()
        }
    
    def close(self) -> None:
        """Close the processor and database connection."""
        if self.db:
            self.db.close()