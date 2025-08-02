"""
Data models for the LOMA Dataset package.

This module defines the core data structures used throughout the package:
- MedicalDocument: Represents medical research papers and documents
- MedicalQA: Represents medical Q&A entries
- Search result classes for different types of queries
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class MedicalDocument:
    """
    Represents a medical research document/paper.
    
    This class stores metadata and content for medical documents,
    including vector embeddings for similarity search.
    
    Attributes:
        id: Unique identifier for the document
        title: Title of the document
        content: Full content of the document
        vector: Vector embedding of the content
        created_at: Timestamp when record was created
        year: Publication year (optional)
        specialty: Medical specialty (e.g., "Cardiology", "Oncology") (optional)
    """
    
    id: str
    title: str
    content: str
    vector: List[float]
    created_at: datetime
    year: Optional[int] = None
    specialty: Optional[str] = None

    def __post_init__(self):
        """Validate the document data after initialization."""
        from .exceptions import ValidationError
        
        if not self.id:
            raise ValidationError("Document ID cannot be empty")
        if not self.title:
            raise ValidationError("Document title cannot be empty")
        if not self.content:
            raise ValidationError("Document content cannot be empty")
        if not self.vector:
            raise ValidationError("Vector cannot be empty")


@dataclass
class MedicalQA:
    """
    Represents a medical Q&A entry with vector embedding.
    
    This class stores medical questions and answers along with
    their vector embeddings for similarity search.
    
    Attributes:
        id: Unique identifier for the Q&A entry
        question: Medical question
        answer: Corresponding answer
        vector: Vector embedding of the question
        document_id: Reference to source document (required)
    """
    
    id: str
    question: str
    answer: str
    vector: List[float]
    document_id: str

    def __post_init__(self):
        """Validate the Q&A data after initialization."""
        from .exceptions import ValidationError
        
        if not self.id:
            raise ValidationError("Q&A ID cannot be empty")
        if not self.question:
            raise ValidationError("Question cannot be empty")
        if not self.answer:
            raise ValidationError("Answer cannot be empty")
        if not self.vector:
            raise ValidationError("Vector cannot be empty")
        if not self.document_id:
            raise ValidationError("Document ID cannot be empty")


@dataclass
class MedicalSearchResult:
    """
    Result of a medical Q&A similarity search.
    
    Attributes:
        qa: The matching Q&A entry
        similarity: Similarity score (0.0 to 1.0)
        document: Associated document (optional)
    """
    
    qa: MedicalQA
    similarity: float
    document: Optional[MedicalDocument] = None

    def __post_init__(self):
        """Validate the search result data."""
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class DocumentSearchResult:
    """
    Result of a document similarity search.
    
    Attributes:
        document: The matching document
        similarity: Similarity score (0.0 to 1.0)
    """
    
    document: MedicalDocument
    similarity: float

    def __post_init__(self):
        """Validate the search result data."""
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class ProcessingConfig:
    """
    Configuration for MIRIAD dataset processing.
    
    Attributes:
        model_name: Name of the embedding model to use
        batch_size: Batch size for processing embeddings
        max_length: Maximum sequence length for tokenization
        device: Device to use for inference ('cpu' or 'cuda')
        cache_dir: Directory to cache models and data
        db_path: Path to the database file
        use_quantized: Whether to use quantized models for inference
        max_samples: Maximum number of samples to process (None for all)
        skip_existing: Whether to skip existing entries in the database
    """
    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"
    cache_dir: str = "./cache"
    db_path: str = "medical.db"
    use_quantized: bool = False
    max_samples: Optional[int] = None
    skip_existing: bool = True

    def __post_init__(self):
        """Validate the configuration."""
        from .exceptions import ValidationError
        
        if not self.model_name:
            raise ValidationError("Model name cannot be empty")
        if self.batch_size <= 0:
            raise ValidationError("Batch size must be positive")
        if self.max_length <= 0:
            raise ValidationError("Max length must be positive")
        if self.device not in ["cpu", "cuda"]:
            raise ValidationError("Device must be 'cpu' or 'cuda'")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValidationError("Max samples must be positive")