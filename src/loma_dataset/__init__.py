"""
LOMA Dataset - Medical Q&A Vector Database
==========================================

A high-quality Python package for managing medical Q&A data with vector search capabilities.

This package provides:
- Medical document and Q&A data models
- Vector database operations with SQLite/libSQL
- MIRIAD dataset processing utilities
- Search and retrieval functionality

Example usage:
    >>> from loma_dataset import MedicalVectorDB, MedicalDocument, MedicalQA
    >>> db = MedicalVectorDB("medical_qa.db")
    >>> db.initialize()
"""

from .models import MedicalDocument, MedicalQA, MedicalSearchResult, DocumentSearchResult
from .database import MedicalVectorDB
from .processor import MiriadProcessor, ProcessingConfig
from .exceptions import LomaDatasetError, DatabaseError, ProcessingError

__version__ = "0.1.0"
__author__ = "LOMA Dataset Team"

__all__ = [
    "MedicalDocument",
    "MedicalQA", 
    "MedicalSearchResult",
    "DocumentSearchResult",
    "MedicalVectorDB",
    "MiriadProcessor",
    "ProcessingConfig",
    "LomaDatasetError",
    "DatabaseError",
    "ProcessingError",
]