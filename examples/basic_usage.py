#!/usr/bin/env python3
"""
Basic usage example for the LOMA Dataset package.

This example demonstrates:
1. Creating a medical vector database
2. Adding documents and Q&A entries
3. Performing similarity searches
4. Retrieving database statistics
"""

import logging
from datetime import datetime
from pathlib import Path

from loma_dataset import MedicalVectorDB, MedicalDocument, MedicalQA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic usage example."""
    # Create database
    db_path = Path("example_medical.db")
    db = MedicalVectorDB(str(db_path))
    db.initialize()
    
    logger.info("Created medical vector database")
    
    # Create sample document
    document = MedicalDocument(
        id="doc_001",
        title="Understanding Hypertension",
        content="Hypertension is a common cardiovascular condition that affects millions of people worldwide. It is characterized by persistently elevated blood pressure levels.",
        vector=[0.1, 0.2, 0.3, 0.4, 0.5],  # Example 5-dimensional vector
        created_at=datetime.now(),
        url="https://example.com/hypertension-research",
        year=2024,
        specialty="Cardiology"
    )
    
    # Add document to database
    db.add_document(document)
    logger.info(f"Added document: {document.title}")
    
    # Create sample Q&A
    qa = MedicalQA(
        id="qa_001",
        question="What is hypertension?",
        answer="Hypertension is high blood pressure that can lead to serious health complications.",
        vector=[0.2, 0.3, 0.4, 0.5, 0.6],  # Example 5-dimensional vector
        document_id="doc_001"
    )
    
    # Add Q&A to database
    db.add_qa(qa)
    logger.info(f"Added Q&A: {qa.question}")
    
    # Perform similarity search
    query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
    results = db.search_similar_qa(query_vector, limit=5)
    
    logger.info(f"Found {len(results)} similar Q&A entries:")
    for result in results:
        logger.info(f"  - {result.qa.question} (similarity: {result.similarity:.3f})")
    
    # Get database statistics
    stats = db.get_stats()
    logger.info(f"Database statistics: {stats}")
    
    # Clean up
    if db_path.exists():
        db_path.unlink()
        logger.info("Cleaned up example database")


if __name__ == "__main__":
    main()