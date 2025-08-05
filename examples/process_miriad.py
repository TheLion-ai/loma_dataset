#!/usr/bin/env python3
"""
MIRIAD dataset processing example.

This example demonstrates:
1. Loading and processing the MIRIAD dataset
2. Generating embeddings using ONNX models
3. Populating the database with processed data
4. Performing searches on the processed data
"""

import logging
from pathlib import Path

from loma_dataset import MedicalVectorDB, MiriadProcessor, ProcessingConfig
from alive_progress import alive_bar

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    """Run MIRIAD dataset processing example."""
    # Create database path
    db_path = Path("miriad_medical_minlm.db")

    # Configuration
    config = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=256,
        max_length=512,
        device="cpu",
        cache_dir="./cache",
        db_path=str(db_path),
        use_quantized=False,  # Use quantized version for better performance
        max_samples=500000,  # Process 50 samples for testing
        model_type="sentence_transformers",
    )

    # Create database
    db = MedicalVectorDB(str(db_path))

    logger.info("Created medical vector database for MIRIAD data")

    # Initialize processor
    processor = MiriadProcessor(config)

    try:
        # Initialize processor and database
        logger.info("Initializing processor...")
        processor.initialize()
        db.initialize()

        # Load MIRIAD dataset
        logger.info("Loading MIRIAD dataset...")
        dataset = processor.load_dataset()
        total_items = len(dataset) if hasattr(dataset, "__len__") else None
        logger.info(
            f"Loaded {total_items if total_items is not None else '?'} entries from MIRIAD dataset"
        )

        # Process and populate database with progress bar
        logger.info("Processing dataset and populating database...")
        if total_items is not None:
            with alive_bar(
                total_items, title="Populating DB", spinner="dots_waves2"
            ) as bar:
                # If populate_database can take an iterable, iterate and step the bar
                for item in dataset:
                    processor.populate_database(db, [item])
                    bar()
        else:
            # Fallback when length is unknown
            with alive_bar(title="Populating DB (stream)", enrich_print=False) as bar:
                for item in dataset:
                    processor.populate_database(db, [item])
                    bar()

        # Get database statistics
        stats = db.get_stats()
        logger.info(f"Database populated successfully: {stats}")

        # Example search
        logger.info("Performing example search...")

        # Search for documents about diabetes
        with alive_bar(
            1, title="Searching documents: diabetes treatment", spinner="classic"
        ) as bar:
            diabetes_results = db.search_documents_text("diabetes treatment", limit=3)
            bar()
        logger.info(f"Found {len(diabetes_results)} documents about diabetes:")
        for result in diabetes_results:
            logger.info(
                f"  - {result.document.title[:50]}... (similarity: {result.similarity:.3f})"
            )

        # Search for Q&A about heart disease
        if stats["qa_count"] > 0:
            # Generate embedding for search query
            embedding_generator = processor.embedding_generator
            with alive_bar(
                1, title="Generating query embedding", spinner="classic"
            ) as bar:
                query_embedding = embedding_generator.generate_embeddings(
                    ["heart disease symptoms"]
                )[0]
                bar()

            with alive_bar(1, title="Searching similar Q&A", spinner="classic") as bar:
                heart_results = db.search_similar_qa(query_embedding, limit=3)
                bar()
            logger.info(f"Found {len(heart_results)} Q&A entries about heart disease:")
            for result in heart_results:
                logger.info(
                    f"  - {result.qa.question[:50]}... (similarity: {result.similarity:.3f})"
                )

    except Exception as e:
        logger.error(f"Error processing MIRIAD dataset: {e}")
        raise


if __name__ == "__main__":
    main()
