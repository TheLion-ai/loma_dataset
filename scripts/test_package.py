#!/usr/bin/env python3
"""
Test script to verify the new package structure works correctly.

This script tests:
1. Package imports
2. Basic functionality
3. Database operations
4. Model validation
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        from loma_dataset import (
            MedicalVectorDB, 
            MedicalDocument, 
            MedicalQA, 
            ProcessingConfig,
            MiriadProcessor,
            __version__
        )
        from loma_dataset.exceptions import (
            LomaDatasetError,
            DatabaseError,
            ValidationError
        )
        logger.info("✓ All imports successful")
        logger.info(f"✓ Package version: {__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_model_validation():
    """Test model validation."""
    logger.info("Testing model validation...")
    
    try:
        from loma_dataset import MedicalDocument, MedicalQA, ProcessingConfig
        from loma_dataset.exceptions import ValidationError
        
        # Test valid document
        doc = MedicalDocument(
            id="test_001",
            title="Test Document",
            content="Test content",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        logger.info("✓ Valid document creation successful")
        
        # Test invalid document (empty ID)
        try:
            MedicalDocument(
                id="",
                title="Test",
                content="Content",
                vector=[0.1],
                created_at=datetime.now()
            )
            logger.error("✗ Empty ID validation failed")
            return False
        except ValidationError:
            logger.info("✓ Empty ID validation successful")
        
        # Test valid Q&A
        qa = MedicalQA(
            id="qa_001",
            question="Test question?",
            answer="Test answer",
            vector=[0.1, 0.2, 0.3],
            document_id="test_001"
        )
        logger.info("✓ Valid Q&A creation successful")
        
        # Test valid config
        config = ProcessingConfig(
            model_name="test-model",
            batch_size=16,
            max_length=256,
            device="cpu"
        )
        logger.info("✓ Valid config creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model validation failed: {e}")
        return False


def test_database_operations():
    """Test basic database operations."""
    logger.info("Testing database operations...")
    
    try:
        from loma_dataset import MedicalVectorDB, MedicalDocument, MedicalQA
        
        # Create temporary database
        db_path = Path("test_package.db")
        db = MedicalVectorDB(str(db_path))
        db.initialize()
        
        # Test initial stats
        stats = db.get_stats()
        assert stats["document_count"] == 0
        assert stats["qa_count"] == 0
        logger.info("✓ Database initialization successful")
        
        # Add document
        doc = MedicalDocument(
            id="test_001",
            title="Test Document",
            content="Test content about medical topics",
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            created_at=datetime.now()
        )
        db.add_document(doc)
        logger.info("✓ Document addition successful")
        
        # Add Q&A
        qa = MedicalQA(
            id="qa_001",
            question="What is a medical test?",
            answer="A medical test is a procedure to diagnose conditions.",
            vector=[0.2, 0.3, 0.4, 0.5, 0.6],
            document_id="test_001"
        )
        db.add_qa(qa)
        logger.info("✓ Q&A addition successful")
        
        # Test search operations
        text_results = db.search_documents_text("medical", limit=5)
        assert len(text_results) > 0
        logger.info("✓ Text search successful")
        
        vector_results = db.search_similar_qa([0.2, 0.3, 0.4, 0.5, 0.6], limit=5)
        assert len(vector_results) > 0
        logger.info("✓ Vector search successful")
        
        # Test final stats
        final_stats = db.get_stats()
        assert final_stats["document_count"] == 1
        assert final_stats["qa_count"] == 1
        logger.info("✓ Final statistics correct")
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()
            logger.info("✓ Database cleanup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Database operations failed: {e}")
        # Cleanup on error
        if 'db_path' in locals() and db_path.exists():
            db_path.unlink()
        return False


def main():
    """Run all tests."""
    logger.info("=== LOMA Dataset Package Test ===")
    
    tests = [
        ("Imports", test_imports),
        ("Model Validation", test_model_validation),
        ("Database Operations", test_database_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Package is working correctly.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())