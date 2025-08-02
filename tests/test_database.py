"""Tests for the database module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from loma_dataset.database import MedicalVectorDB
from loma_dataset.models import MedicalDocument, MedicalQA
from loma_dataset.exceptions import DatabaseError


class TestMedicalVectorDB:
    """Test cases for MedicalVectorDB."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        db = MedicalVectorDB(db_path)
        yield db
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_database_initialization(self, temp_db):
        """Test database initialization."""
        stats = temp_db.get_stats()
        assert stats["document_count"] == 0
        assert stats["qa_count"] == 0
    
    def test_add_document(self, temp_db):
        """Test adding a document to the database."""
        doc = MedicalDocument(
            id="test_001",
            title="Test Document",
            content="This is test content.",
            source_url="https://example.com/test",
            passage_text="Test passage",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        
        temp_db.add_document(doc)
        stats = temp_db.get_stats()
        assert stats["document_count"] == 1
    
    def test_add_qa(self, temp_db):
        """Test adding a Q&A to the database."""
        qa = MedicalQA(
            id="qa_001",
            question="What is test?",
            answer="Test is a procedure.",
            context="Medical context",
            vector=[0.1, 0.2, 0.3]
        )
        
        temp_db.add_qa(qa)
        stats = temp_db.get_stats()
        assert stats["qa_count"] == 1
    
    def test_search_documents_text(self, temp_db):
        """Test text search for documents."""
        doc = MedicalDocument(
            id="test_001",
            title="Diabetes Treatment",
            content="Information about diabetes treatment and management.",
            source_url="https://example.com/diabetes",
            passage_text="Diabetes requires careful management",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        
        temp_db.add_document(doc)
        results = temp_db.search_documents_text("diabetes", limit=5)
        
        assert len(results) == 1
        assert results[0].title == "Diabetes Treatment"
        assert results[0].score > 0
    
    def test_search_similar_qa(self, temp_db):
        """Test vector similarity search for Q&A."""
        qa = MedicalQA(
            id="qa_001",
            question="What is diabetes?",
            answer="Diabetes is a metabolic disorder.",
            context="Medical knowledge",
            vector=[0.1, 0.2, 0.3]
        )
        
        temp_db.add_qa(qa)
        
        # Search with similar vector
        query_vector = [0.1, 0.2, 0.3]
        results = temp_db.search_similar_qa(query_vector, limit=5)
        
        assert len(results) == 1
        assert results[0].question == "What is diabetes?"
        assert results[0].similarity > 0.9  # Should be very similar
    
    def test_duplicate_id_handling(self, temp_db):
        """Test handling of duplicate IDs."""
        doc1 = MedicalDocument(
            id="test_001",
            title="First Document",
            content="First content",
            source_url="https://example.com/1",
            passage_text="First passage",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        
        doc2 = MedicalDocument(
            id="test_001",  # Same ID
            title="Second Document",
            content="Second content",
            source_url="https://example.com/2",
            passage_text="Second passage",
            vector=[0.4, 0.5, 0.6],
            created_at=datetime.now()
        )
        
        temp_db.add_document(doc1)
        
        # Adding duplicate should not raise error (replace behavior)
        temp_db.add_document(doc2)
        
        stats = temp_db.get_stats()
        assert stats["document_count"] == 1  # Should still be 1
    
    def test_invalid_vector_dimension(self, temp_db):
        """Test handling of inconsistent vector dimensions."""
        qa1 = MedicalQA(
            id="qa_001",
            question="Question 1?",
            answer="Answer 1",
            context="Context",
            vector=[0.1, 0.2, 0.3]  # 3D vector
        )
        
        temp_db.add_qa(qa1)
        
        # Search with different dimension should handle gracefully
        query_vector = [0.1, 0.2]  # 2D vector
        results = temp_db.search_similar_qa(query_vector, limit=5)
        
        # Should return empty results or handle gracefully
        assert isinstance(results, list)