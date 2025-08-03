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
        temp_db.initialize()
        stats = temp_db.get_stats()
        assert stats["document_count"] == 0
        assert stats["qa_count"] == 0
    
    def test_add_document(self, temp_db):
        """Test adding a document to the database."""
        temp_db.initialize()
        doc = MedicalDocument(
            id="test_001",
            title="Test Document",
            content="This is test content.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            url="https://example.com/test"
        )
        
        temp_db.add_document(doc)
        stats = temp_db.get_stats()
        assert stats["document_count"] == 1
    
    def test_add_qa(self, temp_db):
        """Test adding a Q&A to the database."""
        temp_db.initialize()
        
        # First add a document that the Q&A can reference
        doc = MedicalDocument(
            id="test_doc",
            title="Test Document",
            content="Test content for Q&A reference.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        temp_db.add_document(doc)
        
        qa = MedicalQA(
            id="qa_001",
            question="What is test?",
            answer="Test is a procedure.",
            vector=[0.1, 0.2, 0.3],
            document_id="test_doc"
        )
        
        temp_db.add_qa(qa)
        stats = temp_db.get_stats()
        assert stats["qa_count"] == 1
    
    def test_search_documents_text(self, temp_db):
        """Test text search for documents."""
        temp_db.initialize()
        doc = MedicalDocument(
            id="test_001",
            title="Diabetes Treatment",
            content="Information about diabetes treatment and management.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            url="https://example.com/diabetes"
        )
        
        temp_db.add_document(doc)
        results = temp_db.search_documents_text("diabetes", limit=5)
        
        assert len(results) == 1
        assert results[0].document.title == "Diabetes Treatment"
        assert results[0].similarity > 0
    
    def test_search_similar_qa(self, temp_db):
        """Test vector similarity search for Q&A."""
        temp_db.initialize()
        
        # First add a document that the Q&A can reference
        doc = MedicalDocument(
            id="doc_001",
            title="Diabetes Document",
            content="Information about diabetes.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        temp_db.add_document(doc)
        
        qa = MedicalQA(
            id="qa_001",
            question="What is diabetes?",
            answer="Diabetes is a metabolic disorder.",
            vector=[0.1, 0.2, 0.3],
            document_id="doc_001"
        )
        
        temp_db.add_qa(qa)
        
        # Search with similar vector
        query_vector = [0.1, 0.2, 0.3]
        results = temp_db.search_similar_qa(query_vector, limit=5)
        
        assert len(results) == 1
        assert results[0].qa.question == "What is diabetes?"
        assert results[0].similarity > 0.9  # Should be very similar
    
    def test_duplicate_id_handling(self, temp_db):
        """Test handling of duplicate IDs."""
        temp_db.initialize()
        doc1 = MedicalDocument(
            id="test_001",
            title="First Document",
            content="First content",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            url="https://example.com/1"
        )
        
        doc2 = MedicalDocument(
            id="test_001",  # Same ID
            title="Second Document",
            content="Second content",
            vector=[0.4, 0.5, 0.6],
            created_at=datetime.now(),
            url="https://example.com/2"
        )
        
        temp_db.add_document(doc1)
        
        # Adding duplicate should not raise error (ignore behavior)
        temp_db.add_document(doc2)
        
        stats = temp_db.get_stats()
        assert stats["document_count"] == 1  # Should still be 1
    
    def test_invalid_vector_dimension(self, temp_db):
        """Test handling of inconsistent vector dimensions."""
        temp_db.initialize()
        
        # First add a document that the Q&A can reference
        doc = MedicalDocument(
            id="doc_001",
            title="Test Document",
            content="Test content.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        temp_db.add_document(doc)
        
        qa1 = MedicalQA(
            id="qa_001",
            question="Question 1?",
            answer="Answer 1",
            vector=[0.1, 0.2, 0.3],  # 3D vector
            document_id="doc_001"
        )
        
        temp_db.add_qa(qa1)
        
        # Search with different dimension should handle gracefully
        query_vector = [0.1, 0.2]  # 2D vector
        results = temp_db.search_similar_qa(query_vector, limit=5)
        
        # Should return empty results or handle gracefully
        assert isinstance(results, list)
    
    def test_document_url_field(self, temp_db):
        """Test that documents properly store and retrieve URL field."""
        temp_db.initialize()
        test_url = "https://example.com/medical-research"
        
        doc = MedicalDocument(
            id="test_url_001",
            title="URL Test Document",
            content="This document tests URL storage.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            url=test_url
        )
        
        temp_db.add_document(doc)
        
        # Search for the document and verify URL is preserved
        results = temp_db.search_documents_text("URL Test", limit=1)
        
        assert len(results) == 1
        assert results[0].document.url == test_url
        assert results[0].document.title == "URL Test Document"
    
    def test_document_without_url(self, temp_db):
        """Test that documents work without URL field."""
        temp_db.initialize()
        
        doc = MedicalDocument(
            id="test_no_url_001",
            title="No URL Document",
            content="This document has no URL.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
            # No URL field provided
        )
        
        temp_db.add_document(doc)
        
        # Search for the document and verify it works without URL
        results = temp_db.search_documents_text("No URL", limit=1)
        
        assert len(results) == 1
        assert results[0].document.url is None
        assert results[0].document.title == "No URL Document"