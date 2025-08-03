"""Tests for the models module."""

import pytest
from datetime import datetime
from loma_dataset.models import MedicalDocument, MedicalQA, ProcessingConfig
from loma_dataset.exceptions import ValidationError


class TestMedicalDocument:
    """Test cases for MedicalDocument."""
    
    def test_valid_document_creation(self):
        """Test creating a valid MedicalDocument."""
        doc = MedicalDocument(
            id="test_001",
            title="Test Document",
            content="This is test content.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            url="https://example.com/test"
        )
        assert doc.id == "test_001"
        assert doc.title == "Test Document"
        assert doc.url == "https://example.com/test"
        assert len(doc.vector) == 3
    
    def test_empty_id_validation(self):
        """Test that empty ID raises ValidationError."""
        with pytest.raises(ValidationError, match="Document ID cannot be empty"):
            MedicalDocument(
                id="",
                title="Test",
                content="Content",
                vector=[0.1],
                created_at=datetime.now(),
                url="https://example.com"
            )
    
    def test_empty_vector_validation(self):
        """Test that empty vector raises ValidationError."""
        with pytest.raises(ValidationError, match="Vector cannot be empty"):
            MedicalDocument(
                id="test_001",
                title="Test",
                content="Content",
                vector=[],
                created_at=datetime.now(),
                url="https://example.com"
            )
    
    def test_document_with_url(self):
        """Test creating a document with URL."""
        doc = MedicalDocument(
            id="test_002",
            title="Document with URL",
            content="This document has a URL.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            url="https://example.com/medical-paper"
        )
        assert doc.url == "https://example.com/medical-paper"
    
    def test_document_without_url(self):
        """Test creating a document without URL."""
        doc = MedicalDocument(
            id="test_003",
            title="Document without URL",
            content="This document has no URL.",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        assert doc.url is None


class TestMedicalQA:
    """Test cases for MedicalQA."""
    
    def test_valid_qa_creation(self):
        """Test creating a valid MedicalQA."""
        qa = MedicalQA(
            id="qa_001",
            question="What is test?",
            answer="Test is a procedure.",
            vector=[0.1, 0.2, 0.3],
            document_id="doc_001"
        )
        assert qa.id == "qa_001"
        assert qa.question == "What is test?"
        assert qa.document_id == "doc_001"
        assert len(qa.vector) == 3
    
    def test_empty_question_validation(self):
        """Test that empty question raises ValidationError."""
        with pytest.raises(ValidationError, match="Question cannot be empty"):
            MedicalQA(
                id="qa_001",
                question="",
                answer="Answer",
                vector=[0.1],  
                document_id="doc_001"
            )
    
    def test_empty_answer_validation(self):
        """Test that empty answer raises ValidationError."""
        with pytest.raises(ValidationError, match="Answer cannot be empty"):
            MedicalQA(
                id="qa_001",
                question="Question?",
                answer="",
                vector=[0.1],
                document_id="doc_001"
            )


class TestProcessingConfig:
    """Test cases for ProcessingConfig."""
    
    def test_valid_config_creation(self):
        """Test creating a valid ProcessingConfig."""
        config = ProcessingConfig(
            model_name="test-model",
            batch_size=16,
            max_length=256,
            device="cpu"
        )
        assert config.model_name == "test-model"
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.device == "cpu"
    
    def test_invalid_batch_size_validation(self):
        """Test that invalid batch size raises ValidationError."""
        with pytest.raises(ValidationError, match="Batch size must be positive"):
            ProcessingConfig(
                model_name="test-model",
                batch_size=0,
                max_length=256,
                device="cpu"
            )
    
    def test_invalid_max_length_validation(self):
        """Test that invalid max length raises ValidationError."""
        with pytest.raises(ValidationError, match="Max length must be positive"):
            ProcessingConfig(
                model_name="test-model",
                batch_size=16,
                max_length=-1,
                device="cpu"
            )