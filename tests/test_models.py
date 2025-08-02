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
            source_url="https://example.com/test",
            passage_text="Test passage",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now()
        )
        assert doc.id == "test_001"
        assert doc.title == "Test Document"
        assert len(doc.vector) == 3
    
    def test_empty_id_validation(self):
        """Test that empty ID raises ValidationError."""
        with pytest.raises(ValidationError, match="Document ID cannot be empty"):
            MedicalDocument(
                id="",
                title="Test",
                content="Content",
                source_url="https://example.com",
                passage_text="Passage",
                vector=[0.1],
                created_at=datetime.now()
            )
    
    def test_empty_vector_validation(self):
        """Test that empty vector raises ValidationError."""
        with pytest.raises(ValidationError, match="Vector cannot be empty"):
            MedicalDocument(
                id="test_001",
                title="Test",
                content="Content",
                source_url="https://example.com",
                passage_text="Passage",
                vector=[],
                created_at=datetime.now()
            )


class TestMedicalQA:
    """Test cases for MedicalQA."""
    
    def test_valid_qa_creation(self):
        """Test creating a valid MedicalQA."""
        qa = MedicalQA(
            id="qa_001",
            question="What is test?",
            answer="Test is a procedure.",
            context="Medical context",
            vector=[0.1, 0.2, 0.3]
        )
        assert qa.id == "qa_001"
        assert qa.question == "What is test?"
        assert len(qa.vector) == 3
    
    def test_empty_question_validation(self):
        """Test that empty question raises ValidationError."""
        with pytest.raises(ValidationError, match="Question cannot be empty"):
            MedicalQA(
                id="qa_001",
                question="",
                answer="Answer",
                context="Context",
                vector=[0.1]
            )
    
    def test_empty_answer_validation(self):
        """Test that empty answer raises ValidationError."""
        with pytest.raises(ValidationError, match="Answer cannot be empty"):
            MedicalQA(
                id="qa_001",
                question="Question?",
                answer="",
                context="Context",
                vector=[0.1]
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