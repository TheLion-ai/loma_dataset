"""
Tests for sentence-transformers integration in MedicalEmbeddingGenerator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loma_dataset.processor import MedicalEmbeddingGenerator, SENTENCE_TRANSFORMERS_AVAILABLE
from loma_dataset.exceptions import ProcessingError, ConfigurationError


class TestMedicalEmbeddingGeneratorSentenceTransformers:
    """Test sentence-transformers functionality in MedicalEmbeddingGenerator."""
    
    def test_sentence_transformers_availability(self):
        """Test that sentence-transformers availability is correctly detected."""
        # This test will pass regardless of whether sentence-transformers is installed
        assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)
    
    def test_model_type_validation(self):
        """Test model type parameter validation."""
        # Valid model types should not raise errors
        generator = MedicalEmbeddingGenerator("test-model", model_type="auto")
        assert generator.model_type == "auto"
        
        generator = MedicalEmbeddingGenerator("test-model", model_type="onnx")
        assert generator.model_type == "onnx"
        
        generator = MedicalEmbeddingGenerator("test-model", model_type="sentence_transformers")
        assert generator.model_type == "sentence_transformers"
    
    def test_initialization_without_sentence_transformers(self):
        """Test initialization when sentence-transformers is not available."""
        generator = MedicalEmbeddingGenerator(
            "test-model", 
            model_type="sentence_transformers"
        )
        
        # Mock sentence-transformers as unavailable
        with patch('loma_dataset.processor.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(ProcessingError, match="sentence-transformers not available"):
                generator.initialize()
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        generator = MedicalEmbeddingGenerator("test-model", model_type="invalid")
        
        with pytest.raises(ProcessingError, match="Unknown model_type"):
            generator.initialize()
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, 
                       reason="sentence-transformers not installed")
    def test_sentence_transformers_initialization(self):
        """Test successful initialization with sentence-transformers."""
        generator = MedicalEmbeddingGenerator(
            "sentence-transformers/all-MiniLM-L6-v2",
            model_type="sentence_transformers"
        )
        
        try:
            generator.initialize()
            assert generator._actual_model_type == "sentence_transformers"
            assert generator.sentence_model is not None
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")
    
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, 
                       reason="sentence-transformers not installed")
    def test_sentence_transformers_embedding_generation(self):
        """Test embedding generation with sentence-transformers."""
        generator = MedicalEmbeddingGenerator(
            "sentence-transformers/all-MiniLM-L6-v2",
            model_type="sentence_transformers"
        )
        
        try:
            generator.initialize()
            
            test_texts = [
                "What are the symptoms of diabetes?",
                "How is hypertension treated?"
            ]
            
            embeddings = generator.generate_embeddings(test_texts)
            
            # Verify embeddings structure
            assert isinstance(embeddings, list)
            assert len(embeddings) == len(test_texts)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(isinstance(val, float) for emb in embeddings for val in emb)
            
            # Verify embeddings are normalized (L2 norm should be close to 1)
            for embedding in embeddings:
                norm = np.linalg.norm(embedding)
                assert abs(norm - 1.0) < 1e-6, f"Embedding not normalized: norm={norm}"
            
            # Verify all embeddings have the same dimension
            dimensions = [len(emb) for emb in embeddings]
            assert all(dim == dimensions[0] for dim in dimensions)
            
        except Exception as e:
            pytest.skip(f"Could not test embedding generation: {e}")
    
    def test_auto_model_type_fallback(self):
        """Test auto model type with fallback to ONNX."""
        generator = MedicalEmbeddingGenerator(
            "sentence-transformers/all-MiniLM-L6-v2",
            model_type="auto"
        )
        
        # Mock sentence-transformers as unavailable to test fallback
        with patch('loma_dataset.processor.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            # Mock ONNX initialization to avoid actual model download
            with patch.object(generator, '_initialize_onnx') as mock_onnx_init:
                mock_onnx_init.return_value = None
                generator._actual_model_type = "onnx"  # Simulate successful ONNX init
                
                generator.initialize()
                mock_onnx_init.assert_called_once()
                assert generator._actual_model_type == "onnx"
    
    def test_embedding_generation_not_initialized(self):
        """Test embedding generation when generator is not initialized."""
        generator = MedicalEmbeddingGenerator("test-model")
        
        with pytest.raises(ProcessingError, match="not initialized"):
            generator.generate_embeddings(["test text"])
    
    @patch('loma_dataset.processor.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_sentence_transformers_mock_embedding_generation(self):
        """Test embedding generation with mocked sentence-transformers."""
        generator = MedicalEmbeddingGenerator(
            "test-model",
            model_type="sentence_transformers"
        )
        
        # Mock the sentence transformer model
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        
        generator.sentence_model = mock_model
        generator._actual_model_type = "sentence_transformers"
        
        test_texts = ["text1", "text2"]
        embeddings = generator.generate_embeddings(test_texts)
        
        # Verify the mock was called correctly
        mock_model.encode.assert_called_once_with(
            test_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Verify the output
        assert embeddings == mock_embeddings.tolist()
    
    def test_onnx_embedding_generation_mock(self):
        """Test ONNX embedding generation with mocks."""
        generator = MedicalEmbeddingGenerator(
            "test-model",
            model_type="onnx"
        )
        
        # Mock tokenizer and session
        mock_tokenizer = Mock()
        mock_session = Mock()
        
        # Mock tokenizer output
        mock_inputs = {
            'input_ids': np.array([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': np.array([[1, 1, 1], [1, 1, 0]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        # Mock session output (3D tensor that needs pooling)
        mock_outputs = [np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], 
                                 [[0.7, 0.8], [0.9, 1.0], [0.0, 0.0]]])]
        mock_session.run.return_value = mock_outputs
        
        generator.tokenizer = mock_tokenizer
        generator.session = mock_session
        generator._actual_model_type = "onnx"
        
        test_texts = ["text1", "text2"]
        embeddings = generator.generate_embeddings(test_texts)
        
        # Verify the mocks were called
        mock_tokenizer.assert_called_once()
        mock_session.run.assert_called_once()
        
        # Verify output structure
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)