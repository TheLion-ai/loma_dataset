#!/usr/bin/env python3
"""
Example: Using Sentence Transformers with LOMA Dataset
======================================================

This example demonstrates how to use sentence-transformers models
with the LOMA dataset processing pipeline.

Requirements:
    pip install sentence-transformers

Usage:
    python examples/sentence_transformers_example.py
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path so we can import loma_dataset
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loma_dataset.models import ProcessingConfig
from loma_dataset.processor import MiriadProcessor
from loma_dataset.database import MedicalVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate sentence-transformers usage."""
    
    # Example 1: Using sentence-transformers with auto detection
    print("=== Example 1: Auto model type (prefers sentence-transformers) ===")
    config_auto = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="auto",  # Will try sentence-transformers first
        batch_size=16,
        max_samples=100,  # Process only 100 samples for demo
        db_path="demo_auto.db"
    )
    
    try:
        processor_auto = MiriadProcessor(config_auto)
        processor_auto.initialize()
        print(f"✓ Successfully initialized with model type: {processor_auto.embedding_generator._actual_model_type}")
        processor_auto.close()
    except Exception as e:
        print(f"✗ Failed to initialize auto model: {e}")
    
    # Example 2: Explicitly using sentence-transformers
    print("\n=== Example 2: Explicit sentence-transformers ===")
    config_st = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence_transformers",
        batch_size=16,
        max_samples=100,
        db_path="demo_st.db"
    )
    
    try:
        processor_st = MiriadProcessor(config_st)
        processor_st.initialize()
        print(f"✓ Successfully initialized with model type: {processor_st.embedding_generator._actual_model_type}")
        
        # Test embedding generation
        test_texts = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes heart disease?"
        ]
        
        embeddings = processor_st.embedding_generator.generate_embeddings(test_texts)
        print(f"✓ Generated embeddings for {len(test_texts)} texts")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print(f"  Sample embedding (first 5 values): {embeddings[0][:5]}")
        
        processor_st.close()
    except Exception as e:
        print(f"✗ Failed with sentence-transformers: {e}")
    
    # Example 3: Using ONNX model (fallback)
    print("\n=== Example 3: Explicit ONNX model ===")
    config_onnx = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="onnx",
        use_quantized=True,
        batch_size=16,
        max_samples=100,
        db_path="demo_onnx.db"
    )
    
    try:
        processor_onnx = MiriadProcessor(config_onnx)
        processor_onnx.initialize()
        print(f"✓ Successfully initialized with model type: {processor_onnx.embedding_generator._actual_model_type}")
        processor_onnx.close()
    except Exception as e:
        print(f"✗ Failed with ONNX model: {e}")
    
    # Example 4: Comparing different sentence-transformer models
    print("\n=== Example 4: Different sentence-transformer models ===")
    
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast, small model
        "sentence-transformers/all-mpnet-base-v2",  # Better quality
        "sentence-transformers/paraphrase-MiniLM-L6-v2",  # Paraphrase detection
    ]
    
    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        config = ProcessingConfig(
            model_name=model_name,
            model_type="sentence_transformers",
            batch_size=8,
            max_samples=10,
            db_path=f"demo_{model_name.split('/')[-1]}.db"
        )
        
        try:
            processor = MiriadProcessor(config)
            processor.initialize()
            
            # Test with a medical question
            test_text = ["What are the side effects of aspirin?"]
            embeddings = processor.embedding_generator.generate_embeddings(test_text)
            
            print(f"  ✓ Model loaded successfully")
            print(f"  ✓ Embedding dimension: {len(embeddings[0])}")
            
            processor.close()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Cleanup demo databases
    print("\n=== Cleanup ===")
    demo_dbs = [
        "demo_auto.db", "demo_st.db", "demo_onnx.db",
        "demo_all-MiniLM-L6-v2.db", "demo_all-mpnet-base-v2.db", 
        "demo_paraphrase-MiniLM-L6-v2.db"
    ]
    
    for db_path in demo_dbs:
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed {db_path}")


if __name__ == "__main__":
    main()