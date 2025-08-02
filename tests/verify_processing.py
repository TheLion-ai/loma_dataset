#!/usr/bin/env python3
"""
Comprehensive verification script for MIRIAD dataset processing.
This script thoroughly tests the processing logic to ensure correctness.
"""

import logging
from pathlib import Path
from datasets import load_dataset
from collections import Counter

from loma_dataset import MedicalVectorDB, MiriadProcessor, ProcessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_raw_dataset(num_samples=50):
    """Analyze the raw MIRIAD dataset to understand its structure."""
    print("=== ANALYZING RAW MIRIAD DATASET ===")
    
    # Load dataset
    dataset = load_dataset('miriad/miriad-5.8M', split='train', streaming=True)
    
    # Collect samples
    samples = []
    paper_ids = []
    questions = []
    
    print(f"Taking first {num_samples} samples...")
    for i, sample in enumerate(dataset.take(num_samples)):
        samples.append(sample)
        paper_ids.append(sample['paper_id'])
        questions.append(sample['question'])
        
        if i < 10:  # Show first 10 for verification
            print(f"Sample {i+1}: Paper {sample['paper_id']} - {sample['question'][:80]}...")
    
    # Analysis
    unique_papers = set(paper_ids)
    unique_questions = set(questions)
    
    print(f"\nRaw Dataset Analysis:")
    print(f"- Total samples: {len(samples)}")
    print(f"- Unique paper IDs: {len(unique_papers)}")
    print(f"- Unique questions: {len(unique_questions)}")
    
    # Check for papers with multiple Q&As
    paper_counts = Counter(paper_ids)
    papers_with_multiple_qa = {pid: count for pid, count in paper_counts.items() if count > 1}
    print(f"- Papers with multiple Q&As: {len(papers_with_multiple_qa)}")
    
    if papers_with_multiple_qa:
        print("  Examples of papers with multiple Q&As:")
        for pid, count in list(papers_with_multiple_qa.items())[:3]:
            print(f"    Paper {pid}: {count} Q&As")
    
    return samples, unique_papers, unique_questions


def test_processing(num_samples=50):
    """Test the processing with our fixed logic."""
    print(f"\n=== TESTING PROCESSING WITH {num_samples} SAMPLES ===")
    
    # Create database path
    db_path = Path("verification_test.db")
    if db_path.exists():
        db_path.unlink()
    
    # Configuration
    config = ProcessingConfig(
        model_name="AleksanderObuchowski/medembed-small-onnx",
        batch_size=5,  # Small batches for better tracking
        max_length=512,
        device="cpu",
        cache_dir="./cache",
        db_path=str(db_path),
        use_quantized=True,
        max_samples=num_samples,
        skip_existing=False  # Ensure all samples are processed
    )
    
    # Create database and processor
    db = MedicalVectorDB(str(db_path))
    processor = MiriadProcessor(config)
    
    try:
        # Initialize
        print("Initializing processor and database...")
        processor.initialize()
        db.initialize()
        
        # Load and process dataset
        print("Loading and processing dataset...")
        dataset = processor.load_dataset()
        processor.populate_database(db, dataset)
        
        # Get statistics
        stats = db.get_stats()
        print(f"\nProcessing Results:")
        print(f"- Q&A entries in database: {stats.get('qa_count', 'N/A')}")
        print(f"- Documents in database: {stats.get('document_count', 'N/A')}")
        print(f"- Available stats: {list(stats.keys())}")
        
        return stats, db_path
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        db.close()


def verify_database_content(db_path):
    """Verify the content of the processed database."""
    print(f"\n=== VERIFYING DATABASE CONTENT ===")
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check Q&A records
    cursor.execute("SELECT id, question, answer, document_id FROM medical_qa LIMIT 5")
    qa_records = cursor.fetchall()
    
    print("Sample Q&A records:")
    for i, (qa_id, question, answer, doc_id) in enumerate(qa_records, 1):
        print(f"{i}. ID: {qa_id[:12]}...")
        print(f"   Question ({len(question)} chars): {question[:80]}...")
        print(f"   Answer ({len(answer)} chars): {answer[:80]}...")
        print(f"   Document ID: {doc_id[:12]}...")
        print()
    
    # Check documents
    cursor.execute("SELECT id, title, specialty, year FROM documents LIMIT 5")
    doc_records = cursor.fetchall()
    
    print("Sample document records:")
    for i, (doc_id, title, specialty, year) in enumerate(doc_records, 1):
        print(f"{i}. ID: {doc_id[:12]}...")
        print(f"   Title: {title[:80]}...")
        print(f"   Specialty: {specialty}")
        print(f"   Year: {year}")
        print()
    
    # Check for potential issues
    cursor.execute("SELECT COUNT(DISTINCT document_id) FROM medical_qa")
    unique_docs_in_qa = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]
    
    print(f"Data integrity check:")
    print(f"- Unique document IDs referenced in Q&A: {unique_docs_in_qa}")
    print(f"- Total documents in database: {total_docs}")
    
    if unique_docs_in_qa == total_docs:
        print("✅ Document references are consistent")
    else:
        print("⚠️  Document reference mismatch detected")
    
    conn.close()


def main():
    """Run comprehensive verification."""
    try:
        # Step 1: Analyze raw dataset
        samples, unique_papers, unique_questions = analyze_raw_dataset(50)
        
        # Step 2: Test processing
        stats, db_path = test_processing(50)
        
        # Step 3: Verify database content
        verify_database_content(db_path)
        
        # Step 4: Summary
        print(f"\n=== VERIFICATION SUMMARY ===")
        print(f"Raw dataset: {len(samples)} samples, {len(unique_papers)} papers, {len(unique_questions)} questions")
        
        qa_count = stats.get('qa_count', stats.get('total_qa', 0))
        doc_count = stats.get('document_count', stats.get('total_documents', 0))
        
        print(f"Processed: {qa_count} Q&As, {doc_count} documents")
        
        # Calculate processing efficiency
        processing_rate = (qa_count / len(samples)) * 100
        print(f"Processing rate: {processing_rate:.1f}% ({qa_count}/{len(samples)})")
        
        if processing_rate > 80:
            print("✅ Processing appears to be working correctly!")
        elif processing_rate > 50:
            print("⚠️  Processing working but some data may be filtered")
        else:
            print("❌ Low processing rate - investigate issues")
        
        # Clean up
        response = input(f"\nDelete test database {db_path}? (y/N): ")
        if response.lower() == 'y':
            db_path.unlink()
            print("Test database deleted")
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise


if __name__ == "__main__":
    main()