# MIRIAD Dataset Processing

This directory contains tools for processing the MIRIAD medical Q&A dataset and converting it to a searchable vector database format using medical text embeddings.

## Overview

The processing pipeline:
1. **Downloads** the MIRIAD dataset from Hugging Face (5.8M medical Q&A pairs)
2. **Generates embeddings** using the MedEmbed-small-onnx model optimized for medical text
3. **Processes in batches** with configurable batch size for memory efficiency
4. **Stores** in SQLite database with vector search capabilities
5. **Provides** both vector similarity and full-text search

## Files

- `process_miriad_dataset.py` - Main processing script
- `example_usage.py` - Example usage and testing script
- `db.py` - Database schema and operations
- `pyproject.toml` - Project dependencies

## Installation

Install the required dependencies:

```bash
# Using pip
pip install datasets onnxruntime transformers tqdm huggingface-hub

# Or using uv (recommended)
uv sync
```

## Quick Start

### 1. Test Processing (Small Sample)

Process 100 samples for testing:

```bash
python example_usage.py --test --demo-search
```

This will:
- Process 100 Q&A pairs from MIRIAD
- Generate embeddings using the medical model
- Create a test database
- Demonstrate search functionality

### 2. Full Dataset Processing

Process the complete MIRIAD dataset:

```bash
python example_usage.py --full
```

⚠️ **Note**: The full dataset contains 5.8M entries and will take significant time and storage.

### 3. Custom Processing

Use the main script directly with custom parameters:

```bash
python process_miriad_dataset.py \
    --batch-size 64 \
    --max-samples 10000 \
    --db-path my_medical_db.db \
    --verbose
```

## Command Line Options

### Main Processing Script (`process_miriad_dataset.py`)

```bash
python process_miriad_dataset.py [OPTIONS]
```

**Options:**
- `--model-name` - Hugging Face model for embeddings (default: AleksanderObuchowski/medembed-small-onnx)
- `--dataset-name` - Dataset name (default: miriad/miriad-5.8M)
- `--batch-size` - Processing batch size (default: 32)
- `--max-samples` - Maximum samples to process (default: all)
- `--db-path` - Database file path (default: medical_qa_db.db)
- `--cache-dir` - Cache directory for models/datasets
- `--no-quantized` - Use full precision model instead of quantized
- `--no-skip-existing` - Don't skip existing entries
- `--verbose` - Enable verbose logging

### Example Usage Script (`example_usage.py`)

```bash
python example_usage.py [OPTIONS]
```

**Options:**
- `--test` - Run test processing (100 samples)
- `--full` - Run full dataset processing
- `--demo-search` - Demonstrate search functionality
- `--batch-size` - Batch size for processing
- `--model-name` - Model name for embeddings
- `--db-path` - Database file path

## Dataset Structure

The MIRIAD dataset contains medical Q&A pairs with metadata:

```python
{
    "id": "sample_id",
    "question": "What are the symptoms of diabetes?",
    "answer": "Common symptoms include increased thirst, frequent urination...",
    "document_id": "doc_123",
    "specialty": "Endocrinology",
    "title": "Diabetes Management Guidelines",
    "url": "https://example.com/paper",
    "year": 2023
}
```

## Database Schema

The processed data is stored in two main tables:

### `medical_qa` Table
- `id` - Unique Q&A identifier
- `question` - Medical question
- `answer` - Medical answer
- `vector` - 384-dimensional embedding (BLOB)
- `document_id` - Reference to source document
- `specialty` - Medical specialty
- `created_at` - Timestamp

### `documents` Table
- `id` - Unique document identifier
- `paper_title` - Research paper title
- `source_url` - URL to paper or source
- `year` - Publication year
- `venue` - Journal/conference
- `specialty` - Medical specialty
- `passage_text` - Text passage
- `abstract` - Paper abstract
- `authors` - Author list (JSON)
- `doi` - Digital Object Identifier

## Search Capabilities

### 1. Vector Similarity Search

Find semantically similar Q&A pairs:

```python
from db import MedicalVectorDB

db = MedicalVectorDB("medical_qa_db.db")
db.initialize()

# Generate query embedding
query_vector = embedding_model.encode("diabetes treatment")

# Search similar Q&A
results = db.search_similar_qa(
    query_vector=query_vector,
    limit=10,
    threshold=0.7,
    specialty="Endocrinology"  # Optional filter
)

for result in results:
    print(f"Q: {result.qa.question}")
    print(f"A: {result.qa.answer}")
    print(f"Similarity: {result.similarity:.3f}")
```

### 2. Full-Text Search

Traditional keyword-based search:

```python
results = db.search_full_text(
    query="heart disease treatment",
    limit=10,
    specialty="Cardiology"  # Optional filter
)
```

### 3. Specialty Filtering

Get documents by medical specialty:

```python
cardiology_docs = db.get_documents_by_specialty("Cardiology")
```

## Performance Considerations

### Batch Size
- **Small batches (16-32)**: Lower memory usage, slower processing
- **Large batches (64-128)**: Higher memory usage, faster processing
- **Recommended**: Start with 32, adjust based on available RAM

### Model Selection
- **Quantized model**: Faster inference, smaller memory footprint
- **Full precision**: Better accuracy, higher resource usage
- **Recommended**: Use quantized for most applications

### Storage Requirements
- **Full dataset**: ~15-20 GB (including vectors)
- **Test dataset (100 samples)**: ~1 MB
- **Embeddings**: ~1.5 KB per Q&A pair (384 dimensions × 4 bytes)

## Monitoring Progress

The processing script provides detailed progress information:

```
Processing samples: 45%|████▌     | 2250/5000 [02:15<02:45, 16.6it/s]
INFO - Processed 2240 samples, 10 errors
INFO - Dataset processing completed!
INFO - Successfully processed: 4990 samples
INFO - Errors encountered: 10 samples
```

## Error Handling

Common issues and solutions:

### 1. Memory Errors
- Reduce batch size: `--batch-size 16`
- Use quantized model: default behavior
- Process in smaller chunks: `--max-samples 1000`

### 2. Network Issues
- Set cache directory: `--cache-dir ./cache`
- Resume processing: `--skip-existing` (default)

### 3. Model Loading Errors
- Check internet connection
- Verify model name: `AleksanderObuchowski/medembed-small-onnx`
- Clear cache and retry

## Example Output

```bash
$ python example_usage.py --test --demo-search

INFO - Starting MIRIAD dataset processing...
INFO - Loading dataset miriad/miriad-5.8M...
INFO - Loaded 100 samples from dataset
INFO - Loading tokenizer from AleksanderObuchowski/medembed-small-onnx
INFO - Loading ONNX model from model_quantized.onnx
Processing samples: 100%|██████████| 100/100 [00:45<00:00, 2.2it/s]

=== Test Processing Results ===
total_qa_entries: 98
total_documents: 87
qa_with_vectors: 98
specialties: ['Cardiology', 'Neurology', 'Endocrinology', 'Pediatrics']
processed_count: 98
error_count: 2

=== Full-Text Search Demo ===
Search for 'heart disease' (3 results):
  1. Q: What are the risk factors for coronary heart disease?
     A: Risk factors include high blood pressure, high cholesterol, smoking...
     Similarity: 0.892

=== Vector Search Demo ===
Found 98 Q&A entries with vectors
Vector similarity search (3 results):
  1. Q: How is myocardial infarction diagnosed?
     Similarity: 0.756

✅ All operations completed successfully!
```

## Contributing

To extend the processing pipeline:

1. **Add new embedding models**: Modify `MedicalEmbeddingGenerator` class
2. **Support new datasets**: Extend `extract_qa_data` method
3. **Add preprocessing**: Modify text processing in `process_batch`
4. **Enhance search**: Add new search methods to database class

## License

This project follows the same license as the original MIRIAD dataset and MedEmbed model.