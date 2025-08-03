# LOMA Dataset - Medical Q&A Vector Database

A high-quality Python package for managing medical Q&A data with vector search capabilities, specifically designed for processing the MIRIAD medical dataset.

## Features

- 🏥 **Medical-focused**: Optimized for medical Q&A data and terminology
- 🔍 **Vector Search**: Advanced similarity search using L2 normalized medical text embeddings
- 📊 **SQLite/libSQL**: Efficient database storage with full-text search
- 🚀 **ONNX Optimized**: Fast inference using quantized ONNX models
- 🤗 **Sentence Transformers**: Easy-to-use sentence-transformers integration with auto-fallback
- 📦 **Easy to Use**: Simple API for data processing and retrieval
- 🔧 **Configurable**: Flexible configuration for different use cases
- ⚡ **L2 Normalized**: All embeddings are automatically L2 normalized for optimal cosine similarity

## Installation

```bash
# Install from source
git clone <repository-url>
cd loma_dataset
pip install -e .

# Install with sentence-transformers support (recommended)
pip install -e ".[sentence-transformers]"

# Install all optional dependencies
pip install -e ".[all]"

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from loma_dataset import MedicalVectorDB, MedicalDocument, MedicalQA

# Initialize database
db = MedicalVectorDB("medical_qa.db")
db.initialize()

# Add a document
document = MedicalDocument(
    id="doc_001",
    paper_title="Diabetes Management Guidelines",
    source_url="https://example.com/diabetes-guidelines",
    passage_text="Diabetes management requires comprehensive care...",
    vector=[0.1] * 384,  # Your embedding vector
    created_at="2023-01-15T10:30:00"
)
db.add_document(document)

# Add Q&A
qa = MedicalQA(
    id="qa_001",
    question="What are the symptoms of diabetes?",
    answer="Common symptoms include increased thirst...",
    vector=[0.2] * 384,  # Your embedding vector
    document_id="doc_001"
)
db.add_qa(qa)

# Search similar Q&A
results = db.search_similar_qa(query_vector, limit=5)
for result in results:
    print(f"Q: {result.qa.question}")
    print(f"A: {result.qa.answer}")
    print(f"Similarity: {result.similarity:.3f}")
```

### Processing MIRIAD Dataset

```python
from loma_dataset import MiriadProcessor, ProcessingConfig

# Configure processing with sentence-transformers (recommended)
config = ProcessingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_type="sentence_transformers",  # or "auto" for auto-detection
    max_samples=1000,  # Process 1000 samples for testing
    batch_size=32,
    db_path="medical_qa.db"
)

# Process dataset
processor = MiriadProcessor(config)
processor.initialize()
processor.process_dataset()

# Get statistics
stats = processor.get_statistics()
print(f"Processed: {stats['processed_count']} entries")
```

### Using Different Model Types

```python
# Option 1: Auto-detection (tries sentence-transformers first, falls back to ONNX)
config_auto = ProcessingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_type="auto"  # Default
)

# Option 2: Explicit sentence-transformers
config_st = ProcessingConfig(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_type="sentence_transformers"
)

# Option 3: ONNX for faster inference
config_onnx = ProcessingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_type="onnx",
    use_quantized=True
)
```

## Project Structure

```
loma_dataset/
├── src/loma_dataset/          # Main package
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Data models and configurations
│   ├── database.py           # Database operations
│   ├── processor.py          # Dataset processing
│   └── exceptions.py         # Custom exceptions
├── examples/                 # Usage examples
├── tests/                    # Test suite
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
└── README.md                # This file
```

## API Reference

### Core Classes

#### `MedicalVectorDB`
Main database interface for storing and retrieving medical data.

**Methods:**
- `initialize()`: Initialize database and create tables
- `add_document(document)`: Add a medical document
- `add_qa(qa)`: Add a Q&A entry
- `search_similar_qa(vector, limit, threshold)`: Vector similarity search
- `get_stats()`: Get database statistics

#### `MedicalDocument`
Represents a medical research document.

**Required Fields:**
- `id`: Unique identifier
- `paper_title`: Document title
- `source_url`: URL to source
- `passage_text`: Text content
- `vector`: Embedding vector (384-dim)
- `created_at`: Creation timestamp

#### `MedicalQA`
Represents a medical Q&A entry.

**Required Fields:**
- `id`: Unique identifier
- `question`: Medical question
- `answer`: Corresponding answer
- `vector`: Embedding vector (384-dim)

#### `MiriadProcessor`
Processes MIRIAD dataset and generates embeddings.

**Methods:**
- `initialize()`: Setup processor and models
- `process_dataset()`: Process complete dataset
- `get_statistics()`: Get processing stats

### Configuration

#### `ProcessingConfig`
Configuration for dataset processing.

**Parameters:**
- `model_name`: Embedding model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `model_type`: Model type - "auto", "sentence_transformers", or "onnx" (default: "auto")
- `max_samples`: Limit number of samples (None for all)
- `batch_size`: Processing batch size (default: 32)
- `db_path`: Database file path
- `use_quantized`: Use quantized model for ONNX (default: False)
- `skip_existing`: Skip existing entries (default: True)
- `cache_dir`: Directory to cache models (default: "./cache")

## Database Schema

### Documents Table
```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    paper_title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    passage_text TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at DATETIME NOT NULL,
    year INTEGER,
    venue TEXT,
    specialty TEXT,
    abstract TEXT,
    authors TEXT,
    doi TEXT
);
```

### Medical Q&A Table
```sql
CREATE TABLE medical_qa (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    vector BLOB NOT NULL,
    document_id TEXT,
    specialty TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic database operations
- `process_miriad.py`: MIRIAD dataset processing
- `sentence_transformers_example.py`: Using sentence-transformers models
- `search_examples.py`: Different search methods
- `batch_processing.py`: Batch processing examples

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=loma_dataset
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{loma_dataset,
  title={LOMA Dataset: Medical Q&A Vector Database},
  author={LOMA Dataset Team},
  year={2024},
  url={https://github.com/your-org/loma-dataset}
}
```