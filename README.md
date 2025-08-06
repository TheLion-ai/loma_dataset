# LOMA Dataset - Medical Q&A Vector Database

A comprehensive Python package for managing medical Q&A data with vector search capabilities, specifically designed as the database backend for the [LOMA (Local Offline Medical Assistant)](https://github.com/TheLion-ai/LOMA) mobile application. <mcreference link="https://github.com/TheLion-ai/LOMA" index="0">0</mcreference>

This package processes and serves the [MIRIAD medical dataset](https://huggingface.co/datasets/miriad/miriad-5.8M) containing 5.8M medical Q&A pairs, providing a robust foundation for medical AI applications. <mcreference link="https://huggingface.co/datasets/miriad/miriad-5.8M" index="1">1</mcreference>

## Features

- 🏥 **Medical-focused**: Optimized for medical Q&A data and terminology from MIRIAD dataset
- 🔍 **Vector Search**: Advanced similarity search using 384-dimensional embeddings with cosine similarity
- 📊 **SQLite Database**: Efficient storage with vector search capabilities and full-text indexing
- 🚀 **Sentence Transformers**: Integration with sentence-transformers models (all-MiniLM-L6-v2)
- 📱 **LOMA Integration**: Designed as the backend database for LOMA mobile medical assistant
- 🌐 **Streamlit App**: Interactive web interface for database exploration and management
- 📦 **Easy to Use**: Simple API for data processing, search, and retrieval
- 🔧 **Configurable**: Flexible configuration for different embedding models and processing options
- ⚡ **Optimized Performance**: Database optimizations for handling large medical datasets (500k+ entries)

## Installation

```bash
# Clone the repository
git clone https://github.com/TheLion-ai/LOMA.git
cd LOMA/loma_dataset

# Install the package
pip install -e .

# Install with all dependencies (recommended)
pip install sentence-transformers streamlit plotly pandas numpy
```

## Quick Start

### 1. Download Pre-built Database

The easiest way to get started is to download a pre-built database:

```python
# Download the MIRIAD database (500k entries)
python examples/download_db.py
```

### 2. Run the Streamlit Web App

Launch the interactive web interface to explore the medical database:

```bash
streamlit run app.py
```

This will start a web application at `http://localhost:8501` with the following features:
- **Dashboard**: Overview of database statistics and medical specialties
- **Search & Browse**: Semantic search, full-text search, and SQL queries
- **Document Submission**: Add new medical documents to the database
- **Analytics**: Database insights and specialty distributions

### 3. Basic Python API Usage

```python
from loma_dataset import MedicalVectorDB, MedicalDocument, MedicalQA
from datetime import datetime

# Initialize database
db = MedicalVectorDB("miriad_medical_minlm.db")
db.initialize()

# Add a document
document = MedicalDocument(
    id="doc_001",
    title="Hypertension Management",
    content="Hypertension is a common cardiovascular condition...",
    vector=[0.1] * 384,  # 384-dimensional embedding vector
    created_at=datetime.now(),
    url="https://example.com/hypertension",
    year=2024,
    specialty="Cardiology"
)
db.add_document(document)

# Add Q&A
qa = MedicalQA(
    id="qa_001",
    question="What is hypertension?",
    answer="Hypertension is high blood pressure that can lead to serious health complications.",
    vector=[0.2] * 384,  # 384-dimensional embedding vector
    document_id="doc_001"
)
db.add_qa(qa)

# Search similar Q&A entries
results = db.search_similar_qa(query_vector, limit=5, threshold=0.65)
for result in results:
    print(f"Q: {result.qa.question}")
    print(f"A: {result.qa.answer}")
    print(f"Similarity: {result.similarity:.3f}")

# Get database statistics
stats = db.get_stats()
print(f"Total Q&A entries: {stats['qa_count']}")
print(f"Total documents: {stats['document_count']}")
print(f"Medical specialties: {len(stats['specialties'])}")
```

### 4. Processing MIRIAD Dataset from Scratch

To process the full MIRIAD dataset and create your own database:

```python
from loma_dataset import MiriadProcessor, ProcessingConfig

# Configure processing with sentence-transformers
config = ProcessingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_type="sentence_transformers",
    max_samples=500000,  # Process 500k samples (or None for all 5.8M)
    batch_size=256,
    db_path="miriad_medical_minlm.db",
    cache_dir="./cache"
)

# Initialize processor and database
processor = MiriadProcessor(config)
processor.initialize()

# Load and process the MIRIAD dataset
dataset = processor.load_dataset()
print(f"Loaded {len(dataset)} entries from MIRIAD dataset")

# Process with progress tracking
from alive_progress import alive_bar
with alive_bar(len(dataset), title="Processing MIRIAD") as bar:
    for item in dataset:
        processor.populate_database(db, [item])
        bar()

# Get final statistics
stats = db.get_stats()
print(f"Processed: {stats['qa_count']} Q&A entries")
print(f"Documents: {stats['document_count']}")
print(f"Specialties: {len(stats['specialties'])}")
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

## Streamlit Web Application

The package includes a comprehensive Streamlit web application for database exploration and management:

### Features

- **📊 Dashboard**: Database overview with statistics and specialty distributions
- **🔍 Search & Browse**: Multiple search modes including:
  - Semantic search using vector embeddings
  - Full-text search with boolean operators
  - SQL query interface for advanced users
  - Hybrid search combining multiple approaches
- **📄 Document Submission**: Add new medical documents to the database
- **📈 Analytics**: Database insights and medical specialty analysis

### Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## API Reference

### Core Classes

#### `MedicalVectorDB`
Main database interface for storing and retrieving medical data with vector search capabilities.

**Key Methods:**
- `initialize()`: Initialize database and create optimized tables/indexes
- `add_document(document)`: Add a medical document with vector embedding
- `add_qa(qa)`: Add a Q&A entry with vector embedding
- `search_similar_qa(vector, limit, threshold, specialty)`: Vector similarity search for Q&A
- `search_similar_documents(vector, limit, threshold, specialty)`: Vector similarity search for documents
- `search_documents_text(query, limit, specialty)`: Full-text search in documents
- `get_stats()`: Get comprehensive database statistics
- `optimize_database()`: Optimize database performance for large datasets
- `get_database_size()`: Get database size information

#### `MedicalDocument`
Represents a medical research document with vector embedding.

**Required Fields:**
- `id`: Unique identifier
- `title`: Document title
- `content`: Full document content
- `vector`: 384-dimensional embedding vector
- `created_at`: Creation timestamp
- `url`: Source URL (optional)
- `year`: Publication year (optional)
- `specialty`: Medical specialty (optional)

#### `MedicalQA`
Represents a medical Q&A entry with vector embedding.

**Required Fields:**
- `id`: Unique identifier
- `question`: Medical question
- `answer`: Corresponding answer
- `vector`: 384-dimensional embedding vector
- `document_id`: Reference to source document

#### `MiriadProcessor`
Processes MIRIAD dataset and generates embeddings using sentence-transformers.

**Key Methods:**
- `initialize()`: Setup processor and embedding models
- `load_dataset()`: Load MIRIAD dataset from Hugging Face
- `populate_database(db, items)`: Process and add items to database
- `get_statistics()`: Get processing statistics

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

The database uses SQLite with optimized indexes for vector similarity search and large dataset performance.

### Documents Table
```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at DATETIME NOT NULL,
    url TEXT,
    year INTEGER,
    specialty TEXT
);
```

### Medical Q&A Table
```sql
CREATE TABLE medical_qa (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    vector BLOB NOT NULL,
    document_id TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

### Performance Optimizations

The database includes several optimizations for handling large medical datasets:

- **Vector Storage**: Efficient BLOB storage for 384-dimensional embeddings
- **Indexes**: Optimized indexes on document_id, specialty, and year fields
- **WAL Mode**: Write-Ahead Logging for better concurrency
- **Memory Mapping**: 256MB memory map for faster access
- **Cache Configuration**: 64MB cache for improved query performance

## Examples

The `examples/` directory contains comprehensive usage examples:

- **`basic_usage.py`**: Basic database operations and API usage
- **`download_db.py`**: Download pre-built MIRIAD database from cloud storage
- **`process_miriad.py`**: Complete MIRIAD dataset processing pipeline
- **`sentence_transformers_example.py`**: Using sentence-transformers for embeddings
- **`upload_db.py`**: Upload processed database to cloud storage

### Running Examples

```bash
# Download pre-built database (recommended for quick start)
python examples/download_db.py

# Process MIRIAD dataset from scratch (requires significant time/resources)
python examples/process_miriad.py

# Basic API usage demonstration
python examples/basic_usage.py

# Sentence transformers integration example
python examples/sentence_transformers_example.py
```

## LOMA Integration

This package serves as the database backend for the [LOMA (Local Offline Medical Assistant)](https://github.com/TheLion-ai/LOMA) mobile application. <mcreference link="https://github.com/TheLion-ai/LOMA" index="0">0</mcreference> LOMA is a React Native medical AI assistant that:

- **Runs Completely Offline**: All AI processing happens locally on mobile devices
- **Uses Gemma 3n Model**: Lightweight language model optimized for mobile inference
- **Provides RAG Capabilities**: Retrieval-Augmented Generation using this medical database
- **Supports iOS and Android**: Cross-platform mobile medical assistance
- **Maintains Privacy**: No data leaves the device, ensuring HIPAA-friendly operation

The database created by this package is downloaded and used by LOMA for:
- Semantic search through medical knowledge
- Context injection for AI responses
- Source citation for medical information
- Specialty-specific medical queries

## MIRIAD Dataset

This package processes the [MIRIAD (Medical Information Retrieval in Argumentative Dialogues) dataset](https://huggingface.co/datasets/miriad/miriad-5.8M), which contains: <mcreference link="https://huggingface.co/datasets/miriad/miriad-5.8M" index="1">1</mcreference>

- **5.8 Million Q&A Pairs**: Comprehensive medical question-answer pairs
- **54 Medical Specialties**: Coverage across all major medical fields including cardiology, oncology, neurology, etc.
- **Research Paper Sources**: Q&A pairs derived from peer-reviewed medical literature
- **Structured Format**: Consistent format with questions, answers, paper metadata, and specialty classifications
- **Quality Controlled**: Curated from reputable medical sources and publications

### Dataset Statistics
- **Total Entries**: 5.82M medical Q&A pairs
- **Specialties**: 54 different medical specialties
- **Time Range**: Publications from 1990-2024
- **Languages**: Primarily English medical content
- **Vector Dimensions**: 384-dimensional embeddings using all-MiniLM-L6-v2

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
  author={TheLion.ai},
  year={2024},
  url={https://github.com/TheLion-ai/LOMA}
}
```

If you use the MIRIAD dataset, please also cite:

```bibtex
@dataset{miriad_dataset,
  title={MIRIAD: Medical Information Retrieval in Argumentative Dialogues},
  author={MIRIAD Team},
  year={2024},
  url={https://huggingface.co/datasets/miriad/miriad-5.8M}
}
```

## Related Projects

- **[LOMA Mobile App](https://github.com/TheLion-ai/LOMA)**: React Native medical AI assistant using this database
- **[MIRIAD Dataset](https://huggingface.co/datasets/miriad/miriad-5.8M)**: Source medical Q&A dataset
- **[Sentence Transformers](https://www.sbert.net/)**: Embedding models used for vector search