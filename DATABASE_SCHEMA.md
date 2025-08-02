# Medical Q&A Database Schema

This document describes the database schema for the Medical Q&A Vector Database system, which stores medical questions, answers, and research documents with vector embeddings for semantic search capabilities.

## Overview

The database uses SQLite with FTS5 (Full-Text Search) extensions and stores vector embeddings as BLOBs for efficient similarity search. The schema consists of two main tables with supporting indexes and virtual tables for full-text search.

## Core Tables

### 1. `documents` Table

Stores medical research papers and documents with their metadata and vector embeddings.

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at DATETIME NOT NULL,
    year INTEGER,
    specialty TEXT
);
```

#### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | TEXT | ✅ | Unique identifier for the document (Primary Key) |
| `title` | TEXT | ✅ | Title of the document |
| `content` | TEXT | ✅ | Full content of the document |
| `vector` | BLOB | ✅ | Vector embedding of the content |
| `created_at` | DATETIME | ✅ | Timestamp when record was created |
| `year` | INTEGER | ❌ | Publication year |
| `specialty` | TEXT | ❌ | Medical specialty (e.g., "Cardiology", "Oncology") |

### 2. `medical_qa` Table

Stores medical questions and answers with their vector embeddings and references to source documents.

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

#### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | TEXT | ✅ | Unique identifier for the Q&A pair (Primary Key) |
| `question` | TEXT | ✅ | Medical question |
| `answer` | TEXT | ✅ | Corresponding answer |
| `vector` | BLOB | ✅ | Vector embedding of the question |
| `document_id` | TEXT | ✅ | Reference to source document (Foreign Key) |

## Indexes

### Performance Indexes

```sql
-- Foreign key index for efficient joins
CREATE INDEX medical_qa_document_id_idx ON medical_qa (document_id);

-- Specialty filtering indexes
CREATE INDEX medical_qa_specialty_idx ON medical_qa (specialty);
CREATE INDEX documents_specialty_idx ON documents (specialty);

-- Document metadata indexes
CREATE INDEX documents_year_idx ON documents (year);
CREATE INDEX documents_venue_idx ON documents (venue);
```

### Full-Text Search Virtual Tables

```sql
-- Q&A full-text search
CREATE VIRTUAL TABLE medical_qa_fts 
USING fts5(question, answer, content='medical_qa');

-- Documents full-text search
CREATE VIRTUAL TABLE documents_fts 
USING fts5(paper_title, passage_text, abstract, content='documents');
```

## Data Classes

### MedicalDocument

Python dataclass representing a medical research document:

```python
@dataclass
class MedicalDocument:
    """Represents a medical research document/paper"""
    
    id: str
    paper_title: str
    source_url: str
    passage_text: str
    vector: List[float]
    created_at: str
    year: Optional[int] = None
    venue: Optional[str] = None
    specialty: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
```

### MedicalQA

Python dataclass representing a medical Q&A entry:

```python
@dataclass
class MedicalQA:
    """Represents a medical Q&A entry with vector embedding"""
    
    id: str
    question: str
    answer: str
    vector: List[float]
    document_id: Optional[str] = None
    specialty: Optional[str] = None
    created_at: Optional[str] = None
```

### Search Result Classes

```python
@dataclass
class MedicalSearchResult:
    """Result of a medical Q&A similarity search"""
    
    qa: MedicalQA
    similarity: float
    document: Optional[MedicalDocument] = None

@dataclass
class DocumentSearchResult:
    """Result of a document similarity search"""
    
    document: MedicalDocument
    similarity: float
```

## Vector Embeddings

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Storage**: BLOB format (converted from float32 array)

### Embedding Content
- **Q&A Vectors**: Generated from concatenated "Question: {question} Answer: {answer}" text
- **Document Vectors**: Generated from `passage_text` field content

### Vector Operations
- **Similarity Metric**: Cosine similarity
- **Search Threshold**: Configurable (default: 0.0)
- **Storage Format**: Float32 array serialized to BLOB

## Relationships

```
documents (1) ←→ (0..n) medical_qa
    ↑                      ↑
    └── document_id ───────┘
```

- One document can have multiple Q&A entries
- Q&A entries can optionally reference a source document
- Foreign key constraint with `ON DELETE SET NULL`

## Search Capabilities

### 1. Vector Similarity Search
- **Q&A Search**: Find similar questions/answers using vector embeddings
- **Document Search**: Find similar documents using passage text embeddings
- **Specialty Filtering**: Filter results by medical specialty
- **Similarity Scoring**: Cosine similarity with configurable thresholds

### 2. Full-Text Search
- **Q&A Search**: Search questions and answers using FTS5
- **Document Search**: Search paper titles, abstracts, and passage text
- **Boolean Operators**: Support for AND, OR, NOT operations
- **Phrase Matching**: Exact phrase search with quotes

### 3. Hybrid Search
- Combine vector similarity with full-text search
- Filter by metadata (specialty, year, venue)
- Sort by relevance or similarity scores

## Usage Examples

### Database Initialization
```python
from db import MedicalVectorDB

db = MedicalVectorDB("medical_qa_db.db")
db.initialize()
```

### Adding Data
```python
# Add a document
document = MedicalDocument(
    id="doc_001",
    paper_title="Diabetes Management Guidelines",
    source_url="https://example.com/diabetes-guidelines",
    passage_text="Diabetes management requires comprehensive care including diet, exercise, and medication monitoring.",
    vector=embedding_model.encode("Diabetes management requires comprehensive care...").tolist(),
    created_at="2023-01-15T10:30:00",
    specialty="Endocrinology",
    year=2023
)
db.add_document(document)

# Add Q&A
qa = MedicalQA(
    id="qa_001",
    question="What are the symptoms of diabetes?",
    answer="Common symptoms include increased thirst, frequent urination, and unexplained weight loss.",
    vector=embedding_model.encode("Question: What are the symptoms of diabetes? Answer: Common symptoms include increased thirst...").tolist(),
    document_id="doc_001",
    specialty="Endocrinology"
)
db.add_qa(qa)
```

### Searching
```python
# Vector search for Q&A
query_vector = embedding_model.encode("diabetes treatment")
results = db.search_similar_qa(query_vector, limit=10)

# Vector search for documents
doc_results = db.search_similar_documents(query_vector, limit=5)

# Full-text search
text_results = db.search_full_text("insulin therapy", specialty="Endocrinology")
```

## Performance Considerations

### Indexing Strategy
- Primary keys for fast lookups
- Foreign key indexes for efficient joins
- Specialty indexes for filtered searches
- FTS5 indexes for text search

### Vector Storage
- BLOB storage for efficient space usage
- In-memory vector operations for similarity calculations
- Batch processing for large datasets

### Query Optimization
- Use specialty filters to reduce search space
- Implement similarity thresholds to limit results
- Consider pagination for large result sets

## Data Sources

The database is populated from the MIRIAD (Medical Information Retrieval in Intensive Care and Acute Medicine Dataset) which contains:
- Medical research papers with abstracts and full text
- Question-answer pairs derived from medical literature
- Specialty classifications for medical domains
- Metadata including publication years, venues, and authors