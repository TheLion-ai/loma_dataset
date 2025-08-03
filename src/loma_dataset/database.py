"""
Database operations for the LOMA Dataset package.

This module provides the MedicalVectorDB class for managing medical Q&A data
with vector search capabilities using SQLite/libSQL.
"""

import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

from .models import MedicalDocument, MedicalQA, MedicalSearchResult, DocumentSearchResult
from .exceptions import DatabaseError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration constants
EMBEDDING_DIMENSION = 384
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.65


class MedicalVectorDB:
    """
    Service for medical Q&A vector database operations using SQLite/libSQL.
    
    This class provides comprehensive database operations including:
    - Document and Q&A storage with vector embeddings
    - Vector similarity search
    - Full-text search capabilities
    - Database statistics and management
    """

    def __init__(self, db_path: str):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.is_initialized = False

    def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self.is_initialized:
            return

        try:
            logger.info(f"Initializing Medical Vector DB at {self.db_path}")
            
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables and indexes
            self._create_tables()
            self._create_indexes()
            
            self.is_initialized = True
            logger.info("Medical Vector DB initialized successfully")
            
        except Exception as error:
            logger.error(f"Failed to initialize Medical Vector DB: {error}")
            if self.conn:
                self.conn.close()
                self.conn = None
            self.is_initialized = False
            raise DatabaseError(f"Failed to initialize database: {error}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.is_initialized = False
            logger.info("Database connection closed")

    def optimize_database(self) -> None:
        """Perform database optimization for better performance with large datasets."""
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        try:
            cursor = self.conn.cursor()
            
            # Update statistics for query optimizer
            cursor.execute("ANALYZE")
            
            # Vacuum database to reclaim space and optimize
            cursor.execute("VACUUM")
            
            # Update index statistics
            cursor.execute("REINDEX")
            
            self.conn.commit()
            logger.info("Database optimization completed successfully")
            
        except Exception as error:
            logger.error(f"Error optimizing database: {error}")
            raise DatabaseError(f"Failed to optimize database: {error}")

    def get_database_size(self) -> Dict[str, Any]:
        """Get database size information for monitoring large datasets."""
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        try:
            cursor = self.conn.cursor()
            
            # Get page count and page size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            # Calculate total size
            total_size_bytes = page_count * page_size
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            # Get table sizes
            cursor.execute("""
                SELECT name, COUNT(*) as row_count 
                FROM sqlite_master 
                WHERE type='table' AND name IN ('documents', 'medical_qa')
            """)
            table_info = cursor.fetchall()
            
            return {
                "total_size_mb": round(total_size_mb, 2),
                "total_size_bytes": total_size_bytes,
                "page_count": page_count,
                "page_size": page_size,
                "tables": dict(table_info) if table_info else {}
            }
            
        except Exception as error:
            logger.error(f"Error getting database size: {error}")
            raise DatabaseError(f"Failed to get database size: {error}")

    def _create_tables(self) -> None:
        """Create the database tables."""
        if not self.conn:
            raise DatabaseError("Database connection not established")

        try:
            cursor = self.conn.cursor()

            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at DATETIME NOT NULL,
                    url TEXT,
                    year INTEGER,
                    specialty TEXT
                )
            """)

            # Create Q&A table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medical_qa (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    document_id TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            self.conn.commit()
            logger.info("Database tables created successfully")

        except Exception as error:
            logger.error(f"Error creating tables: {error}")
            raise DatabaseError(f"Failed to create database tables: {error}")

    def _create_indexes(self) -> None:
        """Create indexes for efficient querying and large dataset optimization."""
        if not self.conn:
            raise DatabaseError("Database connection not established")

        try:
            cursor = self.conn.cursor()

            # Create indexes for optimal performance with large datasets
            indexes = [
                "CREATE INDEX IF NOT EXISTS medical_qa_document_id_idx ON medical_qa (document_id)",
                "CREATE INDEX IF NOT EXISTS documents_specialty_idx ON documents (specialty)",
                "CREATE INDEX IF NOT EXISTS documents_year_idx ON documents (year)",
                "CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents (created_at)",
                # Composite indexes for complex queries
                "CREATE INDEX IF NOT EXISTS documents_specialty_year_idx ON documents (specialty, year)",
                "CREATE INDEX IF NOT EXISTS medical_qa_doc_specialty_idx ON medical_qa (document_id) WHERE document_id IS NOT NULL",
                # Primary key indexes (if not already exists)
                "CREATE INDEX IF NOT EXISTS medical_qa_id_idx ON medical_qa (id)",
                "CREATE INDEX IF NOT EXISTS documents_id_idx ON documents (id)"
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            # Create full-text search virtual tables
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS medical_qa_fts 
                USING fts5(question, answer, content='medical_qa')
            """)

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
                USING fts5(title, content, content='documents')
            """)

            # Apply performance optimizations for large datasets
            performance_settings = [
                "PRAGMA journal_mode = WAL",  # Write-Ahead Logging for better concurrency
                "PRAGMA synchronous = NORMAL",  # Balance between speed and durability
                "PRAGMA cache_size = -64000",  # 64MB cache
                "PRAGMA temp_store = MEMORY",  # Use memory for temporary tables
                "PRAGMA mmap_size = 268435456",  # 256MB memory map
                "ANALYZE"  # Update query planner statistics
            ]

            for pragma_sql in performance_settings:
                cursor.execute(pragma_sql)

            self.conn.commit()
            logger.info("Database indexes and performance optimizations applied successfully")

        except Exception as error:
            logger.error(f"Error creating indexes: {error}")
            raise DatabaseError(f"Failed to create database indexes: {error}")

    def add_document(self, document: MedicalDocument) -> None:
        """
        Add a medical document to the database.
        
        Args:
            document: MedicalDocument instance to add
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If document data is invalid
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        try:
            cursor = self.conn.cursor()
            
            # Convert vector to BLOB
            vector_blob = self._vector_to_blob(document.vector)

            cursor.execute("""
                INSERT OR IGNORE INTO documents 
                (id, title, content, vector, created_at, url, year, specialty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                document.id, document.title, document.content,
                vector_blob, document.created_at, document.url,
                document.year, document.specialty
            ])

            # Update full-text search index
            cursor.execute("""
                INSERT OR IGNORE INTO documents_fts (rowid, title, content)
                VALUES (
                    (SELECT rowid FROM documents WHERE id = ?), ?, ?
                )
            """, [document.id, document.title, document.content])

            self.conn.commit()
            logger.info(f"Document {document.id} added successfully")

        except Exception as error:
            logger.error(f"Error adding document: {error}")
            raise DatabaseError(f"Failed to add document: {error}")

    def add_qa(self, qa: MedicalQA) -> None:
        """
        Add a medical Q&A entry to the database.
        
        Args:
            qa: MedicalQA instance to add
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If Q&A data is invalid
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        try:
            cursor = self.conn.cursor()
            
            # Convert vector to BLOB
            vector_blob = self._vector_to_blob(qa.vector)

            cursor.execute("""
                INSERT OR REPLACE INTO medical_qa 
                (id, question, answer, vector, document_id)
                VALUES (?, ?, ?, ?, ?)
            """, [qa.id, qa.question, qa.answer, vector_blob, qa.document_id])

            # Update full-text search index
            cursor.execute("""
                INSERT OR REPLACE INTO medical_qa_fts (rowid, question, answer)
                VALUES (
                    (SELECT rowid FROM medical_qa WHERE id = ?), ?, ?
                )
            """, [qa.id, qa.question, qa.answer])

            self.conn.commit()
            logger.info(f"Q&A entry {qa.id} added successfully")

        except Exception as error:
            logger.error(f"Error adding Q&A entry: {error}")
            raise DatabaseError(f"Failed to add Q&A entry: {error}")

    def search_similar_qa(
        self,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        specialty: Optional[str] = None,
    ) -> List[MedicalSearchResult]:
        """
        Search for similar medical Q&A entries using vector similarity.
        
        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            specialty: Optional specialty filter
            
        Returns:
            List of MedicalSearchResult objects sorted by similarity
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If query parameters are invalid
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        try:
            if query_vector is None or len(query_vector) == 0:
                raise ValidationError("Vector cannot be empty")
        except (TypeError, ValueError):
            raise ValidationError("Invalid vector format")

        try:
            cursor = self.conn.cursor()
            
            # Build query with optional specialty filter
            query = """
                SELECT 
                    mq.id, mq.question, mq.answer, mq.vector, mq.document_id,
                    d.title, d.content, d.year, d.specialty, d.vector, d.created_at, d.url
                FROM medical_qa mq
                LEFT JOIN documents d ON mq.document_id = d.id
                WHERE mq.vector IS NOT NULL
            """
            
            params = []
            if specialty:
                query += " AND d.specialty = ?"
                params.append(specialty)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Calculate similarities and build results
            results = []
            query_np = np.array(query_vector, dtype=np.float32)

            for row in rows:
                try:
                    stored_vector = self._blob_to_vector(row[3])
                    stored_np = np.array(stored_vector, dtype=np.float32)
                    similarity = self._cosine_similarity(query_np, stored_np)

                    if similarity >= threshold:
                        qa = MedicalQA(
                            id=row[0], question=row[1], answer=row[2],
                            vector=stored_vector, document_id=row[4]
                        )

                        document = None
                        if row[5]:  # title exists
                            document = MedicalDocument(
                                id=row[4] or "", title=row[5], content=row[6],
                                vector=self._blob_to_vector(row[9]),
                                created_at=row[10], url=row[11], year=row[7], specialty=row[8]
                            )

                        results.append(MedicalSearchResult(qa=qa, similarity=similarity, document=document))

                except Exception as e:
                    logger.warning(f"Error processing row in similarity search: {e}")
                    continue

            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:limit]

        except Exception as error:
            logger.error(f"Error in similarity search: {error}")
            raise DatabaseError(f"Failed to perform similarity search: {error}")

    def search_similar_documents(
        self,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        specialty: Optional[str] = None,
    ) -> List[DocumentSearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            specialty: Optional specialty filter
            
        Returns:
            List of DocumentSearchResult objects sorted by similarity
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If query parameters are invalid
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")


        
        try:
            if query_vector is None:
                logger.error("Query vector is None")
                raise ValidationError("Vector cannot be empty")
            if len(query_vector) == 0:
                logger.error("Query vector has zero length")
                raise ValidationError("Vector cannot be empty")
        except (TypeError, ValueError) as e:
            logger.error(f"Vector validation error: {e}")
            raise ValidationError("Invalid vector format")

        try:
            cursor = self.conn.cursor()
            
            # Build query with optional specialty filter
            query = """
                SELECT id, title, content, vector, created_at, url, year, specialty
                FROM documents
                WHERE vector IS NOT NULL
            """
            
            params = []
            if specialty:
                query += " AND specialty = ?"
                params.append(specialty)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Calculate similarities and build results
            results = []
            query_np = np.array(query_vector, dtype=np.float32)

            for row in rows:
                try:
                    if row[3]:  # vector column
                        doc_vector = self._blob_to_vector(row[3])
                        doc_vector_np = np.array(doc_vector, dtype=np.float32)

                        # Calculate cosine similarity (optimized for L2 normalized vectors)
                        similarity = self._cosine_similarity(query_np, doc_vector_np)

                        if similarity >= threshold:
                            document = MedicalDocument(
                                id=row[0], title=row[1] or "", content=row[2] or "",
                                vector=doc_vector, created_at=row[4], url=row[5],
                                year=row[6], specialty=row[7]
                            )

                            results.append(DocumentSearchResult(document=document, similarity=similarity))

                except Exception as e:
                    logger.warning(f"Error processing row in document similarity search: {e}")
                    continue

            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:limit]

        except Exception as error:
            logger.error(f"Error in document similarity search: {error}")
            raise DatabaseError(f"Failed to perform document similarity search: {error}")

    def search_documents_text(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        specialty: Optional[str] = None,
    ) -> List[DocumentSearchResult]:
        """
        Search documents using full-text search.
        
        Args:
            query: Text query for full-text search
            limit: Maximum number of results to return
            specialty: Optional specialty filter
            
        Returns:
            List of DocumentSearchResult objects
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If query parameters are invalid
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        if not query or not query.strip():
            raise ValidationError("Query text is required")

        try:
            cursor = self.conn.cursor()
            
            # Build query with optional specialty filter
            sql_query = """
                SELECT d.id, d.title, d.content, d.vector, d.created_at, d.year, d.specialty, d.url
                FROM documents d
                JOIN documents_fts fts ON d.rowid = fts.rowid
                WHERE documents_fts MATCH ?
            """
            
            params = [query]
            if specialty:
                sql_query += " AND d.specialty = ?"
                params.append(specialty)
                
            sql_query += " ORDER BY rank LIMIT ?"
            params.append(limit)

            cursor.execute(sql_query, params)
            rows = cursor.fetchall()

            # Build results
            results = []
            for row in rows:
                try:
                    document = MedicalDocument(
                        id=row[0], title=row[1] or "", content=row[2] or "",
                        vector=self._blob_to_vector(row[3]) if row[3] else [],
                        created_at=row[4], year=row[5], specialty=row[6], url=row[7]
                    )
                    results.append(DocumentSearchResult(document=document, similarity=1.0))

                except Exception as e:
                    logger.warning(f"Error processing row in text search: {e}")
                    continue

            return results

        except Exception as error:
            logger.error(f"Error in text search: {error}")
            raise DatabaseError(f"Failed to perform text search: {error}")

    def search_qa_text(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        specialty: Optional[str] = None,
    ) -> List[MedicalSearchResult]:
        """
        Search Q&A entries using full-text search.
        
        Args:
            query: Text query for full-text search
            limit: Maximum number of results to return
            specialty: Optional specialty filter
            
        Returns:
            List of MedicalSearchResult objects
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If query parameters are invalid
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        if not query or not query.strip():
            raise ValidationError("Query text is required")

        try:
            cursor = self.conn.cursor()
            
            # Build query with optional specialty filter
            sql_query = """
                SELECT 
                    mq.id, mq.question, mq.answer, mq.vector, mq.document_id,
                    d.title, d.content, d.year, d.specialty, d.vector, d.created_at, d.url
                FROM medical_qa mq
                JOIN medical_qa_fts fts ON mq.rowid = fts.rowid
                LEFT JOIN documents d ON mq.document_id = d.id
                WHERE medical_qa_fts MATCH ?
            """
            
            params = [query]
            if specialty:
                sql_query += " AND d.specialty = ?"
                params.append(specialty)
                
            sql_query += " ORDER BY rank LIMIT ?"
            params.append(limit)

            cursor.execute(sql_query, params)
            rows = cursor.fetchall()

            # Build results
            results = []
            for row in rows:
                try:
                    # Only create QA object if we have valid vector data
                    if row[3]:  # vector column exists and is not None
                        vector = self._blob_to_vector(row[3])
                        if vector:  # vector is not empty
                            qa = MedicalQA(
                                id=row[0], question=row[1], answer=row[2],
                                vector=vector,
                                document_id=row[4]
                            )
                        else:
                            continue  # Skip this row if vector is empty
                    else:
                        continue  # Skip this row if no vector data

                    document = None
                    if row[5]:  # title exists
                        # Only create document if we have valid vector data
                        if row[9]:  # document vector column exists and is not None
                            doc_vector = self._blob_to_vector(row[9])
                            if doc_vector:  # vector is not empty
                                document = MedicalDocument(
                                    id=row[4] or "", title=row[5], content=row[6],
                                    vector=doc_vector,
                                    created_at=row[10], year=row[7], specialty=row[8], url=row[11]
                                )

                    results.append(MedicalSearchResult(qa=qa, similarity=1.0, document=document))

                except Exception as e:
                    logger.warning(f"Error processing row in Q&A text search: {e}")
                    continue

            return results

        except Exception as error:
            logger.error(f"Error in Q&A text search: {error}")
            raise DatabaseError(f"Failed to perform Q&A text search: {error}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        if not self.is_initialized or not self.conn:
            raise DatabaseError("Database not initialized")

        try:
            cursor = self.conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM medical_qa")
            total_qa = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            # Get specialties from documents only (medical_qa no longer has specialty)
            cursor.execute("SELECT DISTINCT specialty FROM documents WHERE specialty IS NOT NULL")
            doc_specialties = [row[0] for row in cursor.fetchall()]
            
            return {
                "qa_count": total_qa,
                "document_count": total_documents,
                "total_qa": total_qa,  # Keep for backward compatibility
                "total_documents": total_documents,  # Keep for backward compatibility
                "specialties": doc_specialties,
                "database_path": self.db_path
            }
            
        except Exception as error:
            logger.error(f"Error getting statistics: {error}")
            raise DatabaseError(f"Failed to get database statistics: {error}")

    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """Convert vector to BLOB for database storage."""
        return np.array(vector, dtype=np.float32).tobytes()

    def _blob_to_vector(self, blob: bytes) -> List[float]:
        """Convert BLOB back to vector."""
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Optimized for L2 normalized vectors where cosine similarity 
        reduces to dot product since ||a|| = ||b|| = 1.
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Handle edge cases
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # For L2 normalized vectors, this should be close to just dot_product
        # but we keep the full calculation for robustness
        similarity = float(dot_product / (norm_a * norm_b))
        
        # Clip to valid range [0.0, 1.0] to handle floating-point precision issues
        return max(0.0, min(1.0, similarity))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()