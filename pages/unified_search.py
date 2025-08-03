"""
Unified Search & Browse page for the Medical Q&A Database Explorer.

This page consolidates search and browse functionality with modern Streamlit components
optimized for large datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from datetime import datetime

try:
    from src.loma_dataset.models import MedicalSearchResult, MedicalDocument, MedicalQA, DocumentSearchResult
    from src.loma_dataset.exceptions import DatabaseError, ValidationError
except ImportError as e:
    st.error(f"Error importing modules: {e}")

logger = logging.getLogger(__name__)

# Configuration constants
RESULTS_PER_PAGE = 20
MAX_SIMILARITY_SEARCH_RESULTS = 100
MAX_TEXT_SEARCH_RESULTS = 200
MAX_SQL_RESULTS = 500

def show():
    """Display the unified search and browse interface."""
    st.header("Database Explorer")
    
    if st.session_state.db is None:
        st.error("Database not initialized.")
        return
    
    # Initialize session state for pagination
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "total_results" not in st.session_state:
        st.session_state.total_results = 0
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False
    if "results_per_page" not in st.session_state:
        st.session_state.results_per_page = RESULTS_PER_PAGE
    
    # Main interface with tabs
    search_tab, browse_tab, analytics_tab = st.tabs([
        "Search", 
        "Browse Data", 
        "Analytics"
    ])
    
    with search_tab:
        show_unified_search()
    
    with browse_tab:
        show_data_browser()
        
    with analytics_tab:
        show_quick_analytics()

def show_unified_search():
    """Show the unified search interface with multiple search modes."""
    st.subheader("Search")
    
    # Search mode selection
    search_mode = st.selectbox(
        "Search Mode:",
        ["Semantic Search", "Full-Text Search", "SQL Query", "Hybrid Search"],
        help="Choose your search approach based on your needs"
    )
    
    # Global filters in sidebar
    with st.sidebar:
        st.subheader("Search Filters")
        
        # Get available specialties
        try:
            stats = st.session_state.db.get_stats()
            specialties = ["All"] + sorted(stats.get('specialties', []))
        except:
            specialties = ["All"]
        
        specialty_filter = st.selectbox("Medical Specialty:", specialties)
        
        # Year range filter
        year_range = st.slider("Publication Year Range:", 1990, 2024, (2010, 2024))
        
        # Results configuration
        st.subheader("Results Settings")
        results_limit = st.slider("Max Results:", 10, 500, 50)
        
        if search_mode == "Semantic Search":
            similarity_threshold = st.slider("Similarity Threshold:", 0.0, 1.0, 0.65, 0.05)
        
        # Export options
        st.subheader("Export")
        if st.session_state.search_results and st.button("Export Results"):
            export_search_results()
    
    # Search interface based on mode
    if search_mode == "Semantic Search":
        show_semantic_search(specialty_filter, year_range, results_limit, similarity_threshold)
    elif search_mode == "Full-Text Search":
        show_fulltext_search(specialty_filter, year_range, results_limit)
    elif search_mode == "SQL Query":
        show_sql_search(results_limit)
    elif search_mode == "Hybrid Search":
        show_hybrid_search(specialty_filter, year_range, results_limit)
    
    # Display results with pagination
    if st.session_state.search_results or st.session_state.get('document_results'):
        display_paginated_results()

def show_semantic_search(specialty_filter, year_range, results_limit, similarity_threshold):
    """Semantic similarity search interface."""
    st.markdown("**Semantic Search**")
    st.info("Find similar medical content using AI-powered semantic understanding.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your medical query:",
            placeholder="e.g., What are the treatment options for type 2 diabetes in elderly patients?",
            height=120,
            help="Describe what you're looking for in natural language"
        )
    
    with col2:
        search_target = st.radio(
            "Search in:",
            ["Q&A Pairs", "Documents", "Both"],
            help="Choose what type of content to search"
        )
        
        include_context = st.checkbox("Include document context", value=True)
    
    if st.button("Search", type="primary", use_container_width=True):
        if query.strip():
            perform_semantic_search(
                query, specialty_filter, year_range, results_limit, 
                similarity_threshold, search_target, include_context
            )
        else:
            st.warning("Please enter a search query.")

def show_fulltext_search(specialty_filter, year_range, results_limit):
    """Full-text search interface."""
    st.markdown("**Full-Text Search**")
    st.info("Search using keywords and boolean operators for precise text matching.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search terms:",
            placeholder="diabetes AND (insulin OR metformin)",
            help="Use AND, OR, NOT operators. Use quotes for exact phrases."
        )
        
        # Advanced search options
        with st.expander("Advanced Options"):
            search_fields = st.multiselect(
                "Search in fields:",
                ["Questions", "Answers", "Document Titles", "Document Content"],
                default=["Questions", "Answers"]
            )
            
            exact_match = st.checkbox("Exact phrase matching")
            case_sensitive = st.checkbox("Case sensitive")
    
    with col2:
        sort_by = st.selectbox(
            "Sort results by:",
            ["Relevance", "Date", "Specialty"]
        )
        
        sort_order = st.radio("Order:", ["Descending", "Ascending"])
    
    if st.button("Search", type="primary", use_container_width=True):
        if query.strip():
            perform_fulltext_search(
                query, specialty_filter, year_range, results_limit,
                search_fields, exact_match, case_sensitive, sort_by, sort_order
            )
        else:
            st.warning("Please enter search terms.")

def show_sql_search(results_limit):
    """SQL query interface for advanced users."""
    st.markdown("**⚡ SQL Query**")
    st.info("Execute custom SQL queries directly on the database. Use with caution!")
    
    # Sample queries
    with st.expander("📋 Sample Queries"):
        sample_queries = {
            "All Q&As by specialty": """
SELECT mq.question, mq.answer, d.specialty, d.title
FROM medical_qa mq 
JOIN documents d ON mq.document_id = d.id 
WHERE d.specialty = 'Cardiology'
LIMIT 10;""",
            
            "Documents by year range": """
SELECT title, specialty, year, created_at
FROM documents 
WHERE year BETWEEN 2020 AND 2024
ORDER BY year DESC;""",
            
            "Q&A count by specialty": """
SELECT d.specialty, COUNT(mq.id) as qa_count
FROM documents d
LEFT JOIN medical_qa mq ON d.id = mq.document_id
GROUP BY d.specialty
ORDER BY qa_count DESC;"""
        }
        
        for name, query in sample_queries.items():
            if st.button(f"📝 {name}", key=f"sample_{name}"):
                st.session_state.sql_query = query
    
    # SQL input
    sql_query = st.text_area(
        "SQL Query:",
        value=st.session_state.get('sql_query', ''),
        height=150,
        placeholder="SELECT * FROM medical_qa LIMIT 10;",
        help="Available tables: medical_qa, documents"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Execute Query", type="primary"):
            if sql_query.strip():
                execute_sql_query(sql_query, results_limit)
            else:
                st.warning("Please enter a SQL query.")
    
    with col2:
        if st.button("🔧 Show Schema"):
            show_database_schema()

def show_hybrid_search(specialty_filter, year_range, results_limit):
    """Hybrid search combining multiple approaches."""
    st.markdown("**🔀 Hybrid Search**")
    st.info("Combine semantic similarity with keyword matching for best results.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        semantic_query = st.text_area(
            "Semantic query (natural language):",
            placeholder="Treatment approaches for cardiovascular disease",
            height=80
        )
        
        keyword_query = st.text_input(
            "Keywords (boost specific terms):",
            placeholder="hypertension, medication, lifestyle",
            help="Comma-separated keywords to boost in results"
        )
    
    with col2:
        semantic_weight = st.slider("Semantic weight:", 0.0, 1.0, 0.7, 0.1)
        keyword_weight = 1.0 - semantic_weight
        st.write(f"Keyword weight: {keyword_weight:.1f}")
        
        min_matches = st.selectbox("Min keyword matches:", [0, 1, 2, 3], index=1)
    
    if st.button("🔍 Hybrid Search", type="primary", use_container_width=True):
        if semantic_query.strip() or keyword_query.strip():
            perform_hybrid_search(
                semantic_query, keyword_query, specialty_filter, year_range,
                results_limit, semantic_weight, keyword_weight, min_matches
            )
        else:
            st.warning("Please enter at least one type of query.")

def perform_semantic_search(query, specialty_filter, year_range, results_limit, 
                          similarity_threshold, search_target, include_context):
    """Execute semantic similarity search."""
    if st.session_state.embedding_generator is None:
        st.error("Embedding generator not available. Semantic search is disabled.")
        return
    
    # Check if embedding generator is properly initialized
    if not hasattr(st.session_state.embedding_generator, '_actual_model_type') or st.session_state.embedding_generator._actual_model_type is None:
        st.error("Embedding generator is not properly initialized. Please restart the application.")
        return
    
    with st.spinner("Performing semantic search..."):
        try:
            # Validate query
            if not query or not query.strip():
                st.error("Please enter a search query.")
                return
            
            # Generate query embedding
            embeddings_result = st.session_state.embedding_generator.generate_embeddings([query.strip()])
            
            if not embeddings_result or len(embeddings_result) == 0:
                st.error("Failed to generate embedding for the query. The embedding generator returned no results.")
                return
                
            query_embedding = embeddings_result[0]
            
            # Validate the embedding
            if not query_embedding or len(query_embedding) == 0:
                st.error("Generated embedding is empty. The model may not be working correctly.")
                return
            
            # Additional validation - check if embedding is a list of numbers
            if not isinstance(query_embedding, (list, np.ndarray)) or len(query_embedding) < 100:
                st.error(f"Invalid embedding format. Got {type(query_embedding)} with length {len(query_embedding) if hasattr(query_embedding, '__len__') else 'unknown'}")
                return
            
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Search based on target
            results = []
            
            if search_target in ["Q&A Pairs", "Both"]:
                qa_results = st.session_state.db.search_similar_qa(
                    query_embedding,
                    limit=results_limit,
                    threshold=similarity_threshold,
                    specialty=None if specialty_filter == "All" else specialty_filter
                )
                results.extend(qa_results)
            
            if search_target in ["Documents", "Both"]:
                doc_results = st.session_state.db.search_similar_documents(
                    query_embedding,
                    limit=results_limit,
                    threshold=similarity_threshold,
                    specialty=None if specialty_filter == "All" else specialty_filter
                )
                # Store document results separately
                st.session_state.document_results = doc_results
                
                # If searching only documents, clear QA results
                if search_target == "Documents":
                    st.session_state.search_results = []
            
            # Filter by year range if applicable
            if year_range != (2010, 2024):
                results = filter_results_by_year(results, year_range)
            
            # Sort by similarity
            results.sort(key=lambda x: x.similarity, reverse=True)
            results = results[:results_limit]
            
            st.session_state.search_results = results
            st.session_state.total_results = len(results)
            st.session_state.search_performed = True
            st.session_state.current_page = 1
            
            # Display results summary
            total_found = len(results)
            if search_target in ["Documents", "Both"] and hasattr(st.session_state, 'document_results'):
                total_found += len(st.session_state.document_results)
            
            # Handle case when only documents are searched
            if search_target == "Documents":
                total_found = len(st.session_state.document_results) if hasattr(st.session_state, 'document_results') else 0
            
            if total_found > 0:
                st.success(f"Found {total_found} semantically similar entries")
            else:
                st.warning("No similar entries found. Try lowering the similarity threshold.")
                
        except Exception as e:
            st.error(f"Semantic search failed: {e}")
            logger.error(f"Semantic search error: {e}")

def perform_fulltext_search(query, specialty_filter, year_range, results_limit,
                          search_fields, exact_match, case_sensitive, sort_by, sort_order):
    """Execute full-text search."""
    with st.spinner("📝 Performing full-text search..."):
        try:
            results = st.session_state.db.search_qa_text(
                query,
                limit=results_limit,
                specialty=None if specialty_filter == "All" else specialty_filter
            )
            
            # Filter by year range if applicable
            if year_range != (2010, 2024):
                results = filter_results_by_year(results, year_range)
            
            st.session_state.search_results = results
            st.session_state.total_results = len(results)
            st.session_state.search_performed = True
            st.session_state.current_page = 1
            
            if results:
                st.success(f"Found {len(results)} text matches")
            else:
                st.warning("No matching entries found.")
                
        except Exception as e:
            st.error(f"Full-text search failed: {e}")
            logger.error(f"Full-text search error: {e}")

def execute_sql_query(sql_query, results_limit):
    """Execute custom SQL query."""
    with st.spinner("⚡ Executing SQL query..."):
        try:
            cursor = st.session_state.db.conn.cursor()
            
            # Add LIMIT if not present and not a SELECT COUNT query
            if "LIMIT" not in sql_query.upper() and "COUNT(" not in sql_query.upper():
                sql_query += f" LIMIT {results_limit}"
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            if rows:
                df = pd.DataFrame(rows, columns=columns)
                
                st.success(f"Query executed successfully. {len(rows)} rows returned.")
                
                # Display results in an interactive table
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=400
                )
                
                # Option to download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Query executed successfully but returned no results.")
                
        except Exception as e:
            st.error(f"SQL query failed: {e}")
            logger.error(f"SQL query error: {e}")

def perform_hybrid_search(semantic_query, keyword_query, specialty_filter, year_range,
                        results_limit, semantic_weight, keyword_weight, min_matches):
    """Execute hybrid search combining semantic and keyword approaches."""
    with st.spinner("🔀 Performing hybrid search..."):
        try:
            all_results = []
            
            # Semantic search component
            if semantic_query.strip() and st.session_state.embedding_generator:
                query_embedding = st.session_state.embedding_generator.generate_embeddings([semantic_query])[0]
                semantic_results = st.session_state.db.search_similar_qa(
                    query_embedding,
                    limit=results_limit * 2,  # Get more to combine with keyword results
                    threshold=0.3,  # Lower threshold for hybrid
                    specialty=None if specialty_filter == "All" else specialty_filter
                )
                
                # Weight semantic results
                for result in semantic_results:
                    result.similarity *= semantic_weight
                    all_results.append(result)
            
            # Keyword search component
            if keyword_query.strip():
                keyword_results = st.session_state.db.search_qa_text(
                    keyword_query,
                    limit=results_limit * 2,
                    specialty=None if specialty_filter == "All" else specialty_filter
                )
                
                # Weight keyword results (simulate similarity score)
                for result in keyword_results:
                    if not hasattr(result, 'similarity') or result.similarity == 0:
                        result.similarity = 0.8 * keyword_weight  # Default keyword score
                    else:
                        result.similarity *= keyword_weight
                    all_results.append(result)
            
            # Remove duplicates and combine scores
            unique_results = {}
            for result in all_results:
                key = result.qa.id
                if key in unique_results:
                    # Combine scores
                    unique_results[key].similarity += result.similarity
                else:
                    unique_results[key] = result
            
            # Convert back to list and sort
            combined_results = list(unique_results.values())
            combined_results.sort(key=lambda x: x.similarity, reverse=True)
            combined_results = combined_results[:results_limit]
            
            # Filter by year range if applicable
            if year_range != (2010, 2024):
                combined_results = filter_results_by_year(combined_results, year_range)
            
            st.session_state.search_results = combined_results
            st.session_state.total_results = len(combined_results)
            st.session_state.search_performed = True
            st.session_state.current_page = 1
            
            if combined_results:
                st.success(f"Found {len(combined_results)} hybrid search results")
            else:
                st.warning("No results found. Try adjusting your queries or filters.")
                
        except Exception as e:
            st.error(f"Hybrid search failed: {e}")
            logger.error(f"Hybrid search error: {e}")

def filter_results_by_year(results, year_range):
    """Filter search results by publication year range."""
    filtered_results = []
    for result in results:
        if result.document and result.document.year:
            if year_range[0] <= result.document.year <= year_range[1]:
                filtered_results.append(result)
        else:
            # Include results without year information
            filtered_results.append(result)
    return filtered_results

def display_paginated_results():
    """Display search results with pagination."""
    # Get QA results
    qa_results = st.session_state.search_results if st.session_state.search_results else []
    
    # Get document results if they exist
    doc_results = st.session_state.get('document_results', [])
    
    # Combine all results for pagination
    all_results = qa_results + doc_results
    
    if not all_results:
        return
    
    # Get current results per page setting
    current_results_per_page = st.session_state.get('results_per_page', RESULTS_PER_PAGE)
    
    total_results = len(all_results)
    total_pages = (total_results + current_results_per_page - 1) // current_results_per_page
    
    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("Previous", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col2:
        page_input = st.number_input(
            "Page:", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.current_page,
            key="page_input"
        )
        if page_input != st.session_state.current_page:
            st.session_state.current_page = page_input
            st.rerun()
    
    with col3:
        st.write(f"Page {st.session_state.current_page} of {total_pages} ({total_results} total results)")
    
    with col4:
        results_per_page = st.selectbox(
            "Per page:", 
            [10, 20, 50, 100], 
            index=1,
            key="results_per_page_selector"
        )
        if results_per_page != current_results_per_page:
            # Update session state and recalculate pagination
            st.session_state.results_per_page = results_per_page
            st.rerun()
    
    with col5:
        if st.button("Next", disabled=st.session_state.current_page >= total_pages):
            st.session_state.current_page += 1
            st.rerun()
    
    # Calculate slice for current page
    start_idx = (st.session_state.current_page - 1) * current_results_per_page
    end_idx = min(start_idx + current_results_per_page, total_results)
    current_results = all_results[start_idx:end_idx]
    
    # Display results
    for i, result in enumerate(current_results, start=start_idx + 1):
        if hasattr(result, 'qa'):  # QA result
            display_search_result(result, i)
        elif hasattr(result, 'document'):  # Document result
            display_document_result(result, i)
        else:
            st.error(f"Unknown result type: {type(result)}")

def display_document_result(result, index: int):
    """Display a single document search result in a clean, minimal expander."""
    
    if not isinstance(result, DocumentSearchResult):
        return
    
    # Create expander title with key information
    specialty = result.document.specialty if result.document.specialty else "General"
    year = f" ({result.document.year})" if result.document.year else ""
    similarity = f" | Score: {result.similarity:.3f}"
    
    expander_title = f"Document #{index} | {specialty}{year}{similarity}"
    
    with st.expander(expander_title, expanded=False):
        # Main content
        st.subheader("Document Title")
        st.write(f"**{result.document.title}**")
        
        st.subheader("Document Content")
        st.text_area("Content:", value=result.document.content, height=400, disabled=True, key=f"doc_content_{index}")
        
        # Metadata section
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Document Info**")
            st.text(f"ID: {result.document.id}")
            if result.document.url:
                st.text(f"URL: {result.document.url}")
            if result.document.created_at:
                # Handle both string and datetime objects
                if hasattr(result.document.created_at, 'strftime'):
                    created_date = result.document.created_at.strftime('%Y-%m-%d')
                else:
                    created_date = str(result.document.created_at)
                st.text(f"Created: {created_date}")
        
        with col2:
            st.write("**Metadata**")
            st.text(f"Specialty: {specialty}")
            if result.document.year:
                st.text(f"Year: {result.document.year}")
            st.text(f"Similarity: {result.similarity:.3f}")
        
        with col3:
            st.write("**Actions**")
            if st.button(f"View Details", key=f"view_doc_{index}"):
                show_document_details(result.document)

def display_search_result(result: MedicalSearchResult, index: int):
    """Display a single Q&A search result in a clean, minimal expander."""
    
    # Create expander title with key information
    specialty = result.document.specialty if result.document and result.document.specialty else "General"
    year = f" ({result.document.year})" if result.document and result.document.year else ""
    similarity = f" | Score: {result.similarity:.3f}" if hasattr(result, 'similarity') else ""
    
    expander_title = f"Q&A #{index} | {specialty}{year}{similarity}"
    
    with st.expander(expander_title, expanded=False):
        # Main content - Q&A result
        st.subheader("Question")
        st.write(f"**{result.qa.question}**")
        
        st.subheader("Answer")
        answer_preview = result.qa.answer[:500] + "..." if len(result.qa.answer) > 500 else result.qa.answer
        st.write(answer_preview)
        
        # Show full answer option
        if len(result.qa.answer) > 500:
            if st.button(f"Show Full Answer", key=f"show_full_qa_{index}"):
                st.text_area("Full Answer", result.qa.answer, height=300, key=f"full_qa_{index}")
        
        # Metadata section
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Q&A Info**")
            st.text(f"ID: {result.qa.id}")
            if result.qa.document_id:
                st.text(f"Doc ID: {result.qa.document_id}")
        
        with col2:
            st.write("**Metadata**")
            st.text(f"Specialty: {specialty}")
            if result.document and result.document.year:
                st.text(f"Year: {result.document.year}")
            st.text(f"Similarity: {result.similarity:.3f}")
        
        with col3:
            st.write("**Actions**")
            if st.button(f"View Details", key=f"view_qa_{index}"):
                show_result_details(result)

def show_document_details(document: MedicalDocument):
    """Show detailed view of a document in a clean, minimal format."""
    
    st.subheader("Document Details")
    
    # Document title
    st.write(f"**Document Title:** {document.title}")
    
    # Basic info in clean columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Specialty:** {document.specialty or 'N/A'}")
    with col2:
        st.write(f"**Year:** {document.year or 'N/A'}")
    with col3:
        st.write(f"**ID:** {document.id}")
    with col4:
        if document.url:
            st.markdown(f"**URL:** [Source]({document.url})")
        else:
            st.write("**URL:** N/A")
    
    # Document content - always show full content
    st.subheader("Document Content")
    st.text_area("Full Content:", value=document.content, height=400, disabled=True, key=f"doc_content_{document.id}")
    
    # Close button
    if st.button("Close Details", key=f"close_doc_{document.id}"):
        st.rerun()

def show_result_details(result: MedicalSearchResult):
    """Show detailed view of a Q&A search result in a clean, minimal format."""
    
    st.subheader("Q&A Details")
    
    # Q&A content
    st.write(f"**Question:** {result.qa.question}")
    st.write(f"**Answer:** {result.qa.answer}")
    
    # Basic info in clean columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Q&A ID:** {result.qa.id}")
    with col2:
        st.write(f"**Document ID:** {result.qa.document_id}")
    with col3:
        similarity = f"{result.similarity:.3f}" if hasattr(result, 'similarity') else "N/A"
        st.write(f"**Similarity:** {similarity}")
    with col4:
        if result.document and result.document.url:
            st.markdown(f"**URL:** [Source]({result.document.url})")
        else:
            st.write("**URL:** N/A")
    
    if result.document:
        # Document info
        st.subheader("Associated Document")
        st.write(f"**Document Title:** {result.document.title}")
        st.write(f"**Specialty:** {result.document.specialty or 'N/A'}")
        st.write(f"**Year:** {result.document.year or 'N/A'}")
        
        # Document content preview
        st.subheader("Document Content Preview")
        content_preview = result.document.content[:1000] + "..." if len(result.document.content) > 1000 else result.document.content
        st.text_area("Content:", value=content_preview, height=200, disabled=True, key=f"doc_preview_{result.qa.id}")
    
    # Close button
    if st.button("Close Details", key=f"close_{result.qa.id}"):
        st.rerun()

def show_data_browser():
    """Show the data browsing interface with advanced filtering."""
    st.subheader("📋 Data Browser")
    
    # Browser mode selection
    browse_mode = st.selectbox(
        "Browse Mode:",
        ["Q&A Entries", "Documents", "Statistics View"],
        help="Choose what type of data to browse"
    )
    
    if browse_mode == "Q&A Entries":
        show_qa_browser_advanced()
    elif browse_mode == "Documents":
        show_document_browser_advanced()
    else:
        show_statistics_browser()

def show_qa_browser_advanced():
    """Advanced Q&A browser with filtering and sorting."""
    st.markdown("**❓ Q&A Entries Browser**")
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                stats = st.session_state.db.get_stats()
                specialties = ["All"] + sorted(stats.get('specialties', []))
            except:
                specialties = ["All"]
            specialty_filter = st.selectbox("Specialty:", specialties)
        
        with col2:
            sort_by = st.selectbox("Sort by:", ["ID", "Question Length", "Answer Length", "Document ID"])
        
        with col3:
            sort_order = st.selectbox("Order:", ["Ascending", "Descending"])
        
        with col4:
            entries_per_page = st.selectbox("Entries per page:", [10, 25, 50, 100], index=1)
    
    # Search within browser
    search_filter = st.text_input("🔍 Filter entries (keywords):", placeholder="Search within loaded entries...")
    
    # Load and display entries
    if st.button("📋 Load Q&A Entries", type="primary"):
        load_qa_entries_paginated(specialty_filter, sort_by, sort_order, entries_per_page, search_filter)

def show_document_browser_advanced():
    """Advanced document browser with filtering and sorting."""
    st.markdown("**📄 Documents Browser**")
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                stats = st.session_state.db.get_stats()
                specialties = ["All"] + sorted(stats.get('specialties', []))
            except:
                specialties = ["All"]
            specialty_filter = st.selectbox("Specialty:", specialties, key="doc_specialty")
        
        with col2:
            year_range = st.slider("Year Range:", 1990, 2024, (2000, 2024))
        
        with col3:
            sort_by = st.selectbox("Sort by:", ["Title", "Year", "Created Date", "Specialty"])
        
        with col4:
            docs_per_page = st.selectbox("Docs per page:", [5, 10, 20, 50], index=1)
    
    # Load and display documents
    if st.button("📄 Load Documents", type="primary"):
        load_documents_paginated(specialty_filter, year_range, sort_by, docs_per_page)

def show_statistics_browser():
    """Show database statistics and insights."""
    st.markdown("**📊 Database Statistics**")
    
    try:
        stats = st.session_state.db.get_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Q&A Entries", stats.get("qa_count", 0))
        
        with col2:
            st.metric("Total Documents", stats.get("document_count", 0))
        
        with col3:
            st.metric("Medical Specialties", len(stats.get("specialties", [])))
        
        with col4:
            # Calculate average Q&As per document
            avg_qa_per_doc = stats.get("qa_count", 0) / max(stats.get("document_count", 1), 1)
            st.metric("Avg Q&As per Doc", f"{avg_qa_per_doc:.1f}")
        
        # Specialty distribution
        if stats.get("specialties"):
            st.subheader("📊 Specialty Distribution")
            
            cursor = st.session_state.db.conn.cursor()
            cursor.execute("""
                SELECT d.specialty, COUNT(mq.id) as qa_count, COUNT(DISTINCT d.id) as doc_count
                FROM documents d
                LEFT JOIN medical_qa mq ON d.id = mq.document_id
                WHERE d.specialty IS NOT NULL
                GROUP BY d.specialty
                ORDER BY qa_count DESC
            """)
            
            specialty_data = cursor.fetchall()
            if specialty_data:
                df = pd.DataFrame(specialty_data, columns=['Specialty', 'Q&A Count', 'Document Count'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(df.set_index('Specialty')['Q&A Count'])
                
                with col2:
                    st.dataframe(df, use_container_width=True)
        
        # Year distribution
        st.subheader("📅 Publication Year Distribution")
        cursor.execute("""
            SELECT year, COUNT(*) as count
            FROM documents
            WHERE year IS NOT NULL
            GROUP BY year
            ORDER BY year
        """)
        
        year_data = cursor.fetchall()
        if year_data:
            df_years = pd.DataFrame(year_data, columns=['Year', 'Count'])
            st.line_chart(df_years.set_index('Year'))
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

def show_quick_analytics():
    """Show quick analytics and insights."""
    st.subheader("📊 Quick Analytics")
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    
    try:
        cursor = st.session_state.db.conn.cursor()
        
        # Most recent documents
        cursor.execute("""
            SELECT title, specialty, created_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        recent_docs = cursor.fetchall()
        if recent_docs:
            st.markdown("**Recent Documents:**")
            for title, specialty, created_at in recent_docs:
                st.markdown(f"- {title} ({specialty}) - {created_at}")
        
        # Top specialties by content volume
        st.markdown("### 🏥 Top Medical Specialties")
        cursor.execute("""
            SELECT specialty, COUNT(*) as count
            FROM documents
            WHERE specialty IS NOT NULL
            GROUP BY specialty
            ORDER BY count DESC
            LIMIT 10
        """)
        
        top_specialties = cursor.fetchall()
        if top_specialties:
            df = pd.DataFrame(top_specialties, columns=['Specialty', 'Document Count'])
            st.bar_chart(df.set_index('Specialty'))
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def load_qa_entries_paginated(specialty_filter, sort_by, sort_order, entries_per_page, search_filter):
    """Load Q&A entries with pagination and filtering."""
    try:
        cursor = st.session_state.db.conn.cursor()
        
        # Build query based on filters
        base_query = """
            SELECT mq.id, mq.question, mq.answer, mq.document_id, d.specialty, d.title
            FROM medical_qa mq
            LEFT JOIN documents d ON mq.document_id = d.id
        """
        
        conditions = []
        params = []
        
        if specialty_filter != "All":
            conditions.append("d.specialty = ?")
            params.append(specialty_filter)
        
        if search_filter:
            conditions.append("(mq.question LIKE ? OR mq.answer LIKE ?)")
            params.extend([f"%{search_filter}%", f"%{search_filter}%"])
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add sorting
        sort_column_map = {
            "ID": "mq.id",
            "Question Length": "LENGTH(mq.question)",
            "Answer Length": "LENGTH(mq.answer)",
            "Document ID": "mq.document_id"
        }
        
        sort_column = sort_column_map.get(sort_by, "mq.id")
        sort_direction = "DESC" if sort_order == "Descending" else "ASC"
        
        base_query += f" ORDER BY {sort_column} {sort_direction}"
        base_query += f" LIMIT {entries_per_page}"
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        if results:
            df = pd.DataFrame(results, columns=[
                'ID', 'Question', 'Answer', 'Document ID', 'Specialty', 'Document Title'
            ])
            
            # Display with enhanced styling
            st.dataframe(
                df,
                use_container_width=True,
                height=600,
                column_config={
                    "Question": st.column_config.TextColumn(width="medium"),
                    "Answer": st.column_config.TextColumn(width="large"),
                    "ID": st.column_config.TextColumn(width="small"),
                    "Document ID": st.column_config.TextColumn(width="small"),
                    "Specialty": st.column_config.TextColumn(width="small"),
                }
            )
            
            st.success(f"Loaded {len(results)} Q&A entries")
            
            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"qa_entries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No entries found with the current filters.")
            
    except Exception as e:
        st.error(f"Error loading Q&A entries: {e}")

def load_documents_paginated(specialty_filter, year_range, sort_by, docs_per_page):
    """Load documents with pagination and filtering."""
    try:
        cursor = st.session_state.db.conn.cursor()
        
        # Build query
        base_query = "SELECT id, title, specialty, year, created_at FROM documents"
        conditions = []
        params = []
        
        if specialty_filter != "All":
            conditions.append("specialty = ?")
            params.append(specialty_filter)
        
        conditions.append("year BETWEEN ? AND ?")
        params.extend([year_range[0], year_range[1]])
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add sorting
        sort_column_map = {
            "Title": "title",
            "Year": "year",
            "Created Date": "created_at",
            "Specialty": "specialty"
        }
        
        sort_column = sort_column_map.get(sort_by, "title")
        base_query += f" ORDER BY {sort_column} DESC"
        base_query += f" LIMIT {docs_per_page}"
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        if results:
            df = pd.DataFrame(results, columns=['ID', 'Title', 'Specialty', 'Year', 'Created'])
            
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    "Title": st.column_config.TextColumn(width="large"),
                    "ID": st.column_config.TextColumn(width="small"),
                }
            )
            
            st.success(f"Loaded {len(results)} documents")
        else:
            st.warning("No documents found with the current filters.")
            
    except Exception as e:
        st.error(f"Error loading documents: {e}")

def show_database_schema():
    """Show the database schema information."""
    st.subheader("🏗️ Database Schema")
    
    try:
        cursor = st.session_state.db.conn.cursor()
        
        # Get table info
        tables = ["documents", "medical_qa"]
        
        for table in tables:
            st.markdown(f"### 📋 `{table}` Table")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            if columns:
                df = pd.DataFrame(columns, columns=['ID', 'Name', 'Type', 'NotNull', 'Default', 'PK'])
                st.dataframe(df[['Name', 'Type', 'NotNull', 'PK']], use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading schema: {e}")

def test_embedding_generator():
    """Test the embedding generator with a simple query."""
    if st.session_state.embedding_generator is None:
        st.error("Embedding generator not available.")
        return
    
    try:
        test_query = "diabetes treatment"
        st.info(f"Testing embedding generation with query: '{test_query}'")
        
        # Check initialization
        if not hasattr(st.session_state.embedding_generator, '_actual_model_type'):
            st.error("Model type not found in embedding generator.")
            return
        
        if st.session_state.embedding_generator._actual_model_type is None:
            st.error("Embedding generator not properly initialized.")
            return
        
        st.info(f"Using model type: {st.session_state.embedding_generator._actual_model_type}")
        
        # Try to generate embedding
        result = st.session_state.embedding_generator.generate_embeddings([test_query])
        
        if result and len(result) > 0:
            embedding = result[0]
            st.success(f"✅ Embedding generated successfully!")
            st.write(f"- Type: {type(embedding)}")
            st.write(f"- Length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
            st.write(f"- First 5 values: {embedding[:5] if hasattr(embedding, '__getitem__') else 'N/A'}")
        else:
            st.error("❌ Embedding generation returned empty result.")
            
    except Exception as e:
        st.error(f"❌ Embedding generation failed: {e}")
        st.write(f"Error type: {type(e).__name__}")

def export_search_results():
    """Export current search results to CSV."""
    if not st.session_state.search_results:
        st.warning("No search results to export.")
        return
    
    try:
        # Prepare data for export
        export_data = []
        for result in st.session_state.search_results:
            row = {
                'QA_ID': result.qa.id,
                'Question': result.qa.question,
                'Answer': result.qa.answer,
                'Document_ID': result.qa.document_id,
                'Similarity_Score': getattr(result, 'similarity', 0.0)
            }
            
            if result.document:
                row.update({
                    'Document_Title': result.document.title,
                    'Specialty': result.document.specialty,
                    'Year': result.document.year,
                    'URL': result.document.url
                })
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Search Results",
            data=csv,
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success(f"Prepared {len(export_data)} results for download!")
        
    except Exception as e:
        st.error(f"Error exporting results: {e}")