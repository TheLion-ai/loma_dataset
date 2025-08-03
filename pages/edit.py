"""
Edit page for the Medical Q&A Database Explorer
"""

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def show():
    """Display the data editing interface."""
    st.header("📝 Edit Database Entries")
    
    if st.session_state.db is None:
        st.error("Database not initialized.")
        return
    
    # Toggle edit mode
    st.session_state.edit_mode = st.checkbox("Enable Edit Mode", value=st.session_state.edit_mode)
    
    if not st.session_state.edit_mode:
        st.info("Enable edit mode to modify database entries.")
        return
    
    st.warning("⚠️ Edit mode is enabled. Changes will directly modify the database.")
    
    # Edit tabs
    tab1, tab2 = st.tabs(["✏️ Edit Q&A", "📄 Edit Documents"])
    
    with tab1:
        show_qa_editor()
    
    with tab2:
        show_document_editor()

def show_qa_editor():
    """Show Q&A entry editor."""
    st.subheader("Edit Q&A Entries")
    
    # Search for Q&A entry to edit
    qa_id = st.text_input("Enter Q&A ID to edit:")
    
    if qa_id and st.button("Load Q&A Entry"):
        try:
            cursor = st.session_state.db.conn.cursor()
            cursor.execute(
                "SELECT id, question, answer, document_id FROM medical_qa WHERE id = ?",
                [qa_id]
            )
            result = cursor.fetchone()
            
            if result:
                st.session_state.current_qa = {
                    'id': result[0],
                    'question': result[1],
                    'answer': result[2],
                    'document_id': result[3]
                }
                st.success("Q&A entry loaded successfully!")
            else:
                st.error("Q&A entry not found.")
                
        except Exception as e:
            st.error(f"Error loading Q&A entry: {e}")
    
    # Edit form
    if hasattr(st.session_state, 'current_qa'):
        with st.form("edit_qa_form"):
            st.write(f"**Editing Q&A ID:** {st.session_state.current_qa['id']}")
            
            new_question = st.text_area(
                "Question:",
                value=st.session_state.current_qa['question'],
                height=100
            )
            
            new_answer = st.text_area(
                "Answer:",
                value=st.session_state.current_qa['answer'],
                height=150
            )
            
            new_document_id = st.text_input(
                "Document ID:",
                value=st.session_state.current_qa['document_id']
            )
            
            if st.form_submit_button("💾 Update Q&A Entry"):
                try:
                    # Generate new embedding if text changed
                    if (new_question != st.session_state.current_qa['question'] or 
                        new_answer != st.session_state.current_qa['answer']):
                        
                        if st.session_state.embedding_generator:
                            embedding_text = f"Question: {new_question} Answer: {new_answer}"
                            new_vector = st.session_state.embedding_generator.generate_embeddings([embedding_text])[0]
                            vector_blob = st.session_state.db._vector_to_blob(new_vector)
                        else:
                            st.warning("Embedding generator not available. Keeping existing vector.")
                            cursor = st.session_state.db.conn.cursor()
                            cursor.execute("SELECT vector FROM medical_qa WHERE id = ?", [qa_id])
                            vector_blob = cursor.fetchone()[0]
                    else:
                        # Keep existing vector
                        cursor = st.session_state.db.conn.cursor()
                        cursor.execute("SELECT vector FROM medical_qa WHERE id = ?", [qa_id])
                        vector_blob = cursor.fetchone()[0]
                    
                    # Update database
                    cursor = st.session_state.db.conn.cursor()
                    cursor.execute("""
                        UPDATE medical_qa 
                        SET question = ?, answer = ?, document_id = ?, vector = ?
                        WHERE id = ?
                    """, [new_question, new_answer, new_document_id, vector_blob, qa_id])
                    
                    # Update FTS index
                    cursor.execute("""
                        UPDATE medical_qa_fts 
                        SET question = ?, answer = ?
                        WHERE rowid = (SELECT rowid FROM medical_qa WHERE id = ?)
                    """, [new_question, new_answer, qa_id])
                    
                    st.session_state.db.conn.commit()
                    st.success("Q&A entry updated successfully!")
                    
                    # Update session state
                    st.session_state.current_qa = {
                        'id': qa_id,
                        'question': new_question,
                        'answer': new_answer,
                        'document_id': new_document_id
                    }
                    
                except Exception as e:
                    st.error(f"Error updating Q&A entry: {e}")
                    logger.error(f"Update error: {e}")

def show_document_editor():
    """Show document editor."""
    st.subheader("Edit Documents")
    
    # Search for document to edit
    doc_id = st.text_input("Enter Document ID to edit:")
    
    if doc_id and st.button("Load Document"):
        try:
            cursor = st.session_state.db.conn.cursor()
            cursor.execute(
                "SELECT id, title, content, specialty, year FROM documents WHERE id = ?",
                [doc_id]
            )
            result = cursor.fetchone()
            
            if result:
                st.session_state.current_doc = {
                    'id': result[0],
                    'title': result[1],
                    'content': result[2],
                    'specialty': result[3],
                    'year': result[4]
                }
                st.success("Document loaded successfully!")
            else:
                st.error("Document not found.")
                
        except Exception as e:
            st.error(f"Error loading document: {e}")
    
    # Edit form
    if hasattr(st.session_state, 'current_doc'):
        with st.form("edit_doc_form"):
            st.write(f"**Editing Document ID:** {st.session_state.current_doc['id']}")
            
            new_title = st.text_area(
                "Title:",
                value=st.session_state.current_doc['title'],
                height=60
            )
            
            new_content = st.text_area(
                "Content:",
                value=st.session_state.current_doc['content'],
                height=200
            )
            
            new_specialty = st.text_input(
                "Specialty:",
                value=st.session_state.current_doc['specialty'] or ""
            )
            
            new_year = st.number_input(
                "Year:",
                value=st.session_state.current_doc['year'] if st.session_state.current_doc['year'] else 2000,
                min_value=1900,
                max_value=2030,
                step=1
            )
            
            if st.form_submit_button("💾 Update Document"):
                try:
                    # Generate new embedding if content changed
                    if new_content != st.session_state.current_doc['content']:
                        if st.session_state.embedding_generator:
                            new_vector = st.session_state.embedding_generator.generate_embeddings([new_content])[0]
                            vector_blob = st.session_state.db._vector_to_blob(new_vector)
                        else:
                            st.warning("Embedding generator not available. Keeping existing vector.")
                            cursor = st.session_state.db.conn.cursor()
                            cursor.execute("SELECT vector FROM documents WHERE id = ?", [doc_id])
                            vector_blob = cursor.fetchone()[0]
                    else:
                        # Keep existing vector
                        cursor = st.session_state.db.conn.cursor()
                        cursor.execute("SELECT vector FROM documents WHERE id = ?", [doc_id])
                        vector_blob = cursor.fetchone()[0]
                    
                    # Update database
                    cursor = st.session_state.db.conn.cursor()
                    cursor.execute("""
                        UPDATE documents 
                        SET title = ?, content = ?, specialty = ?, year = ?, vector = ?
                        WHERE id = ?
                    """, [new_title, new_content, new_specialty or None, new_year, vector_blob, doc_id])
                    
                    # Update FTS index
                    cursor.execute("""
                        UPDATE documents_fts 
                        SET title = ?, content = ?
                        WHERE rowid = (SELECT rowid FROM documents WHERE id = ?)
                    """, [new_title, new_content, doc_id])
                    
                    st.session_state.db.conn.commit()
                    st.success("Document updated successfully!")
                    
                    # Update session state
                    st.session_state.current_doc = {
                        'id': doc_id,
                        'title': new_title,
                        'content': new_content,
                        'specialty': new_specialty,
                        'year': new_year
                    }
                    
                except Exception as e:
                    st.error(f"Error updating document: {e}")
                    logger.error(f"Document update error: {e}")