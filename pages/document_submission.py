"""Document Submission page for the Medical Q&A Database Explorer"""

import streamlit as st
import base64
import json
import uuid
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging


# Configure logging
logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.loma_dataset.models import MedicalDocument, MedicalQA
    from src.loma_dataset.exceptions import DatabaseError, ValidationError
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Import LiteLLM for API calls
try:
    import litellm
except ImportError:
    st.error("LiteLLM not installed. Please install with: pip install litellm")
    st.stop()

# Constants
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
SUPPORTED_FILE_TYPES = ['pdf']
SUBMISSIONS_DB_PATH = "submissions.db"


class SubmissionDatabase:
    """Database handler for document submissions."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the submissions database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self._create_tables()
        except Exception as e:
            logger.error(f"Failed to initialize submissions database: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    def _create_tables(self):
        """Create the submissions database tables."""
        cursor = self.conn.cursor()
        
        # Create submissions documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submission_documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                vector BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT CHECK (source_type IN ('pdf', 'text')) NOT NULL,
                original_filename TEXT,
                processing_metadata TEXT,
                status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
            )
        """)
        
        # Create submissions Q&A table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submission_qa (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                vector BLOB,
                document_id TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'edited')),
                confidence_score REAL DEFAULT 0.0,
                FOREIGN KEY (document_id) REFERENCES submission_documents(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submission_qa_document_id ON submission_qa(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submission_qa_status ON submission_qa(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submission_documents_status ON submission_documents(status)")
        
        self.conn.commit()
    
    def save_document(self, doc_id: str, title: str, content: str, source_type: str, 
                     original_filename: Optional[str] = None) -> bool:
        """Save a document to the submissions database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO submission_documents 
                (id, title, content, source_type, original_filename, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (doc_id, title, content, source_type, original_filename))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            return False
    
    def save_qa_pairs(self, qa_pairs: List[Dict], document_id: str) -> bool:
        """Save Q&A pairs to the submissions database."""
        try:
            cursor = self.conn.cursor()
            for qa in qa_pairs:
                qa_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO submission_qa 
                    (id, question, answer, document_id, confidence_score, status)
                    VALUES (?, ?, ?, ?, ?, 'approved')
                """, (qa_id, qa['question'], qa['answer'], document_id, 
                      qa.get('confidence_score', 0.0)))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving Q&A pairs: {e}")
            return False
    
    def get_document_qa_pairs(self, document_id: str) -> List[Dict]:
        """Get Q&A pairs for a document."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, question, answer, confidence_score, status
                FROM submission_qa
                WHERE document_id = ?
                ORDER BY created_at
            """, (document_id,))
            
            rows = cursor.fetchall()
            return [{
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'confidence_score': row[3],
                'status': row[4]
            } for row in rows]
        except Exception as e:
            logger.error(f"Error getting Q&A pairs: {e}")
            return []


def calculate_question_count(text: str) -> int:
    """Heuristic to determine optimal number of Q&A pairs based on text length."""
    word_count = len(text.split())
    
    # Base calculation: 1 Q&A per 250 words
    base_questions = max(3, min(20, word_count // 250))
    
    # Adjust for medical terminology density (simple heuristic)
    medical_terms = ['medical', 'patient', 'treatment', 'diagnosis', 'therapy', 
                    'clinical', 'disease', 'symptom', 'medication', 'procedure']
    medical_count = sum(1 for term in medical_terms if term.lower() in text.lower())
    
    if medical_count > len(medical_terms) * 0.3:  # High medical density
        base_questions = int(base_questions * 1.3)
    
    return min(20, max(3, base_questions))


def process_pdf_with_gemini(file_bytes: bytes, filename: str) -> Optional[str]:
    """Process PDF file using Gemini 2.5 Flash via LiteLLM."""
    try:
        # Check if GEMINI_API_KEY is set
        if not os.getenv('GEMINI_API_KEY'):
            st.error("GEMINI_API_KEY environment variable not set. Please set your Gemini API key.")
            return None
        
        # Encode PDF to base64
        encoded_pdf = base64.b64encode(file_bytes).decode('utf-8')
        base64_url = f"data:application/pdf;base64,{encoded_pdf}"
        
        # Call Gemini 2.5 Flash via LiteLLM using image_url type (workaround for PDF)
        with st.spinner("Extracting text from PDF using Gemini 2.5 Flash..."):
            response = litellm.completion(
                model="gemini/gemini-2.0-flash-exp",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Convert this PDF to a well formatted text document. Preserve structure, headings, and important formatting. Focus on extracting all readable text content."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )
        
        extracted_text = response.choices[0].message.content
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error processing PDF with Gemini: {e}")
        st.error(f"Error processing PDF: {str(e)}")
        return None


def generate_qa_pairs(text: str, num_questions: int) -> Optional[List[Dict]]:
    """Generate Q&A pairs using Gemini Flash via LiteLLM."""
    try:
        with st.spinner(f"Generating {num_questions} Q&A pairs using Gemini Flash..."):
            response = litellm.completion(
                model="gemini/gemini-1.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Generate exactly {num_questions} medical Q&A pairs from this text:
                        
{text}

Format as a JSON array with objects containing 'question' and 'answer' fields.
Focus on key medical concepts, procedures, diagnoses, treatments, and important medical knowledge.
Make questions specific and answers comprehensive but concise.
Ensure questions are clinically relevant and educational.

Example format:
[
  {{
    "question": "What is the primary treatment for...",
    "answer": "The primary treatment involves..."
  }}
]"""
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=3000
            )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Sometimes the response might have extra text, try to find JSON
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                qa_pairs = json.loads(json_str)
            else:
                qa_pairs = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to parse the entire content
            qa_pairs = json.loads(content)
        
        # Add confidence scores
        for qa in qa_pairs:
            qa['confidence_score'] = 0.85  # Default confidence score
        
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Error generating Q&A pairs: {e}")
        st.error(f"Error generating Q&A pairs: {str(e)}")
        return None


def show():
    """Display the document submission page."""
    st.header("📄 Submit Medical Documents")
    st.write("Upload PDF files or enter text to generate Q&A pairs for the medical database.")
    
    # Display public access info

    
    # Initialize session state
    if 'submission_step' not in st.session_state:
        st.session_state.submission_step = 'upload'
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    if 'document_id' not in st.session_state:
        st.session_state.document_id = None
    
    # Initialize submissions database
    try:
        submissions_db = SubmissionDatabase(SUBMISSIONS_DB_PATH)
    except Exception as e:
        st.error(f"Failed to initialize submissions database: {e}")
        return
    
    # Step 1: Document Upload/Input
    if st.session_state.submission_step == 'upload':
        st.subheader("Step 1: Document Input")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📁 Upload PDF", "✏️ Enter Text"])
        
        with tab1:
            st.write("Upload a PDF file (max 20MB)")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload medical documents, research papers, or clinical guidelines"
            )
            
            if uploaded_file is not None:
                # Validate file size
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error(f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (20MB)")
                    return
                
                st.success(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f}MB)")
                
                if st.button("Process PDF", type="primary"):
                    # Process the PDF
                    file_bytes = uploaded_file.read()
                    extracted_text = process_pdf_with_gemini(file_bytes, uploaded_file.name)
                    
                    if extracted_text:
                        st.session_state.extracted_text = extracted_text
                        st.session_state.document_id = str(uuid.uuid4())
                        st.session_state.submission_step = 'review_text'
                        
                        # Save document to database
                        submissions_db.save_document(
                            st.session_state.document_id,
                            uploaded_file.name,
                            extracted_text,
                            'pdf',
                            uploaded_file.name
                        )
                        
                        st.rerun()
        
        with tab2:
            st.write("Enter or paste medical text directly")
            text_input = st.text_area(
                "Medical Text",
                height=300,
                placeholder="Enter medical text, research content, or clinical information...",
                help="Paste medical documents, research abstracts, or clinical guidelines"
            )
            
            if text_input.strip():
                word_count = len(text_input.split())
                st.info(f"Word count: {word_count}")
                
                if st.button("Process Text", type="primary"):
                    st.session_state.extracted_text = text_input.strip()
                    st.session_state.document_id = str(uuid.uuid4())
                    st.session_state.submission_step = 'review_text'
                    
                    # Save document to database
                    title = f"Text Document - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    submissions_db.save_document(
                        st.session_state.document_id,
                        title,
                        text_input.strip(),
                        'text'
                    )
                    
                    st.rerun()
    
    # Step 2: Review Extracted Text
    elif st.session_state.submission_step == 'review_text':
        st.subheader("Step 2: Review Extracted Text")
        
        # Display extracted text
        st.write("**Extracted/Entered Text:**")
        
        # Allow editing of extracted text
        edited_text = st.text_area(
            "Edit text if needed",
            value=st.session_state.extracted_text,
            height=400,
            help="Review and edit the extracted text before generating Q&A pairs"
        )
        
        # Calculate suggested number of questions
        suggested_questions = calculate_question_count(edited_text)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_questions = st.slider(
                "Number of Q&A pairs to generate",
                min_value=1,
                max_value=25,
                value=suggested_questions,
                help=f"Suggested: {suggested_questions} based on text length"
            )
        
        with col2:
            st.metric("Word Count", len(edited_text.split()))
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back to Upload"):
                st.session_state.submission_step = 'upload'
                st.rerun()
        
        with col2:
            if st.button("Generate Q&A Pairs", type="primary"):
                # Update extracted text if edited
                st.session_state.extracted_text = edited_text
                
                # Generate Q&A pairs
                qa_pairs = generate_qa_pairs(edited_text, num_questions)
                
                if qa_pairs:
                    st.session_state.qa_pairs = qa_pairs
                    st.session_state.submission_step = 'review_qa'
                    st.rerun()
    
    # Step 3: Review and Edit Q&A Pairs
    elif st.session_state.submission_step == 'review_qa':
        st.subheader("Step 3: Review and Edit Q&A Pairs")
        
        if not st.session_state.qa_pairs:
            st.error("No Q&A pairs found. Please go back and generate them.")
            return
        
        st.write(f"**Generated {len(st.session_state.qa_pairs)} Q&A pairs:**")
        
        # Display and allow editing of Q&A pairs
        edited_qa_pairs = []
        
        for i, qa in enumerate(st.session_state.qa_pairs):
            with st.expander(f"Q&A Pair {i+1}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    question = st.text_area(
                        "Question",
                        value=qa['question'],
                        key=f"question_{i}",
                        height=80
                    )
                    
                    answer = st.text_area(
                        "Answer",
                        value=qa['answer'],
                        key=f"answer_{i}",
                        height=120
                    )
                
                with col2:
                    confidence = st.slider(
                        "Confidence",
                        0.0, 1.0,
                        qa.get('confidence_score', 0.85),
                        key=f"confidence_{i}",
                        help="Confidence in the Q&A pair quality"
                    )
                    
                    include = st.checkbox(
                        "Include",
                        value=True,
                        key=f"include_{i}",
                        help="Include this Q&A pair in the final submission"
                    )
                
                if include:
                    edited_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'confidence_score': confidence
                    })
        
        st.write(f"**{len(edited_qa_pairs)} Q&A pairs selected for submission**")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back to Text Review"):
                st.session_state.submission_step = 'review_text'
                st.rerun()
        
        with col2:
            if st.button("Regenerate Q&A"):
                st.session_state.submission_step = 'review_text'
                st.rerun()
        
        with col3:
            if st.button("Submit to Database", type="primary"):
                if edited_qa_pairs:
                    # Save Q&A pairs to submissions database
                    success = submissions_db.save_qa_pairs(edited_qa_pairs, st.session_state.document_id)
                    
                    if success:
                        st.success(f"Successfully submitted {len(edited_qa_pairs)} Q&A pairs to the database!")
                        st.balloons()
                        
                        # Show submission confirmation
                        st.info(f"📊 {len(edited_qa_pairs)} Q&A pairs added to the medical knowledge base!")
                        
                        # Reset session state
                        st.session_state.submission_step = 'upload'
                        st.session_state.extracted_text = ""
                        st.session_state.qa_pairs = []
                        st.session_state.document_id = None
                        
                        st.rerun()
                    else:
                        st.error("Failed to save Q&A pairs to database. Please try again.")
                else:
                    st.warning("No Q&A pairs selected for submission.")
    
    # Sidebar with progress and help
    with st.sidebar:
        st.subheader("Submission Progress")
        
        steps = [
            ("upload", "📁 Document Input"),
            ("review_text", "📝 Review Text"),
            ("review_qa", "❓ Review Q&A"),
        ]
        
        for step_key, step_name in steps:
            if st.session_state.submission_step == step_key:
                st.write(f"**→ {step_name}**")
            else:
                st.write(f"   {step_name}")
        
        st.divider()
        
        st.subheader("💡 Tips")
        st.write("""
        - **PDF files**: Up to 20MB, medical documents work best
        - **Text input**: Paste research abstracts, clinical guidelines
        - **Q&A generation**: Review and edit for accuracy
        - **Quality**: Higher confidence scores indicate better Q&A pairs
        """)
        
        if st.session_state.submission_step != 'upload':
            st.divider()
            if st.button("🔄 Start Over"):
                st.session_state.submission_step = 'upload'
                st.session_state.extracted_text = ""
                st.session_state.qa_pairs = []
                st.session_state.document_id = None
                st.rerun()