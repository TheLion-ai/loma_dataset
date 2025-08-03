"""
Medical Q&A Database Streamlit Application
=========================================

Modern Streamlit application using st.navigation for multipage architecture.
"""

import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.loma_dataset.database import MedicalVectorDB
    from src.loma_dataset.processor import MedicalEmbeddingGenerator
    from src.loma_dataset.exceptions import DatabaseError, ValidationError
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Import page modules
from pages import dashboard, edit, analytics, unified_search

# Configure Streamlit page
st.set_page_config(
    page_title="LOMA Database",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(
    "https://sdmntprukwest.oaiusercontent.com/files/00000000-e6a0-6243-8416-e800c7fa0cd2/raw?se=2025-08-02T16%3A23%3A43Z&sp=r&sv=2024-08-04&sr=b&scid=dca5272d-6240-5bb0-a7e4-49305e62b970&skoid=eb780365-537d-4279-a878-cae64e33aa9c&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-08-02T06%3A08%3A45Z&ske=2025-08-03T06%3A08%3A45Z&sks=b&skv=2024-08-04&sig=NX1pHXxLI1Gfx2oGZ%2B5pT5pXYEDutcqaWscYhna6fSE%3D"
)

# Constants
DB_PATH = "miriad_medical_minlm.db"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None
if "embedding_generator" not in st.session_state:
    st.session_state.embedding_generator = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False


def initialize_database():
    """Initialize the database connection for the current session."""
    try:
        db = MedicalVectorDB(DB_PATH)
        db.initialize()
        return db
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        return None


@st.cache_resource
def initialize_embedding_generator():
    """Initialize and cache the embedding generator."""
    try:
        generator = MedicalEmbeddingGenerator(
            DEFAULT_MODEL, 
            use_quantized=False,  # Not relevant for sentence-transformers
            model_type="auto"     # Use auto mode for sentence-transformers with ONNX fallback
        )
        generator.initialize()
        return generator
    except Exception as e:
        st.error(f"Failed to initialize embedding generator: {e}")
        return None


def main():
    """Main application function with navigation."""

    # Initialize components
    if st.session_state.db is None:
        with st.spinner("Initializing database..."):
            st.session_state.db = initialize_database()

    if st.session_state.embedding_generator is None:
        with st.spinner("Initializing embedding generator..."):
            st.session_state.embedding_generator = initialize_embedding_generator()

    if st.session_state.db is None:
        st.error("Cannot proceed without database connection.")
        return

    # Define pages using st.Page
    pages = [
        st.Page(dashboard.show, title="Dashboard", icon="📊", url_path="dashboard"),
        st.Page(
            unified_search.show, title="Search & Browse", icon="🔍", url_path="search"
        ),
        st.Page(edit.show, title="Edit Data", icon="📝", url_path="edit"),
        st.Page(analytics.show, title="Analytics", icon="📈", url_path="analytics"),
    ]

    # Create navigation
    pg = st.navigation(pages)

    # Show database stats in sidebar
    with st.sidebar:
        st.header("Database Statistics")
        try:
            stats = st.session_state.db.get_stats()
            st.metric("Q&A Entries", stats["qa_count"])
            st.metric("Documents", stats["document_count"])
            st.metric("Specialties", len(stats["specialties"]))
        except Exception as e:
            st.error(f"Error getting stats: {e}")

    # Run the selected page
    pg.run()


if __name__ == "__main__":
    main()
