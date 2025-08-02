#!/usr/bin/env python3
"""
Medical Q&A Search CLI

A command-line interface for searching the medical Q&A database with visual results.
Supports both vector similarity search and full-text search with optional specialty filtering.
"""

import argparse
import sys
import os
from typing import List, Optional
import textwrap

try:
    from loma_dataset.database import MedicalVectorDB
    from loma_dataset.processor import MedicalEmbeddingGenerator
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you have installed the loma_dataset package.")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output."""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ""
        cls.YELLOW = cls.RED = cls.BOLD = cls.UNDERLINE = cls.END = ""


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def print_separator():
    """Print a visual separator."""
    print(f"{Colors.CYAN}{'-' * 60}{Colors.END}")


def wrap_text(text: str, width: int = 50) -> str:
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def format_qa_result(result, index: int, search_type: str = "similarity"):
    """Format a Q&A search result for display."""
    print(f"{Colors.BOLD}{Colors.GREEN}#{index + 1}{Colors.END}")

    if search_type == "similarity" and hasattr(result, "similarity"):
        similarity_color = (
            Colors.GREEN
            if result.similarity > 0.8
            else Colors.YELLOW
            if result.similarity > 0.6
            else Colors.RED
        )
        print(
            f"{Colors.BOLD}Similarity:{Colors.END} {similarity_color}{result.similarity:.3f}{Colors.END}"
        )

    if hasattr(result, "qa"):
        qa = result.qa
        document = result.document if hasattr(result, "document") else None
    else:
        qa = result
        document = None

    # Display specialty if available
    if hasattr(qa, "specialty") and qa.specialty:
        print(
            f"{Colors.BOLD}Specialty:{Colors.END} {Colors.CYAN}{qa.specialty}{Colors.END}"
        )

    # Display question
    print(f"{Colors.BOLD}Question:{Colors.END}")
    question_wrapped = wrap_text(qa.question, width=55)
    for line in question_wrapped.split("\n"):
        print(f"  {Colors.BLUE}{line}{Colors.END}")

    # Display answer (truncated if too long)
    print(f"{Colors.BOLD}Answer:{Colors.END}")
    answer_text = qa.answer
    if len(answer_text) > 200:
        answer_text = answer_text[:200] + "..."

    answer_wrapped = wrap_text(answer_text, width=55)
    for line in answer_wrapped.split("\n"):
        print(f"  {line}")

    # Display related document if available
    if document:
        print(f"{Colors.BOLD}📄 Related Document:{Colors.END}")

        # Display title (check both new and old schema)
        title = None
        if hasattr(document, "title") and document.title:
            title = document.title
        elif hasattr(document, "paper_title") and document.paper_title:
            title = document.paper_title

        if title:
            title_wrapped = wrap_text(title, width=55)
            for line in title_wrapped.split("\n"):
                print(f"  {Colors.UNDERLINE}{line}{Colors.END}")

        # Display content (check both new and old schema)
        passage_text = ""
        if hasattr(document, "content") and document.content:
            passage_text = document.content
        elif hasattr(document, "passage_text") and document.passage_text:
            passage_text = document.passage_text
        elif hasattr(document, "abstract") and document.abstract:
            passage_text = document.abstract

        if passage_text:
            if len(passage_text) > 150:
                passage_text = passage_text[:150] + "..."

            passage_wrapped = wrap_text(passage_text, width=55)
            for line in passage_wrapped.split("\n"):
                print(f"  {Colors.CYAN}{line}{Colors.END}")

        # Show document metadata
        doc_info = []
        if hasattr(document, "year") and document.year:
            doc_info.append(f"Year: {document.year}")
        if hasattr(document, "specialty") and document.specialty:
            doc_info.append(f"Specialty: {document.specialty}")
        if hasattr(document, "venue") and document.venue:
            doc_info.append(f"Venue: {document.venue}")
        if doc_info:
            print(f"  {Colors.YELLOW}({' | '.join(doc_info)}){Colors.END}")

    print_separator()


def format_document_result(doc, index: int):
    """Format a document search result for display."""
    print(f"{Colors.BOLD}{Colors.GREEN}#{index + 1}{Colors.END}")

    # Display title (check both new and old schema)
    title = None
    if hasattr(doc, "title") and doc.title:
        title = doc.title
    elif hasattr(doc, "paper_title") and doc.paper_title:
        title = doc.paper_title

    if title:
        print(f"{Colors.BOLD}Title:{Colors.END}")
        title_wrapped = wrap_text(title, width=55)
        for line in title_wrapped.split("\n"):
            print(f"  {Colors.BLUE}{line}{Colors.END}")

    # Display specialty and year
    info_parts = []
    if hasattr(doc, "specialty") and doc.specialty:
        info_parts.append(f"Specialty: {Colors.CYAN}{doc.specialty}{Colors.END}")
    if hasattr(doc, "year") and doc.year:
        info_parts.append(f"Year: {Colors.YELLOW}{doc.year}{Colors.END}")

    if info_parts:
        print(f"{Colors.BOLD}{' | '.join(info_parts)}")

    # Display content (check both new and old schema)
    text_content = ""
    if hasattr(doc, "content") and doc.content:
        text_content = doc.content
        print(f"{Colors.BOLD}Content:{Colors.END}")
    elif hasattr(doc, "abstract") and doc.abstract:
        text_content = doc.abstract
        print(f"{Colors.BOLD}Abstract:{Colors.END}")
    elif hasattr(doc, "passage_text") and doc.passage_text:
        text_content = doc.passage_text
        print(f"{Colors.BOLD}Passage:{Colors.END}")

    if text_content:
        if len(text_content) > 200:
            text_content = text_content[:200] + "..."

        text_wrapped = wrap_text(text_content, width=55)
        for line in text_wrapped.split("\n"):
            print(f"  {line}")

    print_separator()


def search_qa(
    db: MedicalVectorDB,
    query: str,
    search_type: str,
    embedding_generator: Optional[MedicalEmbeddingGenerator] = None,
    specialty: Optional[str] = None,
):
    """Perform Q&A search and display results with related documents."""
    print_header(f"{search_type.upper()} SEARCH RESULTS")
    print(f"{Colors.BOLD}Query:{Colors.END} {Colors.YELLOW}'{query}'{Colors.END}")
    if specialty:
        print(
            f"{Colors.BOLD}Specialty Filter:{Colors.END} {Colors.CYAN}{specialty}{Colors.END}"
        )
    print()

    try:
        if search_type == "vector":
            if not embedding_generator:
                print(
                    f"{Colors.RED}❌ Vector search requires embedding generator{Colors.END}"
                )
                return

            # Generate query embedding
            query_vector = embedding_generator.generate_embeddings([query])[0]
            results = db.search_similar_qa(
                query_vector=query_vector,
                limit=5,
                threshold=0.0,  # Show all results
                specialty=specialty,
            )
        else:  # full-text search
            results = db.search_qa_text(query=query, limit=5, specialty=specialty)

        if not results:
            print(f"{Colors.YELLOW}No results found for your query.{Colors.END}")
            return

        print(f"{Colors.GREEN}Found {len(results)} result(s):{Colors.END}\n")

        for i, result in enumerate(results):
            format_qa_result(result, i, search_type)

    except Exception as e:
        print(f"{Colors.RED}❌ Search error: {e}{Colors.END}")


def search_documents_by_text(
    db: MedicalVectorDB,
    query: str,
    search_type: str,
    embedding_generator: Optional[MedicalEmbeddingGenerator] = None,
    specialty: Optional[str] = None,
):
    """Perform document search and display results."""
    print_header(f"DOCUMENT {search_type.upper()} SEARCH RESULTS")
    print(f"{Colors.BOLD}Query:{Colors.END} {Colors.YELLOW}'{query}'{Colors.END}")
    if specialty:
        print(
            f"{Colors.BOLD}Specialty Filter:{Colors.END} {Colors.CYAN}{specialty}{Colors.END}"
        )
    print()

    try:
        if search_type == "vector":
            if not embedding_generator:
                print(
                    f"{Colors.RED}❌ Vector search requires embedding generator{Colors.END}"
                )
                return

            # Generate query embedding
            query_vector = embedding_generator.generate_embeddings([query])[0]
            # Vector search for documents is not available in current implementation
            # Fall back to text search
            print(
                f"{Colors.YELLOW}Vector search for documents not available, using text search instead.{Colors.END}"
            )
            results = db.search_documents_text(
                query=query, limit=5, specialty=specialty
            )
        else:  # full-text search
            results = db.search_documents_text(
                query=query, limit=5, specialty=specialty
            )

        if not results:
            print(f"{Colors.YELLOW}No documents found for your query.{Colors.END}")
            return

        print(f"{Colors.GREEN}Found {len(results)} document(s):{Colors.END}\n")

        for i, result in enumerate(results):
            if search_type == "vector" and hasattr(result, "similarity"):
                # For vector search results with similarity scores
                print(f"{Colors.BOLD}{Colors.GREEN}#{i + 1}{Colors.END}")
                similarity_color = (
                    Colors.GREEN
                    if result.similarity > 0.8
                    else Colors.YELLOW
                    if result.similarity > 0.6
                    else Colors.RED
                )
                print(
                    f"{Colors.BOLD}Similarity:{Colors.END} {similarity_color}{result.similarity:.3f}{Colors.END}"
                )
                format_document_result(result.document, i)
            else:
                # For full-text search results - extract document from DocumentSearchResult
                doc = result.document if hasattr(result, "document") else result
                format_document_result(doc, i)

    except Exception as e:
        print(f"{Colors.RED}❌ Document search error: {e}{Colors.END}")


def search_documents(db: MedicalVectorDB, specialty: str):
    """Search documents by specialty and display results."""
    print_header("DOCUMENTS BY SPECIALTY")
    print(f"{Colors.BOLD}Specialty:{Colors.END} {Colors.CYAN}{specialty}{Colors.END}\n")

    try:
        # Use a broad search term that should match most documents
        results = db.search_documents_text(
            query="the",  # Common word that should match most documents
            limit=5,
            specialty=specialty,
        )

        if not results:
            print(
                f"{Colors.YELLOW}No documents found for specialty '{specialty}'.{Colors.END}"
            )
            return

        print(f"{Colors.GREEN}Found {len(results)} document(s):{Colors.END}\n")

        for i, result in enumerate(results):
            # Extract document from DocumentSearchResult
            doc = result.document if hasattr(result, "document") else result
            format_document_result(doc, i)

    except Exception as e:
        print(f"{Colors.RED}❌ Document search error: {e}{Colors.END}")


def show_database_stats(db: MedicalVectorDB):
    """Display database statistics."""
    print_header("DATABASE STATISTICS")

    try:
        stats = db.get_stats()

        print(
            f"{Colors.BOLD}Total Q&A Entries:{Colors.END} {Colors.GREEN}{stats.get('qa_count', 0):,}{Colors.END}"
        )
        print(
            f"{Colors.BOLD}Total Documents:{Colors.END} {Colors.GREEN}{stats.get('document_count', 0):,}{Colors.END}"
        )
        print(
            f"{Colors.BOLD}Database Path:{Colors.END} {Colors.YELLOW}{stats.get('database_path', 'Unknown')}{Colors.END}"
        )

        specialties = stats.get("specialties", [])
        if specialties:
            print(
                f"{Colors.BOLD}Available Specialties ({len(specialties)}):{Colors.END}"
            )
            for specialty in sorted(specialties)[:10]:  # Show first 10
                print(f"  • {Colors.CYAN}{specialty}{Colors.END}")
            if len(specialties) > 10:
                print(f"  ... and {len(specialties) - 10} more")
        else:
            print(f"{Colors.YELLOW}No specialties found in database{Colors.END}")

        print()

    except Exception as e:
        print(f"{Colors.RED}❌ Error getting database stats: {e}{Colors.END}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Search medical Q&A database with visual results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "diabetes treatment"                    # Full-text search in Q&A
  %(prog)s "heart disease" --vector               # Vector similarity search in Q&A
  %(prog)s "cancer" --specialty "Oncology"        # Search with specialty filter
  %(prog)s "treatment" --search-docs              # Full-text search in documents
  %(prog)s "diabetes" --search-docs --vector      # Vector search in documents
  %(prog)s --docs "Cardiology"                    # Browse documents by specialty
  %(prog)s --stats                                # Show database statistics
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--vector",
        "-v",
        action="store_true",
        help="Use vector similarity search (default: full-text)",
    )
    parser.add_argument(
        "--search-docs",
        action="store_true",
        help="Search in documents instead of Q&A entries",
    )
    parser.add_argument("--specialty", "-s", help="Filter by medical specialty")
    parser.add_argument(
        "--docs", "-d", metavar="SPECIALTY", help="Browse documents by specialty"
    )
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument(
        "--db-path",
        default="miriad_medical.db",
        help="Path to database file (default: miriad_medical.db)",
    )
    parser.add_argument(
        "--model-name",
        default="AleksanderObuchowski/medembed-small-onnx",
        help="Model name for vector search",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    args = parser.parse_args()

    # Disable colors if requested or if not in terminal
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"{Colors.RED}❌ Database file not found: {args.db_path}{Colors.END}")
        print(f"Run the processing script first to create the database.")
        sys.exit(1)

    # Initialize database
    try:
        db = MedicalVectorDB(args.db_path)
        db.initialize()
    except Exception as e:
        print(f"{Colors.RED}❌ Error connecting to database: {e}{Colors.END}")
        sys.exit(1)

    # Show stats if requested
    if args.stats:
        show_database_stats(db)
        if not args.query and not args.docs:
            return

    # Browse documents by specialty
    if args.docs:
        search_documents(db, args.docs)
        return

    # Require query for search
    if not args.query:
        print(
            f"{Colors.YELLOW}Please provide a search query or use --stats to see database info.{Colors.END}"
        )
        print(f"Use --help for usage examples.")
        return

    # Initialize embedding generator for vector search
    embedding_generator = None
    if args.vector:
        try:
            print(f"{Colors.BLUE}Loading embedding model...{Colors.END}")
            embedding_generator = MedicalEmbeddingGenerator(args.model_name)
            embedding_generator.initialize()
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading embedding model: {e}{Colors.END}")
            print(f"Falling back to full-text search.")
            args.vector = False

    # Perform search
    search_type = "vector" if args.vector else "full-text"

    if args.search_docs:
        # Search in documents
        search_documents_by_text(
            db, args.query, search_type, embedding_generator, args.specialty
        )
    else:
        # Search in Q&A entries (default)
        search_qa(db, args.query, search_type, embedding_generator, args.specialty)


if __name__ == "__main__":
    main()
