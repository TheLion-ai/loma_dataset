"""
Dashboard page for the Medical Q&A Database Explorer
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def show():
    """Display the main dashboard."""
    st.header("📊 Database Overview")
    
    if st.session_state.db is None:
        st.error("Database not initialized.")
        return
    
    try:
        stats = st.session_state.db.get_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Q&A Entries",
                value=f"{stats['qa_count']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Documents",
                value=f"{stats['document_count']:,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Medical Specialties",
                value=len(stats['specialties']),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Database Size",
                value="SQLite",
                delta=None
            )
        
        # Specialties distribution
        if stats['specialties']:
            st.subheader("📈 Specialties Distribution")
            
            # Get specialty counts
            specialty_counts = {}
            cursor = st.session_state.db.conn.cursor()
            for specialty in stats['specialties']:
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE specialty = ?", 
                    [specialty]
                )
                specialty_counts[specialty] = cursor.fetchone()[0]
            
            # Create bar chart
            if specialty_counts:
                df_specialties = pd.DataFrame(
                    list(specialty_counts.items()),
                    columns=['Specialty', 'Count']
                ).sort_values('Count', ascending=False)
                
                fig = px.bar(
                    df_specialties,
                    x='Specialty',
                    y='Count',
                    title="Documents by Medical Specialty",
                    color='Count',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity (if timestamps are available)
        st.subheader("📅 Database Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.info(f"**Database Path:** {stats['database_path']}")
        
        with info_col2:
            st.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")