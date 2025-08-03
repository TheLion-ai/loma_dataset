"""
Analytics page for the Medical Q&A Database Explorer
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def show():
    """Display analytics and insights."""
    st.header("📈 Database Analytics")
    
    if st.session_state.db is None:
        st.error("Database not initialized.")
        return
    
    try:
        stats = st.session_state.db.get_stats()
        cursor = st.session_state.db.conn.cursor()
        
        # Specialty distribution
        st.subheader("📊 Specialty Analysis")
        
        if stats['specialties']:
            specialty_data = []
            for specialty in stats['specialties']:
                cursor.execute("SELECT COUNT(*) FROM documents WHERE specialty = ?", [specialty])
                doc_count = cursor.fetchone()[0]
                cursor.execute("""
                    SELECT COUNT(*) FROM medical_qa mq 
                    JOIN documents d ON mq.document_id = d.id 
                    WHERE d.specialty = ?
                """, [specialty])
                qa_count = cursor.fetchone()[0]
                specialty_data.append({
                    'Specialty': specialty,
                    'Documents': doc_count,
                    'Q&A Entries': qa_count
                })
            
            df_specialties = pd.DataFrame(specialty_data)
            
            # Create grouped bar chart
            fig = px.bar(
                df_specialties.melt(id_vars=['Specialty'], var_name='Type', value_name='Count'),
                x='Specialty',
                y='Count',
                color='Type',
                title="Documents and Q&A Entries by Specialty",
                barmode='group'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Specialty pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie_docs = px.pie(
                    df_specialties, 
                    values='Documents', 
                    names='Specialty',
                    title="Document Distribution by Specialty"
                )
                st.plotly_chart(fig_pie_docs, use_container_width=True)
            
            with col2:
                fig_pie_qa = px.pie(
                    df_specialties, 
                    values='Q&A Entries', 
                    names='Specialty',
                    title="Q&A Distribution by Specialty"
                )
                st.plotly_chart(fig_pie_qa, use_container_width=True)
        
        # Year distribution (if available)
        st.subheader("📅 Publication Year Distribution")
        cursor.execute("SELECT year, COUNT(*) FROM documents WHERE year IS NOT NULL GROUP BY year ORDER BY year")
        year_data = cursor.fetchall()
        
        if year_data:
            df_years = pd.DataFrame(year_data, columns=['Year', 'Count'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_line = px.line(df_years, x='Year', y='Count', title="Documents by Publication Year")
                st.plotly_chart(fig_line, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(df_years, x='Year', y='Count', title="Documents per Year (Bar Chart)")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No year data available for analysis.")
        
        # Text length analysis
        st.subheader("📝 Text Length Analysis")
        
        # Get text length statistics
        cursor.execute("""
            SELECT 
                LENGTH(question) as question_length,
                LENGTH(answer) as answer_length,
                LENGTH(question) + LENGTH(answer) as total_length
            FROM medical_qa
            LIMIT 1000
        """)
        text_data = cursor.fetchall()
        
        if text_data:
            df_text = pd.DataFrame(text_data, columns=['Question Length', 'Answer Length', 'Total Length'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df_text, 
                    x='Total Length', 
                    title="Distribution of Q&A Text Lengths",
                    nbins=50
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    df_text.melt(var_name='Text Type', value_name='Length'),
                    x='Text Type',
                    y='Length',
                    title="Text Length Distribution by Type"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Display summary statistics
            st.subheader("📊 Text Statistics Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Question Length", f"{df_text['Question Length'].mean():.0f} chars")
                st.metric("Median Question Length", f"{df_text['Question Length'].median():.0f} chars")
            
            with col2:
                st.metric("Avg Answer Length", f"{df_text['Answer Length'].mean():.0f} chars")
                st.metric("Median Answer Length", f"{df_text['Answer Length'].median():.0f} chars")
            
            with col3:
                st.metric("Avg Total Length", f"{df_text['Total Length'].mean():.0f} chars")
                st.metric("Median Total Length", f"{df_text['Total Length'].median():.0f} chars")
        
        # Database size metrics
        st.subheader("💾 Storage Analytics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get database file size
            db_path = stats['database_path']
            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                size_mb = size_bytes / (1024 * 1024)
                st.metric("Database File Size", f"{size_mb:.2f} MB")
            else:
                st.metric("Database File Size", "N/A")
        
        with col2:
            # Table sizes
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            st.metric("Number of Tables", table_count)
        
        with col3:
            # Index count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
            index_count = cursor.fetchone()[0]
            st.metric("Number of Indexes", index_count)
        
        # Top specialties by volume
        if stats['specialties']:
            st.subheader("🏆 Top Medical Specialties")
            
            top_specialties = df_specialties.nlargest(10, 'Q&A Entries')
            
            fig_top = px.bar(
                top_specialties,
                x='Q&A Entries',
                y='Specialty',
                orientation='h',
                title="Top 10 Specialties by Q&A Volume",
                color='Q&A Entries',
                color_continuous_scale='viridis'
            )
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.exception(e)