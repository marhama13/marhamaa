import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch

# Set page config
st.set_page_config(
    page_title="Named Entity Recognition Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .entity-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ner_model' not in st.session_state:
    st.session_state.ner_model = None

@st.cache_resource
def load_ner_model():
    """Load the NER model with caching"""
    try:
        return pipeline("ner", grouped_entities=True)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def analyze_entities(text):
    """Perform NER analysis on input text"""
    if not text.strip():
        return [], {}
    
    ner_results = st.session_state.ner_model(text)
    entity_counts = Counter([entity['entity_group'] for entity in ner_results])
    
    return ner_results, entity_counts

def create_entity_chart(entity_counts):
    """Create an interactive bar chart using Plotly"""
    if not entity_counts:
        return None
    
    df = pd.DataFrame(list(entity_counts.items()), columns=['Entity Type', 'Count'])
    
    fig = px.bar(
        df, 
        x='Entity Type', 
        y='Count',
        title='Named Entities Distribution',
        color='Count',
        color_continuous_scale='Blues',
        text='Count'
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        showlegend=False,
        height=400
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def create_confidence_chart(ner_results):
    """Create a confidence score visualization"""
    if not ner_results:
        return None
    
    df = pd.DataFrame(ner_results)
    
    fig = px.scatter(
        df, 
        x='word', 
        y='score',
        color='entity_group',
        size='score',
        hover_data=['entity_group'],
        title='Entity Confidence Scores',
        labels={'score': 'Confidence Score', 'word': 'Entity'}
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def highlight_entities_in_text(text, ner_results):
    """Highlight entities in the original text"""
    if not ner_results:
        return text
    
    # Color mapping for different entity types
    color_map = {
        'PER': '#FF6B6B',    # Red for persons
        'ORG': '#4ECDC4',    # Teal for organizations
        'LOC': '#45B7D1',    # Blue for locations
        'MISC': '#96CEB4',   # Green for miscellaneous
        'GPE': '#FFEAA7',    # Yellow for geopolitical entities
    }
    
    highlighted_text = text
    
    # Sort entities by start position in reverse order to avoid index shifting
    sorted_entities = sorted(ner_results, key=lambda x: x.get('start', 0), reverse=True)
    
    for entity in sorted_entities:
        word = entity['word']
        entity_type = entity['entity_group']
        color = color_map.get(entity_type, '#DDA0DD')
        
        # Simple replacement (this is a basic approach)
        highlighted_text = highlighted_text.replace(
            word, 
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{word} ({entity_type})</span>'
        )
    
    return highlighted_text

# Main app
def main():
    st.markdown('<h1 class="main-header">🔍 Named Entity Recognition Analyzer</h1>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.ner_model is None:
        with st.spinner("Loading NER model... This may take a moment."):
            st.session_state.ner_model = load_ner_model()
    
    if st.session_state.ner_model is None:
        st.error("Failed to load the NER model. Please check your internet connection and try again.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    
    # Sample texts
    sample_texts = {
        "BTS Example": "BTS performed at Wembley Stadium in London, and their leader RM gave a speech at the United Nations in New York.",
        "Business Example": "Apple Inc. was founded by Steve Jobs in Cupertino, California. The company is now headquartered in Apple Park.",
        "News Example": "President Biden met with Prime Minister Johnson at the White House to discuss climate change and economic policies.",
        "Sports Example": "Lionel Messi scored a hat-trick for Barcelona against Real Madrid at Camp Nou stadium in Spain."
    }
    
    selected_sample = st.sidebar.selectbox("Choose a sample text:", ["Custom"] + list(sample_texts.keys()))
    
    # Text input
    if selected_sample == "Custom":
        default_text = ""
    else:
        default_text = sample_texts[selected_sample]
    
    input_text = st.text_area(
        "Enter text for Named Entity Recognition analysis:",
        value=default_text,
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Entities", type="primary"):
        if input_text.strip():
            with st.spinner("Analyzing entities..."):
                ner_results, entity_counts = analyze_entities(input_text)
            
            if ner_results:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Entities", len(ner_results))
                with col2:
                    st.metric("Unique Types", len(entity_counts))
                with col3:
                    avg_confidence = sum(entity['score'] for entity in ner_results) / len(ner_results)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                with col4:
                    st.metric("Text Length", len(input_text.split()))
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Highlighted Text", "📊 Entity Distribution", "🎯 Confidence Scores", "📋 Detailed Results"])
                
                with tab1:
                    st.subheader("Text with Highlighted Entities")
                    highlighted = highlight_entities_in_text(input_text, ner_results)
                    st.markdown(highlighted, unsafe_allow_html=True)
                    
                    # Legend
                    st.markdown("**Legend:**")
                    legend_cols = st.columns(5)
                    legend_items = [
                        ("PER", "Person", "#FF6B6B"),
                        ("ORG", "Organization", "#4ECDC4"),
                        ("LOC", "Location", "#45B7D1"),
                        ("MISC", "Miscellaneous", "#96CEB4"),
                        ("GPE", "Geopolitical", "#FFEAA7")
                    ]
                    
                    for i, (code, name, color) in enumerate(legend_items):
                        with legend_cols[i]:
                            st.markdown(f'<span style="background-color: {color}; padding: 2px 8px; border-radius: 3px; font-weight: bold;">{code}</span> {name}', unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("Entity Type Distribution")
                    fig_dist = create_entity_chart(entity_counts)
                    if fig_dist:
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                with tab3:
                    st.subheader("Confidence Score Analysis")
                    fig_conf = create_confidence_chart(ner_results)
                    if fig_conf:
                        st.plotly_chart(fig_conf, use_container_width=True)
                
                with tab4:
                    st.subheader("Detailed Entity Information")
                    
                    # Create DataFrame for better display
                    df_results = pd.DataFrame([
                        {
                            'Entity': entity['word'],
                            'Type': entity['entity_group'],
                            'Confidence': f"{entity['score']:.3f}",
                            'Start': entity.get('start', 'N/A'),
                            'End': entity.get('end', 'N/A')
                        }
                        for entity in ner_results
                    ])
                    
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="ner_results.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No entities found in the provided text.")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Information section
    with st.expander("ℹ️ About Named Entity Recognition"):
        st.markdown("""
        **Named Entity Recognition (NER)** is a natural language processing technique that identifies and classifies named entities in text.
        
        **Common Entity Types:**
        - **PER**: Person names (e.g., "John Smith", "Marie Curie")
        - **ORG**: Organizations (e.g., "Google", "United Nations")
        - **LOC**: Locations (e.g., "Paris", "Mount Everest")
        - **MISC**: Miscellaneous entities (e.g., "Nobel Prize", "World Cup")
        - **GPE**: Geopolitical entities (e.g., "France", "European Union")
        
        **Applications:**
        - Information extraction
        - Question answering systems
        - Content categorization
        - Search engine optimization
        - Social media analysis
        """)

if __name__ == "__main__":
    main()