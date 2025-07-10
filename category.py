import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import time

# Set page config
st.set_page_config(
    page_title="Text Classification App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .classification-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching the model
@st.cache_resource
def load_classifier():
    """Load the zero-shot classification model (cached)"""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(text, labels, classifier):
    """Classify text into predefined categories"""
    if not text.strip():
        return None, None
    
    result = classifier(text, candidate_labels=labels)
    return result['labels'][0], dict(zip(result['labels'], result['scores']))

def create_score_chart(scores):
    """Create a horizontal bar chart for classification scores"""
    df = pd.DataFrame(list(scores.items()), columns=['Category', 'Score'])
    df = df.sort_values('Score', ascending=True)
    
    fig = px.bar(
        df, 
        x='Score', 
        y='Category',
        orientation='h',
        title='Classification Confidence Scores',
        color='Score',
        color_continuous_scale='viridis',
        text='Score'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Confidence Score",
        yaxis_title="Category",
        title_x=0.5
    )
    
    return fig

# Main app
def main():
    # Header
    st.title("🔍 Text Classification App")
    st.markdown("**Classify text into custom categories using AI-powered zero-shot learning**")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Default labels
    default_labels = ["food waste", "nature", "waste management"]
    
    # Custom labels input
    st.sidebar.subheader("Categories")
    custom_labels = st.sidebar.text_area(
        "Enter categories (one per line):",
        value="\n".join(default_labels),
        height=100,
        help="Enter each category on a new line"
    )
    
    # Process custom labels
    labels = [label.strip() for label in custom_labels.split('\n') if label.strip()]
    
    if not labels:
        st.sidebar.error("Please enter at least one category!")
        return
    
    st.sidebar.success(f"Using {len(labels)} categories")
    
    # Model loading
    with st.spinner("Loading AI model..."):
        try:
            classifier = load_classifier()
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Use example"],
            horizontal=True
        )
        
        if input_method == "Use example":
            example_texts = [
                "Composting organic leftovers helps reduce food waste and improve soil health.",
                "The forest ecosystem thrives when biodiversity is maintained.",
                "Proper recycling systems are essential for managing municipal waste.",
                "Solar panels convert sunlight into clean renewable energy.",
                "Food banks help distribute surplus groceries to those in need."
            ]
            
            selected_example = st.selectbox(
                "Select an example:",
                example_texts,
                index=0
            )
            text_input = selected_example
        else:
            text_input = st.text_area(
                "Enter your text here:",
                height=120,
                placeholder="Type or paste your text here for classification..."
            )
        
        # Classification button
        if st.button("🚀 Classify Text", type="primary", use_container_width=True):
            if text_input and text_input.strip():
                with st.spinner("Classifying text..."):
                    start_time = time.time()
                    predicted_category, scores = classify_text(text_input, labels, classifier)
                    end_time = time.time()
                    
                    if predicted_category and scores:
                        # Store results in session state
                        st.session_state.last_result = {
                            'text': text_input,
                            'category': predicted_category,
                            'scores': scores,
                            'processing_time': end_time - start_time
                        }
            else:
                st.warning("Please enter some text to classify!")
    
    with col2:
        st.subheader("📊 Categories")
        
        # Display current categories
        for i, label in enumerate(labels, 1):
            st.markdown(f"**{i}.** {label.title()}")
        
        # Add some spacing
        st.markdown("---")
        
        # Model information
        st.subheader("🤖 Model Info")
        st.info(
            """
            **Model:** facebook/bart-large-mnli
            
            **Type:** Zero-shot Classification
            
            **Capability:** Can classify text into any categories without prior training on those specific labels.
            """
        )
    
    # Display results
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        result = st.session_state.last_result
        
        st.markdown("---")
        st.subheader("📈 Classification Results")
        
        # Result summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Category",
                result['category'].title(),
                help="The category with the highest confidence score"
            )
        
        with col2:
            confidence = result['scores'][result['category']]
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                help="How confident the model is in its prediction"
            )
        
        with col3:
            st.metric(
                "Processing Time",
                f"{result['processing_time']:.2f}s",
                help="Time taken to process the text"
            )
        
        # Display input text
        st.subheader("📄 Input Text")
        st.markdown(f"*{result['text']}*")
        
        # Prediction result with styling
        st.markdown(
            f"""
            <div class="classification-result">
                🎯 Predicted Category: {result['category'].upper()}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Detailed scores
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Detailed Scores")
            
            # Create a DataFrame for better display
            scores_df = pd.DataFrame([
                {'Category': cat.title(), 'Score': f"{score:.1%}", 'Raw Score': score}
                for cat, score in result['scores'].items()
            ]).sort_values('Raw Score', ascending=False)
            
            st.dataframe(
                scores_df[['Category', 'Score']],
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.subheader("📈 Visualization")
            fig = create_score_chart(result['scores'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.subheader("💡 Interpretation")
        confidence = result['scores'][result['category']]
        
        if confidence > 0.8:
            st.success(f"**High Confidence:** The model is very confident that this text belongs to the '{result['category']}' category.")
        elif confidence > 0.6:
            st.info(f"**Medium Confidence:** The model thinks this text likely belongs to the '{result['category']}' category.")
        else:
            st.warning(f"**Low Confidence:** The model is uncertain but leans towards the '{result['category']}' category.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with ❤️ using Streamlit and Hugging Face Transformers
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()