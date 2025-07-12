import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch
import re
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Text Fill-Mask Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border-left: 5px solid #FFD700;
    }
    .prediction-text {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .prediction-score {
        font-size: 1rem;
        opacity: 0.9;
    }
    .mask-highlight {
        background-color: #FFD700;
        color: #000;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        border: 2px solid #FFA500;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 5px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.3s ease;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #2E86AB;
    }
    .sample-text {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .sample-text:hover {
        background-color: #e9ecef;
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fill_mask_model' not in st.session_state:
    st.session_state.fill_mask_model = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_fill_mask_model():
    """Load the fill-mask model with caching"""
    try:
        return pipeline("fill-mask", model="bert-base-uncased")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_mask(text):
    """Use the fill-mask pipeline to predict the masked token"""
    if not text.strip():
        return []
    
    # Check if [MASK] token exists
    if "[MASK]" not in text:
        return []
    
    try:
        results = st.session_state.fill_mask_model(text)
        return results
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return []

def create_confidence_chart(results):
    """Create a horizontal bar chart showing confidence scores"""
    if not results:
        return None
    
    # Extract tokens and scores
    tokens = [res['token_str'] for res in results]
    scores = [res['score'] for res in results]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Token': tokens,
        'Confidence': scores,
        'Percentage': [f"{score*100:.1f}%" for score in scores]
    })
    
    # Create horizontal bar chart
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Token',
        orientation='h',
        title='Prediction Confidence Scores',
        color='Confidence',
        color_continuous_scale='Viridis',
        text='Percentage'
    )
    
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Confidence Score",
        yaxis_title="Predicted Token",
        height=400,
        showlegend=False
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def create_word_cloud_style_viz(results):
    """Create a scatter plot resembling word cloud"""
    if not results:
        return None
    
    import numpy as np
    
    tokens = [res['token_str'] for res in results]
    scores = [res['score'] for res in results]
    
    # Create random positions for word cloud effect
    np.random.seed(42)
    x_pos = np.random.uniform(0, 10, len(tokens))
    y_pos = np.random.uniform(0, 10, len(tokens))
    
    fig = go.Figure()
    
    for i, (token, score) in enumerate(zip(tokens, scores)):
        fig.add_trace(go.Scatter(
            x=[x_pos[i]],
            y=[y_pos[i]],
            mode='text',
            text=token,
            textfont=dict(size=20 + score * 30, color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]),
            showlegend=False,
            hovertemplate=f"<b>{token}</b><br>Confidence: {score:.4f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Word Cloud Style Visualization",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def highlight_mask_in_text(text):
    """Highlight [MASK] tokens in the text"""
    highlighted = re.sub(
        r'\[MASK\]', 
        '<span class="mask-highlight">[MASK]</span>', 
        text
    )
    return highlighted

def add_to_history(text, results):
    """Add prediction to history"""
    st.session_state.prediction_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'text': text,
        'top_prediction': results[0]['token_str'] if results else "No prediction",
        'confidence': results[0]['score'] if results else 0
    })

# Main app
def main():
    st.markdown('<h1 class="main-header">🎯 AI Text Fill-Mask Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.fill_mask_model is None:
        with st.spinner("Loading BERT Fill-Mask model... This may take a moment."):
            st.session_state.fill_mask_model = load_fill_mask_model()
    
    if st.session_state.fill_mask_model is None:
        st.error("Failed to load the fill-mask model. Please check your internet connection and try again.")
        return
    
    # Sidebar
    st.sidebar.title("🎯 Prediction Options")
    
    # Model info
    with st.sidebar.expander("📊 Model Information"):
        st.markdown("""
        **Model**: BERT Base Uncased
        **Task**: Fill-Mask
        **Parameters**: 110M
        **Training**: Wikipedia + BookCorpus
        """)
    
    # Sample texts
    st.sidebar.subheader("📝 Sample Texts")
    sample_texts = {
        "Geography": "The Eiffel Tower is located in [MASK].",
        "Science": "The theory of [MASK] was developed by Einstein.",
        "Technology": "Apple is known for making [MASK] devices.",
        "Literature": "Shakespeare wrote many famous [MASK].",
        "History": "World War II ended in [MASK].",
        "Food": "Pizza originated in [MASK].",
        "Sports": "The FIFA World Cup is held every [MASK] years.",
        "Nature": "The largest mammal in the world is the [MASK].",
        "Music": "Mozart was a famous [MASK] composer.",
        "Space": "The first person to walk on the [MASK] was Neil Armstrong."
    }
    
    selected_sample = st.sidebar.selectbox("Choose a sample:", ["Custom"] + list(sample_texts.keys()))
    
    # Number of predictions to show
    num_predictions = st.sidebar.slider("Number of predictions:", 1, 10, 5)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Text")
        
        # Text input
        if selected_sample == "Custom":
            default_text = ""
            placeholder = "Enter your text with [MASK] token(s)..."
        else:
            default_text = sample_texts[selected_sample]
            placeholder = "Your text will appear here..."
        
        input_text = st.text_area(
            "Text with [MASK] token:",
            value=default_text,
            height=100,
            placeholder=placeholder,
            help="Use [MASK] to indicate where you want the model to predict a word."
        )
        
        # Quick sample buttons
        if selected_sample == "Custom":
            st.markdown("**Quick samples:**")
            cols = st.columns(3)
            quick_samples = [
                "The capital of France is [MASK].",
                "I love to eat [MASK] for breakfast.",
                "The [MASK] is shining brightly today."
            ]
            for i, sample in enumerate(quick_samples):
                with cols[i]:
                    if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                        st.session_state.input_text = sample
                        st.experimental_rerun()
    
    with col2:
        st.subheader("🎯 Quick Stats")
        if input_text:
            mask_count = input_text.count('[MASK]')
            word_count = len(input_text.split())
            char_count = len(input_text)
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>{mask_count}</h3>
                <p>MASK Tokens</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>{word_count}</h3>
                <p>Total Words</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        if input_text.strip():
            if "[MASK]" not in input_text:
                st.warning("⚠️ Please include at least one [MASK] token in your text!")
            else:
                with st.spinner("🤔 AI is thinking... Generating predictions..."):
                    results = predict_mask(input_text)
                
                if results:
                    # Limit results to user selection
                    results = results[:num_predictions]
                    
                    # Add to history
                    add_to_history(input_text, results)
                    
                    # Display original text with highlighted mask
                    st.subheader("📄 Original Text")
                    highlighted_text = highlight_mask_in_text(input_text)
                    st.markdown(f'<div style="font-size: 1.2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">{highlighted_text}</div>', unsafe_allow_html=True)
                    
                    # Create tabs for different views
                    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "📊 Confidence Chart", "☁️ Word Cloud", "📋 Detailed Results"])
                    
                    with tab1:
                        st.subheader("🎯 Top Predictions")
                        
                        for i, result in enumerate(results, 1):
                            predicted_sentence = result['sequence']
                            confidence = result['score']
                            token = result['token_str']
                            
                            # Create prediction card
                            st.markdown(f"""
                            <div class="prediction-card">
                                <div class="prediction-text">
                                    <strong>#{i}:</strong> {predicted_sentence}
                                </div>
                                <div class="prediction-score">
                                    <strong>Token:</strong> "{token}" | <strong>Confidence:</strong> {confidence:.4f} ({confidence*100:.1f}%)
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {confidence*100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab2:
                        st.subheader("📊 Confidence Analysis")
                        fig_conf = create_confidence_chart(results)
                        if fig_conf:
                            st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with tab3:
                        st.subheader("☁️ Word Visualization")
                        fig_cloud = create_word_cloud_style_viz(results)
                        if fig_cloud:
                            st.plotly_chart(fig_cloud, use_container_width=True)
                    
                    with tab4:
                        st.subheader("📋 Detailed Results")
                        
                        # Create DataFrame
                        df_results = pd.DataFrame([
                            {
                                'Rank': i+1,
                                'Predicted Token': res['token_str'],
                                'Complete Sentence': res['sequence'],
                                'Confidence': f"{res['score']:.6f}",
                                'Percentage': f"{res['score']*100:.2f}%"
                            }
                            for i, res in enumerate(results)
                        ])
                        
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Download button
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"fill_mask_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ No predictions generated. Please check your input text.")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # History section
    if st.session_state.prediction_history:
        with st.expander("📚 Prediction History"):
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
    
    # Information section
    with st.expander("ℹ️ About Fill-Mask Task"):
        st.markdown("""
        <div class="info-box">
        <h3>🎯 What is Fill-Mask?</h3>
        <p>Fill-Mask is a natural language processing task where an AI model predicts the most likely word(s) to fill in blanks in a sentence. This technique is fundamental to language understanding and is used in various applications.</p>
        
        <h3>🔧 How it Works:</h3>
        <ul>
        <li><strong>BERT Model:</strong> Uses bidirectional context to understand the meaning</li>
        <li><strong>Attention Mechanism:</strong> Focuses on relevant parts of the sentence</li>
        <li><strong>Probability Distribution:</strong> Generates confidence scores for each prediction</li>
        </ul>
        
        <h3>💡 Applications:</h3>
        <ul>
        <li>Auto-completion in search engines</li>
        <li>Grammar and spell checking</li>
        <li>Content generation assistance</li>
        <li>Language learning tools</li>
        <li>Text summarization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()