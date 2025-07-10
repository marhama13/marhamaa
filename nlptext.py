import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import base64
import time
import random
from datetime import datetime
import gc
import re

# Set page config
st.set_page_config(
    page_title="NLP AI Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0 2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .task-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .entity-highlight {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin: 0.1rem;
        display: inline-block;
        font-weight: bold;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stButton > button {
        border-radius: 20px;
        height: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .feature-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {
        'classifier': False,
        'ner': False,
        'fill_mask': False,
        'image_gen': False
    }
    st.session_state.models = {}
    st.session_state.generated_images = []

# Model loading functions
@st.cache_resource
def load_text_classifier():
    """Load text classification model"""
    try:
        return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None

@st.cache_resource
def load_ner_model():
    """Load NER model"""
    try:
        return pipeline("ner", grouped_entities=True)
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

@st.cache_resource
def load_fill_mask_model():
    """Load fill-mask model"""
    try:
        return pipeline("fill-mask", model="bert-base-uncased")
    except Exception as e:
        st.error(f"Error loading fill-mask model: {e}")
        return None

@st.cache_resource
def load_image_generation_model():
    """Load Stable Diffusion model"""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        return pipe
    except Exception as e:
        st.error(f"Error loading image generation model: {e}")
        return None

def create_score_visualization(scores, title="Confidence Scores"):
    """Create a horizontal bar chart for scores"""
    df = pd.DataFrame(list(scores.items()), columns=['Category', 'Score'])
    df = df.sort_values('Score', ascending=True)
    
    fig = px.bar(
        df, 
        x='Score', 
        y='Category',
        orientation='h',
        title=title,
        color='Score',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_x=0.5
    )
    
    return fig

def create_entity_visualization(entities):
    """Create visualization for named entities"""
    entity_counts = Counter([entity['entity_group'] for entity in entities])
    
    fig = px.bar(
        x=list(entity_counts.keys()),
        y=list(entity_counts.values()),
        title="Named Entity Distribution",
        color=list(entity_counts.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        xaxis_title="Entity Type",
        yaxis_title="Count"
    )
    
    return fig

def highlight_entities(text, entities):
    """Highlight entities in text"""
    highlighted_text = text
    offset = 0
    
    for entity in sorted(entities, key=lambda x: x['start']):
        start = entity['start'] + offset
        end = entity['end'] + offset
        entity_text = entity['word']
        entity_type = entity['entity_group']
        
        highlight = f'<span class="entity-highlight" title="{entity_type} ({entity["score"]:.2f})">{entity_text}</span>'
        highlighted_text = highlighted_text[:start] + highlight + highlighted_text[end:]
        offset += len(highlight) - (end - start)
    
    return highlighted_text

def main():
    # Header
    st.markdown('<h1 class="header-gradient">🤖 NLP AI Suite</h1>', unsafe_allow_html=True)
    st.markdown("**Complete Natural Language Processing toolkit with multiple AI models**")
    
    # Feature badges
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="feature-badge">📝 Text Classification</span>
        <span class="feature-badge">🏷️ Named Entity Recognition</span>
        <span class="feature-badge">🔍 Fill Mask Prediction</span>
        <span class="feature-badge">🎨 Text-to-Image</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model management
    st.sidebar.header("🤖 Model Management")
    
    # Model loading buttons
    models_to_load = {
        'Text Classifier': ('classifier', load_text_classifier),
        'NER Model': ('ner', load_ner_model),
        'Fill Mask Model': ('fill_mask', load_fill_mask_model),
        'Image Generator': ('image_gen', load_image_generation_model)
    }
    
    for model_name, (key, loader) in models_to_load.items():
        if not st.session_state.models_loaded[key]:
            if st.sidebar.button(f"Load {model_name}", key=f"load_{key}"):
                with st.spinner(f"Loading {model_name}..."):
                    model = loader()
                    if model:
                        st.session_state.models[key] = model
                        st.session_state.models_loaded[key] = True
                        st.sidebar.success(f"✅ {model_name} loaded!")
                        st.rerun()
        else:
            st.sidebar.success(f"✅ {model_name} Ready")
    
    # System info
    st.sidebar.subheader("💻 System Info")
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"**Device:** {device}")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Classification", "🏷️ Named Entity Recognition", "🔍 Fill Mask", "🎨 Text-to-Image"])
    
    # Tab 1: Text Classification
    with tab1:
        st.header("📝 Text Classification")
        st.markdown("Classify text into custom categories using zero-shot learning")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Custom categories
            st.subheader("📋 Categories")
            custom_categories = st.text_area(
                "Enter categories (one per line):",
                value="food waste\nnature\nwaste management\ntechnology\nhealth",
                height=100
            )
            
            categories = [cat.strip() for cat in custom_categories.split('\n') if cat.strip()]
            
            # Text input
            st.subheader("📝 Input Text")
            text_input = st.text_area(
                "Enter text to classify:",
                value="Composting organic leftovers helps reduce food waste and improve soil health.",
                height=120
            )
            
            # Classify button
            if st.button("🔍 Classify Text", key="classify_btn"):
                if st.session_state.models_loaded['classifier'] and text_input.strip():
                    with st.spinner("Classifying text..."):
                        result = st.session_state.models['classifier'](text_input, candidate_labels=categories)
                        
                        # Display results
                        st.subheader("📊 Results")
                        
                        # Top prediction
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>🎯 Predicted Category: {result['labels'][0].upper()}</h3>
                            <p>Confidence: {result['scores'][0]:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # All scores
                        scores = dict(zip(result['labels'], result['scores']))
                        fig = create_score_visualization(scores, "Classification Confidence")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed table
                        df = pd.DataFrame([
                            {'Category': cat, 'Confidence': f"{score:.1%}", 'Score': score}
                            for cat, score in scores.items()
                        ]).sort_values('Score', ascending=False)
                        
                        st.dataframe(df[['Category', 'Confidence']], use_container_width=True, hide_index=True)
                
                elif not st.session_state.models_loaded['classifier']:
                    st.error("Please load the Text Classifier model first!")
                else:
                    st.warning("Please enter some text to classify!")
        
        with col2:
            st.subheader("💡 Examples")
            examples = [
                "Solar panels generate clean renewable energy",
                "Recycling plastic bottles reduces ocean pollution",
                "AI chatbots are transforming customer service",
                "Regular exercise improves cardiovascular health",
                "Organic farming preserves soil biodiversity"
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"📄 {example[:30]}...", key=f"example_{i}"):
                    st.session_state.example_text = example
                    st.rerun()
    
    # Tab 2: Named Entity Recognition
    with tab2:
        st.header("🏷️ Named Entity Recognition")
        st.markdown("Extract and classify named entities from text")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Input Text")
            ner_text = st.text_area(
                "Enter text for NER analysis:",
                value="BTS performed at Wembley Stadium in London, and their leader RM gave a speech at the United Nations in New York.",
                height=120
            )
            
            if st.button("🔍 Analyze Entities", key="ner_btn"):
                if st.session_state.models_loaded['ner'] and ner_text.strip():
                    with st.spinner("Analyzing entities..."):
                        entities = st.session_state.models['ner'](ner_text)
                        
                        if entities:
                            st.subheader("📊 Results")
                            
                            # Highlighted text
                            st.markdown("**📝 Text with Highlighted Entities:**")
                            highlighted = highlight_entities(ner_text, entities)
                            st.markdown(highlighted, unsafe_allow_html=True)
                            
                            # Visualization
                            fig = create_entity_visualization(entities)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Entity details
                            st.markdown("**🏷️ Entity Details:**")
                            entity_df = pd.DataFrame([
                                {
                                    'Entity': entity['word'],
                                    'Type': entity['entity_group'],
                                    'Confidence': f"{entity['score']:.1%}"
                                }
                                for entity in entities
                            ])
                            st.dataframe(entity_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No entities found in the text.")
                
                elif not st.session_state.models_loaded['ner']:
                    st.error("Please load the NER Model first!")
                else:
                    st.warning("Please enter some text to analyze!")
        
        with col2:
            st.subheader("🏷️ Entity Types")
            entity_types = [
                "**PER** - Person names",
                "**ORG** - Organizations",
                "**LOC** - Locations",
                "**MISC** - Miscellaneous"
            ]
            
            for entity_type in entity_types:
                st.markdown(entity_type)
            
            st.subheader("💡 Examples")
            ner_examples = [
                "Apple Inc. was founded by Steve Jobs in California.",
                "The Eiffel Tower in Paris attracts millions of visitors.",
                "NASA launched the Mars rover from Kennedy Space Center."
            ]
            
            for i, example in enumerate(ner_examples):
                if st.button(f"📄 {example[:25]}...", key=f"ner_example_{i}"):
                    st.session_state.ner_example = example
                    st.rerun()
    
    # Tab 3: Fill Mask
    with tab3:
        st.header("🔍 Fill Mask Prediction")
        st.markdown("Predict missing words in sentences using BERT")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Input Text")
            st.info("Use [MASK] to indicate the word you want to predict")
            
            mask_text = st.text_area(
                "Enter text with [MASK] token:",
                value="The Eiffel Tower is located in [MASK].",
                height=120
            )
            
            if st.button("🔍 Predict Mask", key="mask_btn"):
                if st.session_state.models_loaded['fill_mask'] and mask_text.strip():
                    if '[MASK]' in mask_text:
                        with st.spinner("Predicting masked word..."):
                            results = st.session_state.models['fill_mask'](mask_text)
                            
                            st.subheader("📊 Top Predictions")
                            
                            # Display predictions
                            for i, result in enumerate(results[:5]):
                                confidence = result['score']
                                sentence = result['sequence']
                                
                                st.markdown(f"""
                                <div class="task-card">
                                    <h4>#{i+1} - {confidence:.1%} confidence</h4>
                                    <p style="font-size: 1.1rem;"><strong>"{sentence}"</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Visualization
                            pred_data = {
                                'Prediction': [f"#{i+1}" for i in range(len(results[:5]))],
                                'Confidence': [r['score'] for r in results[:5]],
                                'Word': [r['token_str'] for r in results[:5]]
                            }
                            
                            fig = px.bar(
                                pred_data,
                                x='Prediction',
                                y='Confidence',
                                title="Prediction Confidence",
                                text='Word',
                                color='Confidence',
                                color_continuous_scale='viridis'
                            )
                            fig.update_traces(textposition='outside')
                            fig.update_layout(showlegend=False, title_x=0.5)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Please include [MASK] in your text!")
                
                elif not st.session_state.models_loaded['fill_mask']:
                    st.error("Please load the Fill Mask Model first!")
                else:
                    st.warning("Please enter some text with [MASK]!")
        
        with col2:
            st.subheader("💡 Examples")
            mask_examples = [
                "The capital of France is [MASK].",
                "I love to eat [MASK] for breakfast.",
                "The weather today is [MASK].",
                "My favorite [MASK] is Python.",
                "The [MASK] is shining brightly."
            ]
            
            for i, example in enumerate(mask_examples):
                if st.button(f"📄 {example}", key=f"mask_example_{i}"):
                    st.session_state.mask_example = example
                    st.rerun()
    
    # Tab 4: Text-to-Image
    with tab4:
        st.header("🎨 Text-to-Image Generation")
        st.markdown("Generate images from text descriptions using Stable Diffusion")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Image Description")
            
            image_prompt = st.text_area(
                "Describe the image you want to generate:",
                value="A futuristic city at sunset with flying cars",
                height=120
            )
            
            # Generation parameters
            col_a, col_b = st.columns(2)
            
            with col_a:
                num_steps = st.slider("Inference Steps", 10, 50, 20, 5)
            
            with col_b:
                guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
            
            if st.button("🎨 Generate Image", key="generate_btn"):
                if st.session_state.models_loaded['image_gen'] and image_prompt.strip():
                    with st.spinner("Creating your image... This may take a minute."):
                        start_time = time.time()
                        
                        # Generate image
                        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                            result = st.session_state.models['image_gen'](
                                image_prompt,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance_scale,
                                height=512,
                                width=512
                            )
                        
                        image = result.images[0]
                        generation_time = time.time() - start_time
                        
                        # Display image
                        st.image(image, caption=f"Generated in {generation_time:.1f}s")
                        
                        # Download button
                        buf = io.BytesIO()
                        image.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download Image",
                            data=buf.getvalue(),
                            file_name=f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                        
                        # Clear GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                
                elif not st.session_state.models_loaded['image_gen']:
                    st.error("Please load the Image Generator model first!")
                else:
                    st.warning("Please enter a description!")
        
        with col2:
            st.subheader("💡 Prompt Tips")
            st.info("""
            **For better results:**
            - Be specific about details
            - Mention art style (realistic, cartoon, oil painting)
            - Include lighting (golden hour, dramatic)
            - Add quality keywords (high quality, detailed, 4K)
            - Describe mood and atmosphere
            """)
            
            st.subheader("🎨 Example Prompts")
            image_examples = [
                "A serene mountain landscape at golden hour",
                "A cyberpunk city with neon lights",
                "A magical forest with glowing mushrooms",
                "A steampunk airship in the clouds",
                "A cozy cottage in winter snow"
            ]
            
            for i, example in enumerate(image_examples):
                if st.button(f"🖼️ {example}", key=f"img_example_{i}"):
                    st.session_state.image_example = example
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🤖 <strong>NLP AI Suite</strong> - Complete Natural Language Processing Toolkit</p>
        <p>Built with ❤️ using Streamlit, Hugging Face Transformers, and Stable Diffusion</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()