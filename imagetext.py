import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import PIL.Image
import io
import base64
from datetime import datetime
import json
import os
import time

# Set page config
st.set_page_config(
    page_title="AI Text-to-Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prompt-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .prompt-text {
        color: white;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
        margin: 0;
    }
    
    .image-container {
        border: 3px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .generation-stats {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .sample-prompt {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4299e1;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-prompt:hover {
        background: #edf2f7;
        transform: translateX(5px);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-top: 4px solid #667eea;
    }
    
    .gallery-item {
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .gallery-item:hover {
        transform: scale(1.05);
    }
    
    .progress-container {
        background: #f7fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .model-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""

@st.cache_resource
def load_model():
    """Load the Stable Diffusion model with caching"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        with st.spinner("Loading Stable Diffusion model... This may take several minutes on first run."):
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(device)
            
            # Enable memory efficient attention if available
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            
            return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_image(prompt, num_steps=20, guidance_scale=7.5, width=512, height=512):
    """Generate image from text prompt"""
    if not st.session_state.pipeline:
        return None
    
    try:
        # Generate image
        with torch.no_grad():
            result = st.session_state.pipeline(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )
        
        return result.images[0]
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def add_to_history(prompt, image, generation_time, settings):
    """Add generated image to history"""
    # Convert image to base64 for storage
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.session_state.generation_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'image_base64': img_str,
        'generation_time': generation_time,
        'settings': settings
    })

def display_image_gallery():
    """Display gallery of generated images"""
    if not st.session_state.generation_history:
        st.info("No images generated yet. Create your first image above!")
        return
    
    st.subheader("🖼️ Generated Images Gallery")
    
    # Create columns for gallery
    cols = st.columns(3)
    
    for i, item in enumerate(reversed(st.session_state.generation_history[-9:])):  # Show last 9 images
        with cols[i % 3]:
            # Decode base64 image
            img_data = base64.b64decode(item['image_base64'])
            img = PIL.Image.open(io.BytesIO(img_data))
            
            st.image(img, caption=f"'{item['prompt'][:30]}...'", use_column_width=True)
            st.caption(f"Generated: {item['timestamp']}")

# Main app
def main():
    st.markdown('<h1 class="main-header">🎨 AI Text-to-Image Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your words into stunning visual art with AI</p>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_model()
    
    if st.session_state.pipeline is None:
        st.error("Failed to load the Stable Diffusion model. Please check your system requirements.")
        return
    
    # Display system info
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <div class="model-info">
        <h3>🚀 Model Information</h3>
        <p><strong>Model:</strong> Stable Diffusion v1.5</p>
        <p><strong>Device:</strong> {device_info}</p>
        <p><strong>Status:</strong> Ready for generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar settings
    st.sidebar.title("🎛️ Generation Settings")
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        num_steps = st.slider("Inference Steps", 10, 50, 20, help="More steps = better quality, slower generation")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, help="Higher values = more prompt adherence")
        width = st.selectbox("Width", [512, 576, 640, 704, 768], index=0)
        height = st.selectbox("Height", [512, 576, 640, 704, 768], index=0)
    
    # Sample prompts
    st.sidebar.subheader("💡 Sample Prompts")
    sample_prompts = {
        "Futuristic City": "A futuristic city at sunset with flying cars, neon lights, cyberpunk style",
        "Fantasy Landscape": "A magical forest with glowing mushrooms, fairy lights, and a crystal clear lake",
        "Space Scene": "An astronaut floating in space with colorful nebulae and distant galaxies",
        "Artistic Portrait": "A portrait of a woman with flowing hair, oil painting style, Renaissance art",
        "Nature Scene": "A serene mountain landscape with snow-capped peaks and a flowing river",
        "Steampunk": "A steampunk airship flying through clouds, brass and copper details, Victorian era",
        "Underwater": "An underwater coral reef with tropical fish and rays of sunlight",
        "Architecture": "A modern glass skyscraper with unique geometric patterns and blue sky"
    }
    
    for category, prompt in sample_prompts.items():
        if st.sidebar.button(f"📝 {category}", key=f"sample_{category}"):
            st.session_state.selected_prompt = prompt
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Enter Your Prompt")
        
        # Text input
        prompt = st.text_area(
            "Describe the image you want to generate:",
            value=st.session_state.get('selected_prompt', ''),
            height=100,
            placeholder="A beautiful sunset over a mountain range with a lake in the foreground...",
            help="Be descriptive! Include details about style, colors, mood, and composition."
        )
        
        # Clear selected prompt after use
        if 'selected_prompt' in st.session_state:
            del st.session_state.selected_prompt
        
        # Generation button
        if st.button("🎨 Generate Image", type="primary", use_container_width=True):
            if prompt.strip():
                # Display current prompt
                st.markdown(f"""
                <div class="prompt-container">
                    <p class="prompt-text">Generating: "{prompt}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar and generation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                # Update progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("Initializing generation...")
                    elif i < 80:
                        status_text.text("Creating your masterpiece...")
                    else:
                        status_text.text("Adding final touches...")
                    time.sleep(0.1)
                
                # Generate image
                with st.spinner("Generating image..."):
                    image = generate_image(
                        prompt, 
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height
                    )
                
                generation_time = time.time() - start_time
                
                if image:
                    # Store current image and prompt
                    st.session_state.current_image = image
                    st.session_state.current_prompt = prompt
                    
                    # Add to history
                    settings = {
                        'steps': num_steps,
                        'guidance': guidance_scale,
                        'width': width,
                        'height': height
                    }
                    add_to_history(prompt, image, generation_time, settings)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Image generated successfully in {generation_time:.1f} seconds!")
                else:
                    st.error("❌ Failed to generate image. Please try again.")
            else:
                st.warning("⚠️ Please enter a prompt to generate an image.")
    
    with col2:
        st.subheader("📊 Generation Stats")
        
        if st.session_state.generation_history:
            total_images = len(st.session_state.generation_history)
            avg_time = sum(item['generation_time'] for item in st.session_state.generation_history) / total_images
            
            st.markdown(f"""
            <div class="generation-stats">
                <h3>{total_images}</h3>
                <p>Images Generated</p>
                <h3>{avg_time:.1f}s</h3>
                <p>Average Generation Time</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="generation-stats">
                <h3>0</h3>
                <p>Images Generated</p>
                <p>Generate your first image!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display current image
    if st.session_state.current_image:
        st.subheader("🖼️ Generated Image")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.current_image, caption=f"Prompt: {st.session_state.current_prompt}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        buffered = io.BytesIO()
        st.session_state.current_image.save(buffered, format="PNG")
        st.download_button(
            label="📥 Download Image",
            data=buffered.getvalue(),
            file_name=f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    # Image gallery
    if st.session_state.generation_history:
        st.markdown("---")
        display_image_gallery()
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.generation_history = []
            st.session_state.current_image = None
            st.session_state.current_prompt = ""
            st.experimental_rerun()
    
    # Tips and information
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        ### 🎯 Prompt Writing Tips:
        - **Be specific**: Instead of "a dog", try "a golden retriever puppy sitting in a meadow"
        - **Include style**: Add terms like "oil painting", "photorealistic", "watercolor", "digital art"
        - **Describe lighting**: "soft morning light", "dramatic shadows", "neon lighting"
        - **Add mood**: "serene", "mysterious", "vibrant", "ethereal"
        - **Mention composition**: "close-up", "wide angle", "bird's eye view"
        
        ### ⚙️ Settings Guide:
        - **Inference Steps**: 20-30 for good quality, 40+ for maximum detail
        - **Guidance Scale**: 7-10 for balanced results, higher for more prompt adherence
        - **Resolution**: Higher resolutions take more time and memory
        
        ### 🚀 Performance Tips:
        - Use GPU if available for faster generation
        - Lower resolution for quicker experiments
        - Batch multiple variations of similar prompts
        """)

if __name__ == "__main__":
    main()