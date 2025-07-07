import streamlit as st

# Set page config
st.set_page_config(
    page_title="Home Energy Calculator",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .energy-display {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        margin: 20px 0;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 5px solid #2E86AB;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏠 Home Energy Calculator</h1>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 👤 Personal Information")
    name = st.text_input("Enter your name:", placeholder="Your name here...")
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25)
    city = st.text_input("Enter your city:", placeholder="Your city here...")

with col2:
    st.markdown("### 🏘 Housing Information")
    living = st.selectbox("Are you living in a house or flat?", ["", "house", "flat"])
    
    if living:
        st.success(f"You are living in a {living}")
        bhk = st.selectbox("How many BHK do you have?", [1, 2, 3])
        st.info(f"You have {bhk} BHK")

# Energy calculation section
if living and bhk:
    st.markdown("### ⚡ Energy Calculation")
    
    # Calculate base energy
    base_energy = bhk * 0.4 + bhk * 0.8
    
    # Create columns for appliances
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("#### 🌡 Air Conditioner")
        ac = st.radio("Do you have AC?", ["No", "Yes"], key="ac")
        
    with col4:
        st.markdown("#### 🧊 Refrigerator")
        fridge = st.radio("Do you have a fridge?", ["No", "Yes"], key="fridge")
    
    # Calculate total energy
    total_energy = base_energy
    ac_energy = 3 if ac == "Yes" else 0
    fridge_energy = 4 if fridge == "Yes" else 0
    
    final_energy = total_energy + ac_energy + fridge_energy
    
    # Display energy breakdown
    st.markdown("### 📊 Energy Breakdown")
    
    # Create metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Base Energy", f"{base_energy:.1f} kWh", help="Based on your BHK size")
    
    with col6:
        st.metric("AC Energy", f"{ac_energy} kWh", help="Air conditioner consumption")
    
    with col7:
        st.metric("Fridge Energy", f"{fridge_energy} kWh", help="Refrigerator consumption")
    
    with col8:
        st.metric("Total Energy", f"{final_energy:.1f} kWh", help="Your total energy consumption")
    
    # Final energy display
    st.markdown(f"""
    <div class="energy-display">
        🔋 Your Total Energy Consumption: {final_energy:.1f} kWh
    </div>
    """, unsafe_allow_html=True)
    
    # Additional information
    if name:
        st.markdown(f"""
        <div class="info-box">
            <strong>Summary for {name}:</strong><br>
            📍 Location: {city}<br>
            🏠 Housing: {bhk} BHK {living}<br>
            ⚡ Monthly Energy: {final_energy:.1f} kWh<br>
            💰 Estimated Monthly Cost: ₹{final_energy * 8:.0f} (@ ₹8/kWh)
        </div>
        """, unsafe_allow_html=True)
        
        # Energy efficiency tips
        st.markdown("### 💡 Energy Saving Tips")
        tips = [
            "Use LED bulbs instead of incandescent ones",
            "Set AC temperature to 24°C or higher",
            "Regular maintenance of appliances improves efficiency",
            "Use natural light during daytime",
            "Unplug devices when not in use"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")

# Sidebar with additional info
st.sidebar.markdown("### ℹ About This Calculator")
st.sidebar.info("""
This calculator estimates your home energy consumption based on:
- Number of BHK
- Major appliances (AC, Fridge)
- Basic lighting and fan usage

The calculations are approximate and may vary based on actual usage patterns.
""")

st.sidebar.markdown("### 📈 Energy Efficiency Scale")
if 'final_energy' in locals():
    if final_energy < 200:
        st.sidebar.success("🟢 Excellent - Low consumption")
    elif final_energy < 400:
        st.sidebar.warning("🟡 Good - Moderate consumption")
    else:
        st.sidebar.error("🔴 High - Consider energy saving measures")
