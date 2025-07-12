import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Solar Panel Performance Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("☀️ Solar Panel Performance Analysis Dashboard")
st.markdown("**Analyze solar panel energy output across different seasons with interactive visualizations and machine learning predictions**")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Data Visualization", "Energy Prediction", "Season Classification"])

# Data generation functions
@st.cache_data
def generate_solar_data():
    """Generate synthetic solar panel data for all seasons"""
    
    # Summer data
    summer_months_days = {'March': 31, 'April': 30, 'May': 31, 'June': 30}
    summer_ranges = {
        'irradiance': (600, 1000),
        'humidity': (10, 50),
        'wind_speed': (0, 5),
        'ambient_temperature': (30, 45),
        'tilt_angle': (10, 40),
    }
    
    def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
        return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 0.1 * ambient_temp - 0.03 * abs(tilt_angle - 30))
    
    # Winter data
    winter_months_days = {'November': 30, 'December': 31, 'January': 31, 'February': 28}
    winter_ranges = {
        'irradiance': (300, 700),
        'humidity': (30, 70),
        'wind_speed': (1, 6),
        'ambient_temperature': (5, 20),
        'tilt_angle': (10, 40),
    }
    
    def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
        return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))
    
    # Monsoon data
    monsoon_months_days = {'July': 31, 'August': 31, 'September': 30, 'October': 31}
    monsoon_ranges = {
        'irradiance': (100, 600),
        'humidity': (70, 100),
        'wind_speed': (2, 8),
        'ambient_temperature': (20, 35),
        'tilt_angle': (10, 40),
    }
    
    def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
        return (0.15 * irradiance - 0.1 * humidity + 0.01 * wind_speed + 0.05 * ambient_temp - 0.04 * abs(tilt_angle - 30))
    
    # Generate data for all seasons
    all_data = []
    
    for season, months_days, ranges, calc_func in [
        ('summer', summer_months_days, summer_ranges, calc_kwh_summer),
        ('winter', winter_months_days, winter_ranges, calc_kwh_winter),
        ('monsoon', monsoon_months_days, monsoon_ranges, calc_kwh_monsoon)
    ]:
        for month, days in months_days.items():
            for _ in range(days):
                irr = np.random.uniform(*ranges['irradiance'])
                hum = np.random.uniform(*ranges['humidity'])
                wind = np.random.uniform(*ranges['wind_speed'])
                temp = np.random.uniform(*ranges['ambient_temperature'])
                tilt = np.random.uniform(*ranges['tilt_angle'])
                
                kwh = calc_func(irr, hum, wind, temp, tilt)
                
                all_data.append({
                    'irradiance': round(irr, 2),
                    'humidity': round(hum, 2),
                    'wind_speed': round(wind, 2),
                    'ambient_temperature': round(temp, 2),
                    'tilt_angle': round(tilt, 2),
                    'kwh': round(kwh, 2),
                    'season': season,
                    'month': month
                })
    
    return pd.DataFrame(all_data)

# Load data
df = generate_solar_data()

# Data Overview Page
if page == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Average kWh Output", f"{df['kwh'].mean():.2f}")
    with col3:
        st.metric("Seasons Analyzed", df['season'].nunique())
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    st.subheader("Season Distribution")
    season_counts = df['season'].value_counts()
    fig = px.pie(values=season_counts.values, names=season_counts.index, 
                 title="Distribution of Records by Season")
    st.plotly_chart(fig, use_container_width=True)

# Data Visualization Page
elif page == "Data Visualization":
    st.header("📈 Data Visualization")
    
    # Energy output by season
    st.subheader("Energy Output by Season")
    fig = px.box(df, x='season', y='kwh', color='season',
                 title="Solar Panel Energy Output Distribution by Season")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    numeric_cols = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Daily energy output
    st.subheader("Daily Energy Output Across All Days")
    fig = px.bar(df, x=df.index, y='kwh', color='season',
                 title="Day-wise Solar Panel Energy Output",
                 labels={'x': 'Day Index', 'kwh': 'Energy Output (kWh)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions by Season")
    feature = st.selectbox("Select Feature to Analyze", 
                          ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle'])
    
    fig = px.histogram(df, x=feature, color='season', nbins=30,
                      title=f"{feature.replace('_', ' ').title()} Distribution by Season")
    st.plotly_chart(fig, use_container_width=True)

# Energy Prediction Page
elif page == "Energy Prediction":
    st.header("🔮 Energy Output Prediction")
    
    # Prepare data for regression
    X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
    y = df['kwh']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted Energy Output")
    fig = px.scatter(x=y_test, y=y_pred, 
                     title="Actual vs Predicted kWh Output",
                     labels={'x': 'Actual kWh', 'y': 'Predicted kWh'})
    
    # Add diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="red", dash="dash"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance (Model Coefficients)")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    fig = px.bar(feature_importance, x='Coefficient', y='Feature', orientation='h',
                 title="Linear Regression Coefficients")
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive prediction
    st.subheader("🎯 Make Your Own Prediction")
    st.write("Adjust the parameters below to predict energy output:")
    
    col1, col2 = st.columns(2)
    with col1:
        irradiance = st.slider("Irradiance", 100, 1000, 500)
        humidity = st.slider("Humidity (%)", 10, 100, 50)
        wind_speed = st.slider("Wind Speed", 0, 8, 3)
    
    with col2:
        ambient_temp = st.slider("Ambient Temperature (°C)", 5, 45, 25)
        tilt_angle = st.slider("Tilt Angle (°)", 10, 40, 30)
    
    # Make prediction
    user_input = np.array([[irradiance, humidity, wind_speed, ambient_temp, tilt_angle]])
    prediction = model.predict(user_input)[0]
    
    st.success(f"🔋 Predicted Energy Output: **{prediction:.2f} kWh**")

# Season Classification Page
elif page == "Season Classification":
    st.header("🌦️ Season Classification")
    
    # Prepare data for classification
    X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']]
    y = df['season']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model performance
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Classification Accuracy", f"{accuracy:.4f}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted Season')
    ax.set_ylabel('Actual Season')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Interactive season prediction
    st.subheader("🎯 Predict Season from Parameters")
    st.write("Enter the parameters to predict the season:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        irradiance = st.number_input("Irradiance", 100, 1000, 500)
        humidity = st.number_input("Humidity (%)", 10, 100, 50)
    
    with col2:
        wind_speed = st.number_input("Wind Speed", 0, 8, 3)
        ambient_temp = st.number_input("Ambient Temperature (°C)", 5, 45, 25)
    
    with col3:
        tilt_angle = st.number_input("Tilt Angle (°)", 10, 40, 30)
        kwh = st.number_input("Energy Output (kWh)", 0, 200, 100)
    
    # Make prediction
    user_input = np.array([[irradiance, humidity, wind_speed, ambient_temp, tilt_angle, kwh]])
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0]
    
    predicted_season = le.inverse_transform([prediction])[0]
    
    st.success(f"🌦️ Predicted Season: **{predicted_season.title()}**")
    
    # Show probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Season': le.classes_,
        'Probability': prediction_proba
    }).sort_values('Probability', ascending=False)
    
    fig = px.bar(prob_df, x='Season', y='Probability', 
                 title="Season Prediction Probabilities")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Solar Panel Performance Analysis Dashboard** | Built with Streamlit 🚀")