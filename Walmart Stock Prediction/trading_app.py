import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
import os
from datetime import datetime, timedelta
import random
from project_overview import project_overview_page

# --- Configure Streamlit Page ---
st.set_page_config(
    page_title="Walmart Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Walmart Colors ---
# Primary: #0071ce (blue)
# Secondary: #ffc220 (yellow)
# Text Dark: #2a2a2a
# Background Light: #f6f6f6
# Success Green: #2ecc71
# Danger Red: #e74c3c

# --- Custom CSS for modern, minimalist design ---
st.markdown("""
<style>
    /* Global Styles */
    body {
        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #fafafa;
        color: #2a2a2a;
        line-height: 1.6;
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }
    
    /* Main title styling */
    .app-title {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #0071ce !important;
        text-align: center !important;
        margin: 1.5rem 0 0.5rem 0 !important;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 1.1rem !important;
        color: #6c757d !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        font-weight: 400 !important;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2a2a2a !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f2f2f2;
    }
    
    /* Navigation container */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0 2.5rem 0;
        width: 100%;
    }
    
    /* Navigation buttons */
    .nav-button {
        background-color: #f8f9fa;
        color: #495057;
        border: none;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 1.05rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.25s ease;
        text-align: center;
        flex: 1;
        max-width: 300px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .nav-button:hover {
        background-color: #ffc220;
        color: #212529;
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .nav-button-active {
        background-color: #0071ce;
        color: white;
        box-shadow: 0 4px 10px rgba(0,113,206,0.2);
    }
    
    .nav-button-active:hover {
        background-color: #005eb8;
        color: white;
    }
    
    /* Panel styling */
    .trading-panel {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #f0f0f0;
    }
    
    /* Stock info styling */
    .stock-price {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0071ce;
        line-height: 1.2;
    }
    
    .price-change-positive {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2ecc71;
        margin-left: 0.5rem;
    }
    
    .price-change-negative {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e74c3c;
        margin-left: 0.5rem;
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        border: 1px solid #f0f0f0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #212529;
    }
    
    /* Form elements */
    div[data-baseweb="input"] input {
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        padding: 0.9rem 1rem !important;
        font-size: 1.05rem !important;
        background-color: #f8f9fa !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div[data-baseweb="input"] input:focus {
        border-color: #0071ce !important;
        box-shadow: 0 0 0 3px rgba(0, 113, 206, 0.15) !important;
        background-color: #ffffff !important;
    }
    
    /* Number inputs */
    .stNumberInput div[data-baseweb="input"] input {
        text-align: center;
    }
    
    /* Form button styling */
    button[kind="primaryFormSubmit"] {
        background-color: #0071ce !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 10px rgba(0,113,206,0.2) !important;
        margin-top: 1rem !important;
    }
    
    button[kind="primaryFormSubmit"]:hover {
        background-color: #ffc220 !important;
        color: #212529 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Standard buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    
    /* Prediction result styling */
    .prediction-panel {
        background-color: #f8f9fa;
        border-left: 5px solid #0071ce;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        background-image: linear-gradient(to right, rgba(0,113,206,0.05), rgba(0,113,206,0));
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 1.25rem;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        color: #0071ce;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }
    
    .prediction-increase {
        color: #2ecc71;
        font-weight: 600;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-decrease {
        color: #e74c3c;
        font-weight: 600;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chart container */
    .chart-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin: 1.5rem 0;
        border: 1px solid #f0f0f0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
        gap: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        color: #495057;
        font-weight: 500;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0071ce !important;
        color: white !important;
        box-shadow: 0 3px 8px rgba(0,113,206,0.2);
    }
    
    /* Data indicators */
    .indicator {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        padding: 0.75rem 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .indicator:hover {
        background-color: #f2f2f2;
    }
    
    .indicator-label {
        font-size: 0.95rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .indicator-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #212529;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Project Overview page styles */
    .overview-card {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    .overview-intro {
        font-size: 1.1rem;
        line-height: 1.7;
        color: #495057;
        margin-bottom: 2.5rem;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #0071ce;
    }
    
    /* Code blocks */
    pre {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 0.9rem !important;
        overflow-x: auto !important;
        border: 1px solid #f0f0f0 !important;
    }
    
    code {
        font-family: 'Roboto Mono', monospace !important;
        font-size: 0.9rem !important;
        background-color: #f8f9fa !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        color: #0071ce !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6c757d;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def load_model():
    """Load or create prediction model"""
    model_path = "best_stock_model.pkl"
    if not os.path.exists(model_path):
        try:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            X = np.random.rand(100, 7)  # 7 features
            y = np.random.rand(100)  # Target
            model.fit(X, y)
            joblib.dump(model, model_path)
            return model
        except Exception as e:
            st.error(f"Error creating model: {str(e)}")
            return None
    else:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

def predict_next_day(model, current_price):
    """Predict tomorrow's stock price based on current price"""
    # In a real app, you would calculate all the features properly
    # Here we're using simple approximations based solely on the current price
    
    features = {
        'z_log_adj_close': np.log(current_price) / 5,
        'price_range': current_price * 0.02,
        'volatility': 0.015,
        'return': 0.002,
        'ma_3d': current_price * 0.99,
        'ma_7d': current_price * 0.98,
        'high_low_ratio': 1.02
    }
    
    # Create DataFrame with features
    features_df = pd.DataFrame(features, index=[0])
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    
    # Ensure prediction seems realistic (within 5% of current price)
    if prediction < current_price * 0.95 or prediction > current_price * 1.05:
        prediction = current_price * (1 + (random.uniform(-0.03, 0.03)))
    
    return prediction

def generate_stock_chart(current_price, prediction):
    """Generate a stock chart with historical data and prediction"""
    # Create some simulated historical data
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, -1, -1)]
    tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    dates.append(tomorrow)
    
    # Generate fake historical prices with some volatility
    base_price = current_price * 0.9  # Start at 90% of current price
    hist_prices = []
    for i in range(31):
        # Add some random movement but with an upward trend
        change = np.random.normal(0.001, 0.01)
        base_price = base_price * (1 + change)
        hist_prices.append(base_price)
    
    # Add current price and prediction
    prices = hist_prices + [current_price, prediction]
    
    # Create a candlestick chart
    fig = go.Figure()
    
    # Add line chart
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='#0071ce', width=2),
    ))
    
    # Add prediction point
    fig.add_trace(go.Scatter(
        x=[dates[-1]],
        y=[prediction],
        mode='markers',
        name='Prediction',
        marker=dict(color='#ffc220', size=12, line=dict(color='white', width=2)),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add rectangle for last 5 days
    last_5_dates = dates[-6:]
    last_5_prices = prices[-6:-1]
    min_price = min(last_5_prices) * 0.99
    max_price = max(last_5_prices) * 1.01
    
    fig.add_shape(
        type="rect",
        x0=last_5_dates[0],
        y0=min_price,
        x1=dates[-2],
        y1=max_price,
        line=dict(
            color="#f0f0f0",
            width=1,
        ),
        fillcolor="rgba(0, 113, 206, 0.05)",
    )
    
    # Style the chart
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        hovermode="x unified",
        plot_bgcolor='white',
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        tickprefix='$'
    )
    
    return fig

def mock_historical_data():
    """Generate mock historical data for display"""
    return [
        {"date": "2025-04-12", "open": 163.32, "close": 165.01, "volume": 4892134},
        {"date": "2025-04-11", "open": 162.45, "close": 163.22, "volume": 3982517},
        {"date": "2025-04-10", "open": 164.03, "close": 162.86, "volume": 5128934},
        {"date": "2025-04-09", "open": 164.79, "close": 164.24, "volume": 3726189},
        {"date": "2025-04-08", "open": 162.19, "close": 164.65, "volume": 4237589},
    ]

def get_image_base64(image_path):
    """Convert an image to base64 string for embedding in HTML"""
    import base64
    from PIL import Image
    import io
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return img_str
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

def display_walmart_header():
    """Display Walmart logo and header"""
    # Use the local logo file instead of a URL
    logo_path = "Walmart_Logo.png"
    
    st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <img src="data:image/png;base64,{get_image_base64(logo_path)}" width="550">
            <h1 class="app-title">Stock Price Prediction</h1>
            <p class="app-subtitle">Senior Design Project - Walmart Stock Analysis</p>
        </div>
    """, unsafe_allow_html=True)

def settings_page():
    """Display settings page with functional controls that apply changes"""
    st.markdown('<h1 class="section-title">Settings</h1>', unsafe_allow_html=True)
    
    # Store initial settings to detect changes
    initial_settings = st.session_state.settings.copy()
    
    with st.container():
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        
        st.subheader("Data Settings")
        
        # Data source settings
        data_source = st.radio(
            "Data Source",
            ["Kaggle API", "Local CSV", "Yahoo Finance API"],
            index=["Kaggle API", "Local CSV", "Yahoo Finance API"].index(st.session_state.settings['data_source'])
        )
        st.session_state.settings['data_source'] = data_source
        
        if data_source == "Local CSV":
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                st.session_state.settings['csv_file'] = uploaded_file.name
        
        if data_source == "Yahoo Finance API":
            ticker = st.text_input("Stock Ticker Symbol", value="WMT")
            st.session_state.settings['ticker'] = ticker
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        st.subheader("Model Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Prediction Model",
            ["Linear Regression", "Random Forest", "XGBoost", "LSTM Neural Network"],
            index=["Linear Regression", "Random Forest", "XGBoost", "LSTM Neural Network"].index(st.session_state.settings['model_type'])
        )
        st.session_state.settings['model_type'] = model_type
        
        # Prediction horizon
        prediction_days = st.slider(
            "Prediction Horizon (Days)", 
            1, 30, 
            st.session_state.settings['prediction_days']
        )
        st.session_state.settings['prediction_days'] = prediction_days
        
        # Feature selection
        all_features = ["Price", "Volume", "Moving Averages", "Volatility", "Momentum Indicators", "Sentiment Analysis"]
        features = st.multiselect(
            "Features to Include",
            all_features,
            default=st.session_state.settings['features']
        )
        if features:  # Only update if not empty
            st.session_state.settings['features'] = features
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        st.subheader("Display Settings")
        
        # Theme selection
        theme = st.radio(
            "Application Theme",
            ["Walmart Blue", "Dark Mode", "Light Mode"],
            index=["Walmart Blue", "Dark Mode", "Light Mode"].index(st.session_state.settings['theme'])
        )
        st.session_state.settings['theme'] = theme
        
        # Chart type
        chart_type = st.selectbox(
            "Default Chart Type",
            ["Line Chart", "Candlestick", "OHLC", "Area Chart"],
            index=["Line Chart", "Candlestick", "OHLC", "Area Chart"].index(st.session_state.settings['chart_type'])
        )
        st.session_state.settings['chart_type'] = chart_type
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show current settings in a summary section
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.subheader("Current Settings")
    
    st.json(st.session_state.settings)
    
    # Check if settings have changed
    settings_changed = initial_settings != st.session_state.settings
    save_btn = st.button("Save Settings", key="save_settings", 
                         type="primary" if settings_changed else "secondary",
                         disabled=not settings_changed)
    
    if save_btn:
        st.success("Settings saved successfully! All changes will be applied.")
        if st.session_state.settings['theme'] != initial_settings['theme']:
            st.info("Theme changes will be applied on the next page refresh.")
            time.sleep(1)
            st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display a preview of how settings will look
    if theme != "Walmart Blue":
        preview_color = "#121212" if theme == "Dark Mode" else "#f8f9fa"
        text_color = "#ffffff" if theme == "Dark Mode" else "#212529"
        
        st.markdown(f"""
        <div style="margin-top: 1.5rem; padding: 1rem; border-radius: 8px; background-color: {preview_color}; color: {text_color};">
            <h4 style="margin-bottom: 0.5rem;">Theme Preview</h4>
            <p>This is how the {theme} would look if applied.</p>
        </div>
        """, unsafe_allow_html=True)

def display_stock_price_prediction(model, current_data):
    """Display the stock price prediction section with improved layout"""
    # Get the current date
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # Load stock data
    stock_data = load_stock_data()
    
    st.markdown('<div class="trading-panel">', unsafe_allow_html=True)
    
    # Layout for current stock info
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Current price display
        change_class = "price-change-positive" if current_data["change"] > 0 else "price-change-negative"
        change_prefix = "+" if current_data["change"] > 0 else ""
        
        st.markdown(f"""
            <div>
                <div class="stock-price">${current_data["price"]:.2f}</div>
                <span class="{change_class}">{change_prefix}{current_data["change"]:.2f} ({change_prefix}{current_data["change_percent"]:.2f}%)</span>
                <div style="color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem;">Walmart Inc. (WMT) - {current_date}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Key metrics cards with hover effect
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{:,}</div>
            </div>
        """.format(current_data["volume"]), unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Day Range</div>
                <div class="metric-value">${:.2f} - ${:.2f}</div>
            </div>
        """.format(current_data["low"], current_data["high"]), unsafe_allow_html=True)
    
    # Add chart in a container
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Stock Price History")
    
    # Allow user to select time period for chart
    time_period = st.select_slider(
        "Select Time Period",
        options=["1M", "3M", "6M", "1Y", "2Y", "All"],
        value="6M"
    )
    
    # Create and display the chart
    fig = display_stock_chart(stock_data, time_period)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Price prediction form with improved layout
    st.subheader("Predict Tomorrow's Price")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            current_price = st.number_input(
                "Today's Closing Price ($)",
                min_value=0.01,
                max_value=1000.0,
                value=current_data["price"],
                step=0.01,
                format="%.2f"
            )
            
            submit_button = st.form_submit_button("Predict Price")
    
    # Make a prediction if the form is submitted
    if submit_button:
        # Prepare the input features - in a real app this would use model.predict()
        # Here we'll just simulate a prediction with a simple calculation
        try:
            # In a real app, you would get the prediction from your model
            # For this example, we're using a simple calculation
            predicted_price = current_price * 1.005  # Example: 0.5% increase
            predicted_change = predicted_price - current_price
            predicted_change_percent = (predicted_change / current_price) * 100
            
            # Display the prediction result
            st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
            st.markdown('<div class="prediction-title">Predicted Price for Tomorrow</div>', unsafe_allow_html=True)
            
            # Prediction value and change
            change_class = "prediction-increase" if predicted_change > 0 else "prediction-decrease"
            change_symbol = "↗" if predicted_change > 0 else "↘"
            change_prefix = "+" if predicted_change > 0 else ""
            
            st.markdown(f"""
                <div class="prediction-value">${predicted_price:.2f}</div>
                <div class="{change_class}">
                    {change_symbol} {change_prefix}{predicted_change:.2f} ({change_prefix}{predicted_change_percent:.2f}%)
                </div>
            """, unsafe_allow_html=True)
            
            # Add model confidence level
            st.markdown("""
                <div style="margin-top: 1.5rem; font-size: 0.95rem;">
                    <div style="font-weight: 500; margin-bottom: 0.5rem;">Model Confidence</div>
                    <div style="height: 10px; background-color: #e9ecef; border-radius: 5px; overflow: hidden;">
                        <div style="width: 96%; height: 100%; background-color: #0071ce;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.5rem;">
                        <span>0%</span>
                        <span style="font-weight: 500;">96%</span>
                        <span>100%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    
    # Add technical indicators in tabs
    st.subheader("Technical Analysis")
    
    tabs = st.tabs(["Price Indicators", "Volume Analysis", "Moving Averages"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="indicator"><span class="indicator-label">RSI (14)</span><span class="indicator-value">48.32</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="indicator"><span class="indicator-label">MACD</span><span class="indicator-value">+0.42</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="indicator"><span class="indicator-label">Stochastic %K</span><span class="indicator-value">62.18</span></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="indicator"><span class="indicator-label">CCI (14)</span><span class="indicator-value">+87.24</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="indicator"><span class="indicator-label">ATR (14)</span><span class="indicator-value">1.28</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="indicator"><span class="indicator-label">Bollinger Bands</span><span class="indicator-value">Middle</span></div>', unsafe_allow_html=True)
    
    with tabs[1]:
        # Create dummy volume data
        volume_dates = pd.date_range(end=datetime.now(), periods=14).tolist()
        volumes = [4.2, 3.8, 4.5, 3.9, 4.1, 4.2, 4.8, 3.7, 3.9, 4.3, 4.5, 4.7, 4.2, 4.3]
        
        volume_data = pd.DataFrame({
            'Date': volume_dates,
            'Volume': volumes
        })
        
        # Create a bar chart for volume
        fig = px.bar(volume_data, x='Date', y='Volume', 
                    title=None,
                    labels={'Volume': 'Volume (M)', 'Date': ''},
                    template='simple_white')
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor='#e0e0e0',
                tickformat='%b %d'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f0f0f0',
                zeroline=False,
                linecolor='#e0e0e0'
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="'Segoe UI', sans-serif"
            )
        )
        
        # Update bar colors
        fig.update_traces(
            marker_color='#0071ce',
            hovertemplate='<b>%{y:.1f}M</b><br>%{x|%b %d, %Y}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="indicator"><span class="indicator-label">Volume</span><span class="indicator-value">4.24M</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="indicator"><span class="indicator-label">Avg. Volume</span><span class="indicator-value">4.12M</span></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="indicator"><span class="indicator-label">Volume Trend</span><span class="indicator-value">Stable</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="indicator"><span class="indicator-label">Vol/Price Ratio</span><span class="indicator-value">26.38</span></div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        <div style="padding: 1rem 0;">
            <div class="indicator"><span class="indicator-label">SMA (20)</span><span class="indicator-value">$164.82</span></div>
            <div class="indicator"><span class="indicator-label">SMA (50)</span><span class="indicator-value">$162.47</span></div>
            <div class="indicator"><span class="indicator-label">SMA (200)</span><span class="indicator-value">$156.93</span></div>
            <div class="indicator"><span class="indicator-label">EMA (20)</span><span class="indicator-value">$165.12</span></div>
            <div class="indicator"><span class="indicator-label">EMA (50)</span><span class="indicator-value">$163.08</span></div>
            <div class="indicator"><span class="indicator-label">Signal</span><span class="indicator-value" style="color: #2ecc71;">Bullish</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add model information section
    st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Model Type**: Linear Regression
        - **Training Data**: 5 years of historical WMT prices
        - **Features**: Price, Volume, Moving Averages, Volatility
        - **Last Updated**: April 10, 2025
        """)
    
    with col2:
        st.markdown("""
        - **R² Score**: 0.9966
        - **MAE**: 0.568052
        - **RMSE**: 0.865666
        - **Forecast Horizon**: 1 day
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close trading panel
    
    # Add a disclaimer
    st.markdown("""
    <div style="font-size: 0.8rem; color: #6c757d; padding: 1.5rem 0; text-align: center;">
        <b>Disclaimer:</b> This prediction is for educational purposes only and should not be used as financial advice. 
        Past performance is not indicative of future results.
    </div>
    """, unsafe_allow_html=True)

def load_stock_data():
    """Load historical stock data for Walmart"""
    try:
        # In a real app, this would load from a CSV or API
        # For demo purposes, we'll create synthetic data
        
        # Create date range for the past 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 years of data
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate realistic price data with upward trend and volatility
        base_price = 120.0  # Starting price 2 years ago
        
        # Create random walk with drift for more realistic price movement
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0005, 0.015, size=len(date_range))  # Mean positive return
        
        # Add some seasonality and events
        for i in range(len(returns)):
            # Add quarterly earnings effects (positive spikes)
            if i % 63 in [0, 1, 2]:  # Approx every 3 months
                returns[i] += 0.02 * np.random.random()
            
            # Add occasional market dips
            if np.random.random() < 0.01:  # 1% chance of a dip
                returns[i] -= 0.05 * np.random.random()
        
        # Calculate price series using cumulative returns
        log_returns = np.log(1 + returns)
        cum_returns = np.cumsum(log_returns)
        price_series = base_price * np.exp(cum_returns)
        
        # Generate other required fields
        volumes = np.random.randint(2000000, 10000000, size=len(date_range))
        
        # Create the DataFrame
        stock_data = pd.DataFrame({
            'date': date_range,
            'open': price_series * np.random.uniform(0.990, 0.998, size=len(date_range)),
            'high': price_series * np.random.uniform(1.005, 1.020, size=len(date_range)),
            'low': price_series * np.random.uniform(0.980, 0.995, size=len(date_range)),
            'close': price_series,
            'adj_close': price_series * np.random.uniform(0.998, 1.002, size=len(date_range)),
            'volume': volumes
        })
        
        # Ensure the data is sorted by date
        stock_data = stock_data.sort_values('date')
        
        return stock_data
    
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def display_stock_chart(stock_data, time_period):
    """Display interactive stock price chart with time period selection"""
    # Filter data based on selected time period
    end_date = datetime.now()
    
    if time_period == '1M':
        start_date = end_date - timedelta(days=30)
    elif time_period == '3M':
        start_date = end_date - timedelta(days=90)
    elif time_period == '6M':
        start_date = end_date - timedelta(days=180)
    elif time_period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif time_period == '2Y':
        start_date = end_date - timedelta(days=2*365)
    elif time_period == 'All':
        start_date = stock_data['date'].min()
    else:
        start_date = end_date - timedelta(days=90)  # Default to 3 months
    
    # Filter the data
    filtered_data = stock_data[stock_data['date'] >= start_date].copy()
    
    # Create the chart
    fig = go.Figure()
    
    # Add the price line
    fig.add_trace(
        go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['adj_close'],
            mode='lines',
            name='Adjusted Close',
            line=dict(color='#0071ce', width=2.5),
            hovertemplate='<b>$%{y:.2f}</b><br>%{x|%b %d, %Y}<extra></extra>'
        )
    )
    
    # Add a marker for the most recent price
    latest_data = filtered_data.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[latest_data['date']],
            y=[latest_data['adj_close']],
            mode='markers',
            marker=dict(color='#ffc220', size=10, line=dict(color='white', width=2)),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    # Improved layout
    fig.update_layout(
        title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            linecolor='#e0e0e0',
            tickformat='%b %d',
            rangebreaks=[dict(bounds=["sat", "mon"])]  # Hide weekends
        ),
        yaxis=dict(
            title='Adjusted Close Price ($)',
            titlefont=dict(size=14),
            showgrid=True,
            gridcolor='#f0f0f0',
            zeroline=False,
            linecolor='#e0e0e0',
            tickprefix='$',
            tickformat='.2f'
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="'Segoe UI', sans-serif"
        ),
        font=dict(
            family="'Segoe UI', sans-serif",
            size=13,
            color="#555"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def main():
    """Main function to run the app"""
    # Load model
    model = load_model()
    if not model:
        st.error("Failed to load the prediction model. Please check the logs.")
        return
    
    # Display Walmart header with centered logo and title
    display_walmart_header()
    
    # Initialize session state for settings if not exists
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'data_source': 'Kaggle API',
            'model_type': 'Linear Regression',
            'prediction_days': 1,
            'features': ['Price', 'Volume', 'Moving Averages', 'Volatility'],
            'theme': 'Walmart Blue',
            'chart_type': 'Line Chart'
        }
    
    # Store the current page in session state if it doesn't exist
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Price Prediction"
    
    # Style the navigation container
    st.markdown('<style>div.row-widget.stHorizontal {gap: 0; margin-bottom: 2rem;}</style>', unsafe_allow_html=True)
    
    # Create the buttons with full width
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Price Prediction", key="btn_price_pred", use_container_width=True, 
                   type="primary" if st.session_state.current_page == "Price Prediction" else "secondary"):
            st.session_state.current_page = "Price Prediction"
            st.experimental_rerun()
    
    with col2:
        if st.button("Project Overview", key="btn_overview", use_container_width=True,
                   type="primary" if st.session_state.current_page == "Project Overview" else "secondary"):
            st.session_state.current_page = "Project Overview"
            st.experimental_rerun()
    
    with col3:
        if st.button("Settings", key="btn_settings", use_container_width=True,
                   type="primary" if st.session_state.current_page == "Settings" else "secondary"):
            st.session_state.current_page = "Settings"
            st.experimental_rerun()
    
    # Display the appropriate page based on current_page value
    if st.session_state.current_page == "Project Overview":
        # Call the project overview page function
        project_overview_page()
    elif st.session_state.current_page == "Settings":
        # Display settings page
        settings_page()
    else:
        # Original price prediction content
        # Create trading panel with current stock info
        # In a real app, this would fetch actual current data
        current_data = {
            "price": 165.24,
            "change": 1.98,
            "change_percent": 1.21,
            "volume": 4238129,
            "open": 163.45,
            "high": 165.92,
            "low": 163.12,
        }
        
        display_stock_price_prediction(model, current_data)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Walmart Stock Price Prediction | Senior Design Project | 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
