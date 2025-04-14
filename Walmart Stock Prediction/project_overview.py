import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

def project_overview_page():
    """Render the Project Overview page"""
    # Page title
    st.markdown('<h1 class="main-title">Project Overview</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="overview-intro">
        This page provides a comprehensive overview of the end-to-end data science project for predicting Walmart's 
        next-day adjusted close stock price using regression techniques. Each section details the methodology, 
        decisions, and implementation steps taken throughout the project lifecycle.
    </div>
    """, unsafe_allow_html=True)
    
    # Display PowerBI Dashboard image if it exists
    image_path = "Walmart_PowerBI_Dashboard.png"
    if os.path.exists(image_path):
        st.markdown('<div class="section-title">📊 Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        image = Image.open(image_path)
        st.image(image, caption="Walmart Stock Analysis PowerBI Dashboard", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 1: Data Import & EDA
    st.markdown('<div class="section-title">📊 Step 1: Data Import & EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * Used historical Walmart stock data from Kaggle
    * Explored trends using a Power BI dashboard (shown above)
    * This step helped identify early volatility patterns, pricing bands across years, and guided feature selection
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Data Preprocessing
    st.markdown('<div class="section-title">🧼 Step 2: Data Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * Converted date column into datetime and extracted year, month, and day
    * Created Price_Change, pct_change, and volatility to reflect daily movement and market behavior
    * Ensured null values were checked and addressed
    
    ```python
    # Sample code from the project
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
    df['Price_Change'] = df['close'] - df['open']
    df['pct_change'] = ((df['close'] - df['open']) / df['open']) * 100
    df['volatility'] = (df['high'] - df['low']) / df['open']
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Feature Engineering
    st.markdown('<div class="section-title">🏗️ Step 3: Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    Added multiple derived features:
    
    * **5_day_ma, ma_3d, ma_7d** – moving averages of adjusted close
    * **prev_adj_close** – yesterday's adjusted close for comparison
    * **return, return_1d, return_3d** – percent changes over various spans
    * **price_range, high_low_ratio, volatility_3d** – price movement metrics
    
    These features helped capture different aspects of stock price movement patterns including trends,
    momentum, and volatility over different time periods.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Standardization & Normalization
    st.markdown('<div class="section-title">⚖️ Step 4: Standardization & Normalization</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * Applied log transformation to skewed columns (open, close, adj_close, etc.)
    * Used StandardScaler to normalize log-transformed data
    * Did this for price columns, volume, and volatility to improve model performance
    
    ```python
    # Log transform example
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']
    for col in price_cols:
        df[f'log_{col}'] = np.log(df[col])
        
    # Standardization example
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[log_cols]), 
        columns=[f'z_{col}' for col in log_cols]
    )
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 5: Outlier Detection
    st.markdown('<div class="section-title">🚩 Step 5: Outlier Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * Created outlier flags for major columns using IQR method
    * Initially filtered outliers, but decided to not drop them as this would remove ~20% of the dataset, leading to potential bias or underfitting
    
    This approach allowed the model to learn from both normal market conditions and unusual or extreme market events,
    making it more robust for real-world predictions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 6: ETL Pipeline
    st.markdown('<div class="section-title">🔁 Step 6: ETL Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * Automated all of the above using a class called `WalmartStockPipeline`
    * Each transformation (log, standardization, feature creation, etc.) became a method within this class
    * Allows the project to scale easily and be reused for future data or deployment
    
    ```python
    # Pipeline class structure
    class WalmartStockPipeline:
        def __init__(self, df):
            self.df = df.copy()
            self.scaler_price = StandardScaler()
            self.scaler_volume = StandardScaler()
            self.scaler_volatility = StandardScaler()
            
        def convert_date(self):
            # Convert date column
            
        def create_classification_and_features(self):
            # Create derived features
            
        def log_transform_and_standardize_prices(self):
            # Transform price data
            
        # Other methods...
            
        def run(self):
            # Execute all pipeline steps in sequence
            # Return processed dataframe
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 7: Feature Selection
    st.markdown('<div class="section-title">🧠 Step 7: Feature Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Used three methods:
        
        * **Mutual Information Regression**: Measures non-linear relationships between features and target
        * **Recursive Feature Elimination (RFE)**: Iteratively removes the least important features
        * **Permutation Importance**: Evaluates feature importance by randomly shuffling feature values
        
        Combined scores from all three methods and selected the top features with highest predictive power:
        
        * Examples: z_log_adj_close, prev_adj_close, ma_3d, high_low_ratio
        """)
    
    with col2:
        # Create a simple chart to illustrate feature importance
        feature_importance = {
            'z_log_adj_close': 0.92,
            'price_range': 0.78,
            'volatility': 0.65,
            'ma_3d': 0.88,
            'ma_7d': 0.82,
            'high_low_ratio': 0.73
        }
        
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            labels={'x': 'Importance Score', 'y': 'Feature'},
            title='Key Features by Importance',
            color=list(feature_importance.values()),
            color_continuous_scale='Blues',
        )
        
        fig.update_layout(
            height=300,
            xaxis_title='Normalized Importance',
            yaxis_title=None,
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=40, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 8: Model Selection & Hyperparameter Tuning
    st.markdown('<div class="section-title">🤖 Step 8: Model Selection & Hyperparameter Tuning</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    Built a pipeline to automatically:
    
    * Train multiple regression models:
        * Linear Regression
        * Ridge Regression (with alpha values: 0.01, 0.1, 1.0, 10)
        * Lasso Regression (with alpha values: 0.001, 0.01, 0.1, 1.0)
        * Random Forest (with various n_estimators and max_depth values)
        * Gradient Boosting (with various learning rates and depths)
        * XGBoost
        
    * Tune hyperparameters using GridSearchCV with 5-fold cross-validation
    * Evaluate based on R², MAE, and RMSE on holdout test set
    
    ```python
    # Example model comparison code
    models = {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'alpha': [0.01, 0.1, 1.0, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
        'RandomForest': (RandomForestRegressor(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10]
        }),
        # Other models...
    }
    
    # Run GridSearch for each model
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, scoring='r2', cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        # Evaluate and compare performance...
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 9: Results
    st.markdown('<div class="section-title">🏆 Step 9: Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        * **Best performing model**: Linear Regression
        
        * **Metrics**:
            * R² Score: 0.9966
            * Mean Absolute Error (MAE): 0.568052
            * Root Mean Squared Error (RMSE): 0.865666
            
        * Saved the trained model with joblib for later use with:
            ```python
            joblib.dump(best_model, 'best_stock_model.pkl')
            ```
        """)
    
    with col2:
        # Create a gauge chart to show R² score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 0.9966,
            title = {'text': "R² Score"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "#0071ce"},
                'steps': [
                    {'range': [0, 0.5], 'color': "#f2f2f2"},
                    {'range': [0.5, 0.75], 'color': "#cce5ff"},
                    {'range': [0.75, 0.9], 'color': "#99caff"},
                    {'range': [0.9, 1], 'color': "#66b0ff"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9966
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 10: Deep Learning Experiment
    st.markdown('<div class="section-title">🧠 Step 10: Deep Learning Experiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * Created a PyTorch neural network to classify whether price would increase or decrease
    * Tried multiple optimizers (Adam, SGD, RMSprop, AdamW)
    * Despite tuning, accuracy plateaued at ~60%
    * Chose to focus on regression since performance was significantly better
    
    ```python
    # Neural network structure (simplified)
    class StockNN(nn.Module):
        def __init__(self, input_dim):
            super(StockNN, self).__init__()
            self.layer1 = nn.Linear(input_dim, 64)
            self.layer2 = nn.Linear(64, 32)
            self.layer3 = nn.Linear(32, 16)
            self.output = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.relu(self.layer3(x))
            x = self.sigmoid(self.output(x))
            return x
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary
    st.markdown('<div class="section-title">🧩 Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-card">', unsafe_allow_html=True)
    st.markdown("""
    * This project reflects the end-to-end structure of a real-world data science workflow
    * It combines EDA, preprocessing, feature engineering, model automation, and evaluation
    * Final output is a model capable of near-perfect prediction on unseen Walmart stock data
    * The entire pipeline demonstrates a systematic approach to stock price prediction, from data collection to model deployment
    
    The skills demonstrated in this project include:
    * Data acquisition and cleaning
    * Feature engineering and selection
    * Time series analysis
    * Model building and evaluation
    * Pipeline automation
    * Visualization and interpretation of results
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Walmart Stock Prediction Tool | Senior Design Project</p>
        <p style="font-size: 0.75rem;">Built with Streamlit | Data for educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)
