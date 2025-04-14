# Walmart Stock Price Prediction App

A Streamlit web application that predicts Walmart stock prices using a machine learning model trained on historical data.

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate the prediction model (if not already done):
   ```
   python 4050_final_project.py
   ```
   This will create the `best_stock_model.pkl` file needed by the app.

3. Run the Streamlit app:
   ```
   streamlit run stock_prediction_app.py
   ```

## Using the App

1. Enter the current day's stock information in the form
2. Provide the previous 7 days' adjusted closing prices
3. Click "Predict" to get tomorrow's predicted stock price
4. View the prediction results and visualization

## Features

- Interactive web interface with Streamlit
- Visual representation of historical prices and prediction
- Uses machine learning to predict next-day adjusted closing price
- Calculates derived metrics like moving averages automatically

## Note

This app is for educational purposes only and does not constitute financial advice. Stock market investments involve risk, and predictions are not guaranteed.
