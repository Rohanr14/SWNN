import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from swnn import SynapseWeightedNeuralNetwork

# Data preprocessing function
def preprocess_data(df):
    # Ensure the dataframe has only one column (excluding index)
    if len(df.columns) > 1:
        st.warning("Multiple columns detected. Using only the first numeric column for analysis.")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            df = df[[numeric_columns[0]]]
        else:
            st.error("No numeric columns found in the dataset.")
            return None

    # Convert to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with the mean of the column
    df = df.fillna(df.mean())

    # Check if we have enough data left
    if len(df) < 50:  # Arbitrary threshold, adjust as needed
        st.error("Not enough valid numeric data points after preprocessing.")
        return None

    return df

# SWNN model (your implementation)
@st.cache_data
def train_swnn(X_train, y_train, X_test):
    model = SynapseWeightedNeuralNetwork([X_train.shape[1], 64, 32, 1], ['relu', 'relu', 'tanh'])
    errors = model.train(X_train, y_train, epochs=1000, batch_size=32)
    predictions = [model.forward(x)[0] for x in X_test]
    return np.array(predictions)

# ARIMA model
@st.cache_data
def train_arima(data):
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    return model_fit.forecast(steps=len(data))

# Prophet model
@st.cache_data
def train_prophet(data):
    df = data.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=len(data), freq='MS')
    forecast = model.predict(future)
    return forecast['yhat'].values

# LSTM model
@st.cache_data
def train_lstm(X_train, y_train, X_test):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model.predict(X_test)

# Prepare data for models
def prepare_data(data, n_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:(i + n_steps), 0])
        y.append(scaled_data[i + n_steps, 0])
    return np.array(X), np.array(y), scaler

# Evaluate models
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return mse, mae

# Main Streamlit app
def main():
    st.title('Time Series Model Comparison')
    st.write('Upload a CSV file with your time series data to compare different models.')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    is_data_clean = st.checkbox("Data is already cleaned (skip preprocessing)")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
        
        st.write('Preview of uploaded data:')
        st.write(df.head())

        if not is_data_clean:
            # Preprocess the data
            df = preprocess_data(df)
            if df is None:
                return

            st.write('Preview of preprocessed data:')
            st.write(df.head())
        else:
            st.write("Skipping preprocessing as data is marked as clean.")

        # Prepare data
        n_steps = 3
        X, y, scaler = prepare_data(df.values, n_steps)
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train models
        with st.spinner('Training models...'):
            models = {
                'SWNN': train_swnn(X_train, y_train, X_test),
                'ARIMA': train_arima(df.iloc[:, 0]),
                'Prophet': train_prophet(df),
                'LSTM': train_lstm(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),
                                   y_train, X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
            }

        # Inverse transform predictions
        for model_name, predictions in models.items():
            if model_name in ['SWNN', 'LSTM']:
                models[model_name] = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Calculate metrics
        results = {}
        for model_name, predictions in models.items():
            mse, mae = evaluate_model(df.iloc[-len(y_test):, 0].values, predictions[-len(y_test):])
            results[model_name] = {'MSE': mse, 'MAE': mae}

        # Display results
        st.subheader('Model Performance Comparison')
        results_df = pd.DataFrame(results).T
        st.table(results_df.style.highlight_min(axis=0))

        # Plot predictions
        st.subheader('Predictions Visualization')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-len(y_test):], df.iloc[-len(y_test):, 0].values, label='Actual', linewidth=2)
        for model_name, predictions in models.items():
            ax.plot(df.index[-len(y_test):], predictions[-len(y_test):], label=f'{model_name} Prediction')
        ax.legend()
        ax.set_title('Time Series Prediction Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        st.pyplot(fig)

if __name__ == '__main__':
    main()