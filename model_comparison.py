import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from swnn import SynapseWeightedNeuralNetwork

# Load and preprocess the data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', parse_dates=['Month'], index_col='Month')
    return df

def prepare_data(data, n_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:(i + n_steps), 0])
        y.append(scaled_data[i + n_steps, 0])
    return np.array(X), np.array(y), scaler

# SWNN model (your implementation)
def train_swnn(X_train, y_train, X_test, y_test):
    model = SynapseWeightedNeuralNetwork([X_train.shape[1], 64, 32, 1], ['relu', 'relu', 'tanh'])
    errors = model.train(X_train, y_train, epochs=1000, batch_size=32)
    predictions = [model.forward(x)[0] for x in X_test]
    return np.array(predictions)

# ARIMA model
def train_arima(data):
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    return model_fit.forecast(steps=len(data))

# Prophet model
def train_prophet(data):
    df = data.reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=len(data), freq='MS')
    forecast = model.predict(future)
    return forecast['yhat'].values

# LSTM model
def train_lstm(X_train, y_train, X_test):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model.predict(X_test)

# Evaluate models
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return mse, mae

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    n_steps = 3
    X, y, scaler = prepare_data(df.values, n_steps)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train and evaluate models
    models = {
        'SWNN': train_swnn(X_train, y_train, X_test, y_test),
        'ARIMA': train_arima(df['Passengers']),
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
        mse, mae = evaluate_model(df['Passengers'].values[-len(y_test):], predictions[-len(y_test):])
        results[model_name] = {'MSE': mse, 'MAE': mae}

    # Print results
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE = {metrics['MSE']:.2f}, MAE = {metrics['MAE']:.2f}")

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(y_test):], df['Passengers'].values[-len(y_test):], label='Actual')
    for model_name, predictions in models.items():
        plt.plot(df.index[-len(y_test):], predictions[-len(y_test):], label=f'{model_name} Prediction')
    plt.legend()
    plt.title('Air Passengers Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')
    plt.show()