import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Configuration
LATITUDE = 34.0522
LONGITUDE = -118.2437
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"
FORECAST_DAYS = 7

# Fetch historical weather data from Open-Meteo API with error handling
def fetch_weather_data(start_date, end_date):
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain,shortwave_radiation",
            "timezone": "America/Los_Angeles"
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if API returned an error
        if "error" in data:
            print(f"API Error: {data['error']}")
            return create_dummy_weather_data(start_date, end_date)
        
        # Check if hourly data exists in response
        if "hourly" not in data:
            print("No hourly data in API response. Using dummy data.")
            return create_dummy_weather_data(start_date, end_date)
            
        hourly_data = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature": hourly_data["temperature_2m"],
            "humidity": hourly_data["relative_humidity_2m"],
            "wind_speed": hourly_data["wind_speed_10m"],
            "rain": hourly_data["rain"],
            "solar_radiation": hourly_data.get("shortwave_radiation", [0] * len(hourly_data["time"]))
        })
        return df
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return create_dummy_weather_data(start_date, end_date)

# Create dummy weather data as fallback
def create_dummy_weather_data(start_date, end_date):
    print("Creating dummy weather data...")
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame({
        "timestamp": date_range,
        "temperature": np.sin(2 * np.pi * np.arange(len(date_range)) / 24) * 10 + 20,  # Daily cycle
        "humidity": np.random.uniform(40, 80, len(date_range)),
        "wind_speed": np.random.uniform(0, 10, len(date_range)),
        "rain": np.random.uniform(0, 5, len(date_range)),
        "solar_radiation": np.maximum(0, np.sin(2 * np.pi * (np.arange(len(date_range)) % 24) / 24)) * 800
    })
    return df

# Generate dummy load data (replace with real data when available)
def generate_dummy_load_data(weather_df):
    # Simulate school load profile: higher during occupied hours, lower otherwise
    base_load = np.random.normal(50, 10, len(weather_df))  # Base load in kW
    temperature_effect = weather_df["temperature"] * 0.5
    solar_effect = weather_df["solar_radiation"] * -0.02
    # Create occupancy pattern (school hours on weekdays)
    hour = weather_df["timestamp"].dt.hour
    weekday = weather_df["timestamp"].dt.weekday
    occupancy_effect = np.where(
        (hour >= 7) & (hour <= 17) & (weekday < 5),  # School hours on weekdays
        30, 0
    )
    noise = np.random.normal(0, 5, len(weather_df))
    load = base_load + temperature_effect + solar_effect + occupancy_effect + noise
    load = np.clip(load, 10, 100)  # Ensure realistic range
    return load

# Add school schedule features
def add_school_features(df):
    us_holidays = holidays.US(years=df["timestamp"].dt.year.unique())
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["timestamp"].dt.date.isin(us_holidays).astype(int)
    df["is_summer_break"] = df["month"].isin([6, 7, 8]).astype(int)  # June-August
    df["is_occupied"] = (
        (df["hour"] >= 7) & (df["hour"] <= 17) &  # School hours
        (df["is_weekend"] == 0) &
        (df["is_holiday"] == 0) &
        (df["is_summer_break"] == 0)
    ).astype(int)
    return df

# Fetch future weather forecast
def fetch_forecast_data():
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain,shortwave_radiation",
            "timezone": "America/Los_Angeles",
            "forecast_days": FORECAST_DAYS
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if "error" in data or "hourly" not in data:
            print("Error fetching forecast data. Creating dummy forecast.")
            return create_dummy_forecast_data()
            
        hourly_data = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature": hourly_data["temperature_2m"],
            "humidity": hourly_data["relative_humidity_2m"],
            "wind_speed": hourly_data["wind_speed_10m"],
            "rain": hourly_data["rain"],
            "solar_radiation": hourly_data.get("shortwave_radiation", [0] * len(hourly_data["time"]))
        })
        return df
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        return create_dummy_forecast_data()

# Create dummy forecast data as fallback
def create_dummy_forecast_data():
    print("Creating dummy forecast data...")
    start_date = datetime.now() + timedelta(days=1)
    end_date = start_date + timedelta(days=FORECAST_DAYS)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
    
    df = pd.DataFrame({
        "timestamp": date_range,
        "temperature": np.sin(2 * np.pi * np.arange(len(date_range)) / 24) * 8 + 22,  # Daily cycle
        "humidity": np.random.uniform(45, 85, len(date_range)),
        "wind_speed": np.random.uniform(1, 8, len(date_range)),
        "rain": np.random.uniform(0, 3, len(date_range)),
        "solar_radiation": np.maximum(0, np.sin(2 * np.pi * (np.arange(len(date_range)) % 24) / 24)) * 750
    })
    return df

# Calculate and print model accuracy metrics
def evaluate_model(model, X_val_reshaped, y_val):
    # Make predictions
    y_pred = model.predict(X_val_reshaped).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Print metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.2f} kW")
    print(f"Mean Squared Error (MSE): {mse:.2f} kW²")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kW")
    print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    # Additional interpretation
    if mape < 10:
        print("Excellent accuracy (MAPE < 10%)")
    elif mape < 20:
        print("Good accuracy (MAPE < 20%)")
    elif mape < 30:
        print("Reasonable accuracy (MAPE < 30%)")
    else:
        print("Poor accuracy (MAPE > 30%)")
    
    return y_pred

# Main execution
if __name__ == "__main__":
    # Fetch historical weather data
    print("Fetching historical weather data...")
    weather_df = fetch_weather_data(START_DATE, END_DATE)

    # Generate dummy load data and add to DataFrame
    weather_df["load_kw"] = generate_dummy_load_data(weather_df)

    # Add school-specific features
    df = add_school_features(weather_df)

    # Feature selection
    features = ["temperature", "humidity", "wind_speed", "rain", "solar_radiation",
                "hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_summer_break", "is_occupied"]
    target = "load_kw"

    # Prepare data for training
    X = df[features]
    y = df[target]

    # Split data into train and validation (time-based split)
    split_index = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Reshape data for LSTM [samples, timesteps, features]
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train model
    print("Training model...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=20,  # Reduced for faster execution
        batch_size=32,
        validation_data=(X_val_reshaped, y_val),
        verbose=1
    )

    # Evaluate model with comprehensive metrics
    y_val_pred = evaluate_model(model, X_val_reshaped, y_val)

    # Fetch future weather forecast
    print("Fetching future weather forecast...")
    future_weather = fetch_forecast_data()

    # Add school features to forecast data
    future_weather = add_school_features(future_weather)
    X_future = future_weather[features]

    # Scale future features
    X_future_scaled = scaler.transform(X_future)
    X_future_reshaped = X_future_scaled.reshape((X_future_scaled.shape[0], 1, X_future_scaled.shape[1]))

    # Predict future load
    future_predictions = model.predict(X_future_reshaped).flatten()
    future_weather["predicted_load_kw"] = future_predictions

    # Export predictions
    output = future_weather[["timestamp", "predicted_load_kw"]]
    output.to_csv("predicted_hourly_load.csv", index=False)
    print("Predictions exported to predicted_hourly_load.csv")

    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training history
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Validation set predictions vs actual
    plt.subplot(2, 1, 2)
    # For clarity, show only a subset of validation data (first 200 points)
    sample_size = min(200, len(y_val))
    plt.plot(y_val.values[:sample_size], label='Actual Load', alpha=0.7)
    plt.plot(y_val_pred[:sample_size], label='Predicted Load', alpha=0.7)
    plt.title('Validation Set: Actual vs Predicted Load')
    plt.ylabel('Load (kW)')
    plt.xlabel('Time (hours)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.show()

    # Plot 3: Future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_weather["timestamp"], future_weather["predicted_load_kw"], label="Predicted Load")
    plt.xlabel("Time")
    plt.ylabel("Load (kW)")
    plt.title("7-Day Hourly Load Forecast for School Building")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('load_forecast.png')
    plt.show()