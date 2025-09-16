# Load-Forecast-PV-and-Battery-Optimization


Energy Management System for School Buildings
1. Load Forecasting System
Overview
The Load Forecasting System predicts hourly energy consumption for a school building in Los Angeles using historical weather data and machine learning techniques.

Methodology
Data Acquisition
Weather Data: Fetched from Open-Meteo API using latitude/longitude coordinates

Temporal Features: Hour, day of week, month, weekend flags

School-Specific Features: Holiday schedules, summer break periods, occupancy patterns

Feature Engineering
Temporal Features:

Hour of day (24-hour cycle)

Day of week (7-day cycle)

Month of year (seasonal patterns)

Weekend/holiday indicators

School Schedule Features:

US holiday calendar

Summer break (June-August)

Occupancy hours (7 AM - 5 PM on school days)

Weather Features:

Temperature

Humidity

Wind speed

Rainfall

Solar radiation

Machine Learning Model
Algorithm: Long Short-Term Memory (LSTM) Neural Network

Architecture:

Input layer: 12 features (weather + temporal)

Hidden layers: Two LSTM layers (64 and 32 units) with dropout regularization

Output layer: Single neuron for kW prediction

Training: Time-based split (80% training, 20% validation)

Optimization: Adam optimizer with MSE loss function

Why LSTM?
LSTMs are particularly suited for this application because:

They capture temporal dependencies in time series data

They can learn complex patterns across multiple time scales (hourly, daily, weekly, seasonal)

They handle the sequential nature of energy consumption data effectively

Key Metrics for Evaluation
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Mean Absolute Percentage Error (MAPE)

R² Score

Usage
Set location parameters (latitude, longitude)

Define date range for historical data

Run the script to train the model and generate forecasts

Results are saved to predicted_hourly_load.csv

Dependencies
pandas, numpy, matplotlib

tensorflow/keras

scikit-learn

requests, holidays

2. Battery Optimization System
Overview
The Battery Optimization System creates an optimal charge/discharge schedule for a battery storage system integrated with solar PV, aiming to minimize grid electricity usage and carbon emissions.

Methodology
PV Production Forecasting
Model: Physical model using PVlib library

Inputs:

Solar radiation data

Temperature, wind speed

PV system parameters (capacity, tilt, azimuth, efficiency)

Output: Hourly AC power production forecast

Battery Optimization Algorithm
Objective: Minimize grid electricity usage while considering:

PV production patterns

Load consumption patterns

Battery constraints (capacity, charge/discharge rates, efficiency)

Grid carbon intensity variations

Decision Logic:

Excess PV Production:

Charge battery (within operational limits)

Priority: Maximize self-consumption of solar energy

Energy Deficit:

Discharge battery to cover load (within operational limits)

Consider grid carbon intensity when deciding to use battery vs grid

Preserve battery life by avoiding unnecessary cycling

Constraints:

State of Charge (SOC) limits: 20%-95%

Maximum charge/discharge rates

Battery efficiency (round-trip)

Time-based carbon intensity considerations

Carbon Awareness
Daytime (8 AM - 6 PM): Lower carbon intensity (0.3 kgCO₂/kWh)

Nighttime: Higher carbon intensity (0.6 kgCO₂/kWh)

Optimization: Prioritize battery discharge during high carbon intensity periods

Key Performance Indicators
Self-Consumption Ratio: Percentage of PV energy used directly

Self-Sufficiency Ratio: Percentage of load covered by PV

Grid Independence: Reduction in grid dependence

Carbon Emissions: Total kgCO₂ emissions from grid electricity

Battery Throughput: Total energy cycled through battery

Usage
Define PV system parameters (capacity, orientation, efficiency)

Define battery system parameters (capacity, rates, efficiency)

Provide load forecast (from previous system)

Run optimization to generate battery schedule

Results are saved to battery_optimization_schedule.csv

Dependencies
pandas, numpy, matplotlib

pvlib

requests

System Integration
Data Flow
Weather API → Load Forecast → Predicted Load

Weather API → PV Production Forecast → PV Generation

Predicted Load + PV Generation → Battery Optimization → Charge/Discharge Schedule

Practical Considerations
Data Quality: Real building data will improve accuracy significantly

Parameter Tuning: System parameters should be adjusted based on actual equipment specifications

Temporal Patterns: School calendar events should be accurately reflected for best results

Weather Data: Higher resolution weather data improves forecasting accuracy

Future Enhancements
Real-time Optimization: Incorporate real-time pricing and grid conditions

Predictive Control: Use model predictive control for better optimization

Multiple Energy Sources: Integrate with other renewable sources

Demand Response: Participate in utility demand response programs

Fault Detection: Add anomaly detection for equipment monitoring

This integrated system provides a comprehensive solution for school building energy management, combining load forecasting with optimal battery dispatch to minimize costs and carbon emissions while maintaining energy reliability.
