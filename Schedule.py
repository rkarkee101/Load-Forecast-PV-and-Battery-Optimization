import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pvlib
from pvlib import location, irradiance, temperature
import requests
import warnings
warnings.filterwarnings('ignore')

# Configuration
LATITUDE = 34.0522
LONGITUDE = -118.2437
TIMEZONE = 'America/Los_Angeles'

# PV System Parameters (dummy values - to be replaced with actual values)
PV_CAPACITY = 400  # kW
INVERTER_EFFICIENCY = 0.96
AREA_TILT = 20  # degrees
AREA_AZIMUTH = 180  # degrees (south-facing)

# Battery Parameters (dummy values)
BATTERY_CAPACITY = 200  # kWh
MAX_CHARGE_RATE = 50  # kW
MAX_DISCHARGE_RATE = 50  # kW
BATTERY_EFFICIENCY = 0.92
MIN_SOC = 0.2  # Minimum state of charge (20%)
MAX_SOC = 0.95  # Maximum state of charge (95%)
INITIAL_SOC = 0.5  # Initial state of charge (50%)

# Grid Carbon Intensity (kgCO2/kWh) - dummy values
DAYTIME_CARBON_INTENSITY = 0.3  # Lower during day (more renewables)
NIGHTTIME_CARBON_INTENSITY = 0.6  # Higher at night (more fossil fuels)

# Weather data fetching functions
def fetch_forecast_data():
    """Fetch weather forecast data from Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain,shortwave_radiation",
            "timezone": TIMEZONE,
            "forecast_days": 7
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

def create_dummy_forecast_data():
    """Create dummy forecast data as fallback"""
    print("Creating dummy forecast data...")
    start_date = datetime.now() + timedelta(days=1)
    end_date = start_date + timedelta(days=7)
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

def forecast_pv_production(weather_data):
    """Forecast PV production using a simplified model"""
    # Prepare weather data
    weather_data.index = pd.to_datetime(weather_data['timestamp'])
    
    # Get solar position
    site_location = location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solar_position = site_location.get_solarposition(weather_data.index)
    
    # Calculate plane of array irradiance
    poa_irradiance = irradiance.get_total_irradiance(
        surface_tilt=AREA_TILT,
        surface_azimuth=AREA_AZIMUTH,
        dni=weather_data['solar_radiation'] * 0.7,  # Approximate DNI from GHI
        ghi=weather_data['solar_radiation'],
        dhi=weather_data['solar_radiation'] * 0.3,  # Approximate DHI from GHI
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    
    # Calculate cell temperature (simple model)
    cell_temperature = temperature.sapm_cell(
        poa_global=poa_irradiance['poa_global'],
        temp_air=weather_data['temperature'],
        wind_speed=weather_data['wind_speed'],
        a=-3.56,  # SAPM parameters for open rack glass-glass
        b=-0.075,
        deltaT=3
    )
    
    # Simple PV model: power = irradiance * capacity * efficiency * temperature correction
    temperature_coeff = -0.004  # -0.4% per degree C above 25°C
    temperature_factor = 1 + temperature_coeff * (cell_temperature - 25)
    
    # Calculate DC power - use a more realistic efficiency factor
    system_efficiency = 0.85  # Account for losses (soiling, wiring, etc.)
    dc_power = poa_irradiance['poa_global'] * PV_CAPACITY * 0.001 * temperature_factor * system_efficiency
    
    # Apply inverter efficiency to get AC power
    ac_power = dc_power * INVERTER_EFFICIENCY
    
    # Ensure non-negative values
    ac_power = ac_power.clip(lower=0)
    
    return ac_power
    """Forecast PV production using a simplified model"""
    # Prepare weather data
    weather_data.index = pd.to_datetime(weather_data['timestamp'])
    
    # Get solar position
    site_location = location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solar_position = site_location.get_solarposition(weather_data.index)
    
    # Calculate plane of array irradiance
    poa_irradiance = irradiance.get_total_irradiance(
        surface_tilt=AREA_TILT,
        surface_azimuth=AREA_AZIMUTH,
        dni=weather_data['solar_radiation'] * 0.7,  # Approximate DNI from GHI
        ghi=weather_data['solar_radiation'],
        dhi=weather_data['solar_radiation'] * 0.3,  # Approximate DHI from GHI
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    
    # Calculate cell temperature (simple model)
    cell_temperature = temperature.sapm_cell(
        poa_global=poa_irradiance['poa_global'],
        temp_air=weather_data['temperature'],
        wind_speed=weather_data['wind_speed'],
        a=-3.56,  # SAPM parameters for open rack glass-glass
        b=-0.075,
        deltaT=3
    )
    
    # Simple PV model: power = irradiance * capacity * efficiency * temperature correction
    temperature_coeff = -0.004  # -0.4% per degree C above 25°C
    temperature_factor = 1 + temperature_coeff * (cell_temperature - 25)
    
    # Calculate DC power
    dc_power = poa_irradiance['poa_global'] * PV_CAPACITY * 0.001 * temperature_factor  # kW
    
    # Apply inverter efficiency to get AC power
    ac_power = dc_power * INVERTER_EFFICIENCY
    
    # Ensure non-negative values
    ac_power = ac_power.clip(lower=0)
    
    return ac_power

def calculate_carbon_intensity(hour):
    """Calculate carbon intensity based on time of day"""
    # Simple model: lower carbon intensity during daytime hours
    if 8 <= hour <= 18:  # Daytime hours
        return DAYTIME_CARBON_INTENSITY
    else:
        return NIGHTTIME_CARBON_INTENSITY

def battery_optimization(pv_production, load_forecast, initial_soc=INITIAL_SOC):
    """
    Optimize battery charge/discharge schedule to minimize grid usage and carbon emissions
    """
    n_periods = len(pv_production)
    time_index = pv_production.index
    
    # Initialize arrays
    grid_import = np.zeros(n_periods)
    battery_charge = np.zeros(n_periods)
    battery_discharge = np.zeros(n_periods)
    soc = np.zeros(n_periods)
    soc[0] = initial_soc * BATTERY_CAPACITY  # Convert to kWh
    
    # Carbon intensity for each hour
    carbon_intensity = np.array([calculate_carbon_intensity(t.hour) for t in time_index])
    
    # Optimization for each time step
    for t in range(n_periods):
        # Net energy available (PV production minus load)
        net_energy = pv_production.iloc[t] - load_forecast.iloc[t]
        
        # Determine optimal battery action based on net energy and carbon intensity
        if net_energy > 0:
            # Excess PV energy - charge battery if possible
            max_charge = min(net_energy, MAX_CHARGE_RATE, 
                            (MAX_SOC * BATTERY_CAPACITY - soc[t]) / BATTERY_EFFICIENCY)
            battery_charge[t] = max_charge
            grid_import[t] = 0
            
            # Update SOC
            if t < n_periods - 1:
                soc[t+1] = soc[t] + battery_charge[t] * BATTERY_EFFICIENCY
        else:
            # Energy deficit - consider discharging battery
            energy_needed = -net_energy
            
            # Check if we should discharge based on carbon intensity
            # Always discharge during high carbon intensity, but also during low carbon intensity
            # if we have sufficient battery charge
            if carbon_intensity[t] > DAYTIME_CARBON_INTENSITY or soc[t] > MIN_SOC * BATTERY_CAPACITY + 10:
                # Discharge battery to avoid grid or use stored energy
                max_discharge = min(energy_needed, MAX_DISCHARGE_RATE, 
                                   (soc[t] - MIN_SOC * BATTERY_CAPACITY) * BATTERY_EFFICIENCY)
                battery_discharge[t] = max_discharge
                grid_import[t] = max(0, energy_needed - max_discharge)
            else:
                # Low carbon intensity and low battery - use grid to preserve battery
                battery_discharge[t] = 0
                grid_import[t] = energy_needed
            
            # Update SOC
            if t < n_periods - 1:
                soc[t+1] = soc[t] - battery_discharge[t] / BATTERY_EFFICIENCY
    
    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': time_index,
        'pv_production': pv_production.values,
        'load_forecast': load_forecast.values,
        'grid_import': grid_import,
        'battery_charge': battery_charge,
        'battery_discharge': battery_discharge,
        'soc': soc[:n_periods] / BATTERY_CAPACITY,  # Convert to fraction
        'carbon_intensity': carbon_intensity
    })
    
    return results

def calculate_metrics(optimization_results):
    """Calculate performance metrics"""
    total_load = optimization_results['load_forecast'].sum()
    total_pv = optimization_results['pv_production'].sum()
    total_grid = optimization_results['grid_import'].sum()
    total_battery_charge = optimization_results['battery_charge'].sum()
    total_battery_discharge = optimization_results['battery_discharge'].sum()
    
    # Self-consumption and self-sufficiency
    self_consumption = min(total_pv, total_load) / total_pv if total_pv > 0 else 0
    self_sufficiency = (total_pv - max(0, total_pv - total_load)) / total_load
    
    # Carbon emissions
    carbon_emissions = (optimization_results['grid_import'] * 
                       optimization_results['carbon_intensity']).sum()
    
    # Grid independence
    grid_independence = 1 - (total_grid / total_load)
    
    metrics = {
        'total_load_kwh': total_load,
        'total_pv_kwh': total_pv,
        'total_grid_kwh': total_grid,
        'self_consumption_ratio': self_consumption,
        'self_sufficiency_ratio': self_sufficiency,
        'carbon_emissions_kg': carbon_emissions,
        'grid_independence_ratio': grid_independence,
        'battery_throughput_kwh': total_battery_charge + total_battery_discharge
    }
    
    return metrics

def plot_optimization_results(optimization_results, metrics):
    """Plot optimization results"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Energy flows
    time = optimization_results['timestamp']
    axes[0].plot(time, optimization_results['load_forecast'], label='Load', linewidth=2)
    axes[0].plot(time, optimization_results['pv_production'], label='PV Production', linewidth=2)
    axes[0].plot(time, optimization_results['grid_import'], label='Grid Import', linewidth=2)
    axes[0].fill_between(time, 0, optimization_results['battery_charge'], 
                        alpha=0.3, label='Battery Charge')
    axes[0].fill_between(time, 0, -optimization_results['battery_discharge'], 
                        alpha=0.3, label='Battery Discharge')
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title('Energy Flow Optimization')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: State of Charge
    axes[1].plot(time, optimization_results['soc'] * 100, label='Battery SOC', color='purple', linewidth=2)
    axes[1].axhline(y=MAX_SOC * 100, color='r', linestyle='--', label='Max SOC')
    axes[1].axhline(y=MIN_SOC * 100, color='r', linestyle='--', label='Min SOC')
    axes[1].set_ylabel('State of Charge (%)')
    axes[1].set_title('Battery State of Charge')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Carbon Intensity
    axes[2].plot(time, optimization_results['carbon_intensity'], label='Carbon Intensity', color='brown', linewidth=2)
    axes[2].set_ylabel('Carbon Intensity (kgCO2/kWh)')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Grid Carbon Intensity')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('battery_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    print("\n" + "="*50)
    print("OPTIMIZATION METRICS")
    print("="*50)
    print(f"Total Load: {metrics['total_load_kwh']:.2f} kWh")
    print(f"Total PV Production: {metrics['total_pv_kwh']:.2f} kWh")
    print(f"Total Grid Import: {metrics['total_grid_kwh']:.2f} kWh")
    print(f"Self-Consumption Ratio: {metrics['self_consumption_ratio']:.2%}")
    print(f"Self-Sufficiency Ratio: {metrics['self_sufficiency_ratio']:.2%}")
    print(f"Grid Independence Ratio: {metrics['grid_independence_ratio']:.2%}")
    print(f"Carbon Emissions: {metrics['carbon_emissions_kg']:.2f} kgCO2")
    print(f"Battery Throughput: {metrics['battery_throughput_kwh']:.2f} kWh")
    print("="*50)

def main():
    """Main function to run the PV forecast and battery optimization"""
    print("Starting PV forecast and battery optimization...")
    
    # Load the load forecast from previous step
    try:
        load_forecast = pd.read_csv('predicted_hourly_load.csv')
        load_forecast['timestamp'] = pd.to_datetime(load_forecast['timestamp'])
        load_forecast.set_index('timestamp', inplace=True)
        print("Loaded existing load forecast data")
    except FileNotFoundError:
        print("Load forecast file not found. Using dummy data...")
        # Create dummy load forecast for demonstration
        dates = pd.date_range(start=datetime.now(), periods=24*7, freq='H')
        load_forecast = pd.DataFrame({
            'timestamp': dates,
            'predicted_load_kw': np.sin(2 * np.pi * np.arange(len(dates)) / 24) * 20 + 50
        })
        load_forecast.set_index('timestamp', inplace=True)
    
    # Get weather data for PV forecasting
    weather_data = fetch_forecast_data()
    
    # Forecast PV production
    print("Forecasting PV production...")
    pv_production = forecast_pv_production(weather_data)
    
    # Align PV production with load forecast
    pv_production = pv_production.reindex(load_forecast.index, method='nearest')
    
    # Optimize battery dispatch
    print("Optimizing battery dispatch...")
    optimization_results = battery_optimization(pv_production, load_forecast['predicted_load_kw'])
    
    # Calculate metrics
    metrics = calculate_metrics(optimization_results)
    
    # Plot results
    plot_optimization_results(optimization_results, metrics)
    
    # Export results
    optimization_results.to_csv('battery_optimization_schedule.csv', index=False)
    print("Optimization results exported to battery_optimization_schedule.csv")
    
    return optimization_results, metrics

if __name__ == "__main__":
    optimization_results, metrics = main()