import openpyxl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
 #Load data from Excel sheets
df_demand = pd.read_excel('Book2.xlsx', sheet_name='Electricity_demad')
df_weather = pd.read_excel('Book2.xlsx', sheet_name='Load_vs_temp_humidity')
df_area = pd.read_excel('Book2.xlsx', sheet_name='Real_estate')
print(df_demand.columns)
# Datentime demand prediction 
df_demand['Date'] = df_demand['Date'].astype(str)

# If 'Time' is a datetime object, convert it to a string in 'HH:MM:SS' format
df_demand['Time'] = df_demand['Time'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, datetime) else str(x))

# Combine 'Date' and 'Time' to create a new 'DateTime' column
df_demand['DateTime'] = pd.to_datetime(df_demand['Date'] + ' ' + df_demand['Time'], errors='coerce')

# Drop any rows where 'DateTime' conversion failed (if any)
df_demand.dropna(subset=['DateTime'], inplace=True)


# Set the 'DateTime' column as the index
df_demand.set_index('DateTime', inplace=True)

# Drop any rows where the 'DateTime' conversion failed
# df_demand.dropna(subset=['DateTime'], inplace=True)
# Check if the 'DateTime' column exists
print(df_demand.columns)

# If 'DateTime' is not present, reapply the code to combine 'Date' and 'Time'
if 'DateTime' not in df_demand.columns:
    # Ensure 'Date' and 'Time' are in string format
    df_demand['Date'] = df_demand['Date'].astype(str)
    df_demand['Time'] = df_demand['Time'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, datetime) else str(x))

    # Combine 'Date' and 'Time' into 'DateTime'
    df_demand['DateTime'] = pd.to_datetime(df_demand['Date'] + ' ' + df_demand['Time'], errors='coerce')

# Display a preview of the dataframe to check if the 'DateTime' column exists
print(df_demand.head())

# Now drop rows with missing 'DateTime' values
df_demand.dropna(subset=['DateTime'], inplace=True)


# Convert the 'Time' column to string format if it's of type datetime.time
if df_weather['Time'].dtype == 'O':  # Check if it's an object type
    # df_weather['Time'] = df_weather['Time'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, datetime.time) else str(x))
    import pandas as pd
from datetime import datetime  # Import datetime module

# Assuming df_weather is already defined and loaded with data

# Convert the 'Time' column to string format if it's of type datetime.time
df_weather['Time'] = df_weather['Time'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, datetime) else str(x))

# Now create the 'DateTime' column by combining 'Date' and 'Time'
df_weather['DateTime'] = pd.to_datetime(df_weather['Date'].astype(str) + ' ' + df_weather['Time'], errors='coerce')

# Check for any NaT values after the conversion
print("NaT values in DateTime:", df_weather['DateTime'].isna().sum())


# Now create the 'DateTime' column by combining 'Date' and 'Time'
df_weather['DateTime'] = pd.to_datetime(df_weather['Date'].astype(str) + ' ' + df_weather['Time'], errors='coerce')

# Check for any NaT values after the conversion
print("NaT values in DateTime:", df_weather['DateTime'].isna().sum())


# Drop any rows with invalid 'DateTime' conversion
df_weather.dropna(subset=['DateTime'], inplace=True)

# Set the 'DateTime' as index for df_weather too
df_weather.set_index('DateTime', inplace=True)
# Now that both DataFrames have 'DateTime' as index, we can merge them
df_combined = pd.merge(df_demand, df_weather, left_index=True, right_index=True, how='inner')

# Check the combined data
print(df_combined.head())


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Function to get weather forecast
def get_weather_forecast(api_key, location=None, latitude=None, longitude=None):
    if location:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}"
    elif latitude is not None and longitude is not None:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={api_key}"
    else:
        raise ValueError("You must provide either a location name or latitude and longitude.")

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None

    data = response.json()

    # Extract relevant data from the JSON response
    dates = []
    temperatures = []
    humidities = []

    for forecast in data['list']:
        dates.append(datetime.fromtimestamp(forecast['dt']))
        temperatures.append(forecast['main']['temp'] - 273.15)  # Convert from Kelvin to Celsius
        humidities.append(forecast['main']['humidity'])

    # Create a DataFrame with the forecast data
    forecasts = pd.DataFrame({
        'DateTime': dates,
        'Temperature (°C)': temperatures,
        'Humidity (%)': humidities
    })

    return forecasts

# Example usage:
api_key = "c75117b44f72f0277e89ea15fde518a8"  # Replace with your actual API key

# Get weather forecast for Delhi using city name
forecast = get_weather_forecast(api_key, location="Delhi, India")

# Print the forecast DataFrame
print(forecast)

# Add time-related features to the forecast DataFrame
forecast['Hour'] = forecast['DateTime'].dt.hour
forecast['DayOfWeek'] = forecast['DateTime'].dt.dayofweek
forecast['Month'] = forecast['DateTime'].dt.month
forecast['IsWeekend'] = forecast['DayOfWeek'].isin([5, 6]).astype(int)

# Example training data
# Ensure the following columns exist in the actual dataset you are using
X = pd.DataFrame({
    'Temperature (°C)': np.random.uniform(20, 35, size=len(forecast)),  # Example temperature data
    'Humidity (%)': np.random.uniform(30, 90, size=len(forecast)),  # Example humidity data
    'Hour': np.random.randint(0, 24, size=len(forecast)),  # Random example hours
    'DayOfWeek': np.random.randint(0, 7, size=len(forecast)),  # Random example days
    'Month': np.random.randint(8, 13, size=len(forecast)),  # Random example months
    'IsWeekend': np.random.choice([0, 1], size=len(forecast)),  # Random example weekend status
    'No. of Bedrooms': np.random.randint(1, 5, size=len(forecast)),  # Random example data
    'PowerBackup': np.random.choice([0, 1], size=len(forecast)),  # Random example data
})

# Ensure all necessary features are included in both DataFrames
features = ['Temperature (°C)', 'Humidity (%)', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'No. of Bedrooms', 'PowerBackup']

# Check if all required features are in X
missing_features = [feature for feature in features if feature not in X.columns]
if missing_features:
    raise KeyError(f"The following features are missing in X: {missing_features}")

# Add area features to forecast DataFrame
forecast['No. of Bedrooms'] = X['No. of Bedrooms'].mean()  # Using mean of random example data
forecast['PowerBackup'] = X['PowerBackup'].mode()[0]  # Using mode of random example data

# Check if all required features are in forecast
missing_forecast_features = [feature for feature in features if feature not in forecast.columns]
if missing_forecast_features:
    raise KeyError(f"The following features are missing in forecast: {missing_forecast_features}")

# Scale training features
scaler = StandardScaler()  # Initialize scaler
scaler.fit(X[features])  # Fit on training data
X_scaled = scaler.transform(X[features])  # Scale the training features

# Initialize and train the model
model = RandomForestRegressor()
# Assuming y is your target variable (this should be defined appropriately)
y = np.random.randint(50, 150, size=len(X))  # Example target variable (random data)
model.fit(X_scaled, y)  # Fit the model with scaled training data

# Scale forecast features
forecast_features = forecast[features]  # Extract the relevant features from forecast
forecast_scaled = scaler.transform(forecast_features)  # Scale the forecast features

# Make predictions
forecast['Predicted_Demand'] = model.predict(forecast_scaled)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(forecast['DateTime'], forecast['Predicted_Demand'])
plt.title('Forecasted Electricity Demand')
plt.xlabel('Date')
plt.ylabel('Demand (MW)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the predictions along with datetime
print(forecast[['DateTime', 'Predicted_Demand']])



