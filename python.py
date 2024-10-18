#All the necessary libraries
import openpyxl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import joblib  

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
        temperatures.append(forecast['main']['temp'] - 273.15)  
        humidities.append(forecast['main']['humidity'])

    # Create a DataFrame with the forecast data
    forecasts = pd.DataFrame({
        'DateTime': dates,
        'Temperature (°C)': temperatures,
        'Humidity (%)': humidities
    })

    return forecasts

# Example usage:
api_key = "c75117b44f72f0277e89ea15fde518a8"  

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
    'Temperature (°C)': np.random.uniform(20, 35, size=len(forecast)),  
    'Humidity (%)': np.random.uniform(30, 90, size=len(forecast)),  
    'Hour': np.random.randint(0, 24, size=len(forecast)),
    'DayOfWeek': np.random.randint(0, 7, size=len(forecast)), 
    'Month': np.random.randint(8, 13, size=len(forecast)), 
    'IsWeekend': np.random.choice([0, 1], size=len(forecast)),  
    'No. of Bedrooms': np.random.randint(1, 5, size=len(forecast)), 
    'PowerBackup': np.random.choice([0, 1], size=len(forecast)),  
})

# Ensure all necessary features are included in both DataFrame
features = ['Temperature (°C)', 'Humidity (%)', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'No. of Bedrooms', 'PowerBackup']

# Check if all required features are in X
missing_features = [feature for feature in features if feature not in X.columns]
if missing_features:
    raise KeyError(f"The following features are missing in X: {missing_features}")

# Add area features to forecast DataFrame
forecast['No. of Bedrooms'] = X['No. of Bedrooms'].mean()  
forecast['PowerBackup'] = X['PowerBackup'].mode()[0] 

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
y = np.random.randint(50, 150, size=len(X))  
model.fit(X_scaled, y)  

# Scale forecast features
forecast_features = forecast[features]  
forecast_scaled = scaler.transform(forecast_features)  

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



# Load data from Excel sheets
df_demand = pd.read_excel('Book2.xlsx', sheet_name='Electricity_demad')
df_weather = pd.read_excel('Book2.xlsx', sheet_name='Load_vs_temp_humidity')
df_area = pd.read_excel('Book2.xlsx', sheet_name='Real_estate')

# Handle missing values in the Demand DataFrame
print("Missing values in Demand DataFrame:")
print(df_demand.isnull().sum())

# Drop rows with missing values in the Demand DataFrame
df_demand.dropna(subset=['Time', 'Demand', 'Un-Restricted Demand', 'Date'], inplace=True)

# Handle missing values in the Weather DataFrame
print("Missing values in Weather DataFrame:")
print(df_weather.isnull().sum())

# Drop rows with missing values in the Weather DataFrame
df_weather.dropna(subset=['Date', 'Time', 'Load (MW)', 'Temperature (°C)', 'Humidity (%)'], inplace=True)

# Ensure 'Date' columns in both DataFrames are datetime types
df_demand['Date'] = pd.to_datetime(df_demand['Date'])
df_weather['Date'] = pd.to_datetime(df_weather['Date'])

# Convert Time columns to timedelta, handling possible errors
def convert_time_to_timedelta(time):
    if isinstance(time, str) and time:
        try:
            return pd.to_timedelta(time)
        except ValueError:
            return pd.NaT
    return pd.NaT

# Apply the function to both DataFrames
df_demand['Time'] = df_demand['Time'].apply(convert_time_to_timedelta)
df_weather['Time'] = df_weather['Time'].apply(convert_time_to_timedelta)

# Drop any rows in Demand or Weather DataFrames that still have NaT in 'Time'
df_demand.dropna(subset=['Time'], inplace=True)
df_weather.dropna(subset=['Time'], inplace=True)



# Assuming df_demand and df_weather are your DataFrames

# First, ensure that the 'Date' and 'Time' columns are of string type
df_demand['Date'] = df_demand['Date'].astype(str)
df_demand['Time'] = df_demand['Time'].astype(str)
df_weather['Date'] = df_weather['Date'].astype(str)
df_weather['Time'] = df_weather['Time'].astype(str)

# Create DateTime columns by combining 'Date' and 'Time'
df_demand['DateTime'] = pd.to_datetime(df_demand['Date'] + ' ' + df_demand['Time'])
df_weather['DateTime'] = pd.to_datetime(df_weather['Date'] + ' ' + df_weather['Time'])

# Now you can proceed with merging the DataFrames or any further operations


# Merge the two DataFrames on the DateTime column
df_combined = pd.merge(df_demand, df_weather, on='DateTime', how='inner')

# Check the shape and contents of the combined DataFrame
print("Combined DataFrame Shape:", df_combined.shape)
print(df_combined.head())

# Clean column names by stripping whitespace and replacing spaces with underscores
df_combined.columns = df_combined.columns.str.strip().str.replace(' ', '_')

# Print cleaned column names for debugging
print("Cleaned Combined DataFrame Columns:", df_combined.columns)

# Define target variable after confirming the column name
target = 'Un-Restricted_Demand'  

# Feature Engineering
if not df_combined.empty:
    # Create additional time-based features
    df_combined['Hour'] = df_combined['DateTime'].dt.hour
    df_combined['DayOfWeek'] = df_combined['DateTime'].dt.dayofweek
    df_combined['Month'] = df_combined['DateTime'].dt.month
    df_combined['IsWeekend'] = df_combined['DayOfWeek'].isin([5, 6]).astype(int)

    # Display the first few rows with new features
    print("DataFrame with Features:")
    print(df_combined[['DateTime', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend']].head())

    # Define features and target
    features = ['Temperature_(°C)', 'Humidity_(%)', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend']

    # Check for missing features
    missing_features = [feature for feature in features if feature not in df_combined.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
    else:
        X = df_combined[features]
        y = df_combined[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R-squared Score: {r2}")

        # Data Visualization
        plt.show()


        # 6. Check if the merged DataFrame is empty
        if df_combined.empty:
            print("Merged DataFrame is empty; skipping feature engineering.")
else:
    print("Merged DataFrame is empty; skipping feature engineering.")




# Set random seed for reproducibility
np.random.seed(42)

# Generate random data for 10 days at hourly intervals
date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H')

# Create random demand and weather data
df = pd.DataFrame(date_rng, columns=['DateTime'])
df['Demand'] = np.random.randint(50, 100, size=(len(date_rng)))
df['Load_Shedding'] = np.random.randint(0, 20, size=(len(date_rng)))
df['Unrestricted_Demand'] = df['Demand'] + df['Load_Shedding']
df['Temperature'] = np.random.uniform(15, 30, size=(len(date_rng)))
df['Humidity'] = np.random.uniform(40, 100, size=(len(date_rng)))

# Feature Engineering: Extract hour, day of week, month, and weekend status
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Define features and target
features = ['Temperature', 'Humidity', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend']
target = 'Unrestricted_Demand'

# Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Visualizations

# 1. Actual vs Predicted Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Actual vs Predicted Demand')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Feature Importance Plot
plt.figure(figsize=(10, 6))
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(np.array(features)[sorted_idx], feature_importance[sorted_idx], color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


# 4. Demand vs Temperature Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature'], df['Unrestricted_Demand'], alpha=0.5)
plt.title('Demand vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Unrestricted Demand')
plt.tight_layout()
plt.show()

# 5. Demand by Hour of the Day (Box Plot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Hour', y='Unrestricted_Demand', data=df)
plt.title('Demand Distribution by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Unrestricted Demand')
plt.tight_layout()
plt.show()
# Actual vs Predicted Load 
plt.figure(figsize=(12,6))
plt.plot(y_test.reset_index(drop=True), label='Actual Load')
plt.plot(y_pred, label='Predicted Load', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.title('Actual vs Predicted Load')
plt.legend()
plt.show()




# Load the data
path = "/content/Year Data.xlsx"  
data = pd.read_excel(path, parse_dates=["Date"])
data.columns = data.columns.str.strip()  

# Set the "Date" column as the index
data.set_index("Date", inplace=True)


print("Columns in DataFrame:", data.columns)

if 'Energy' not in data.columns:
    raise ValueError("The 'Energy' column is not present in the data.")

def create_features(df):
    df = df.copy()

    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Creating basic datetime features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear

    # Create lag features
    for lag in [1, 7, 14, 30]:
        df[f'Energy_lag{lag}'] = df['Energy'].shift(lag)

    # Create rolling mean features
    for window in [7, 14, 30]:
        df[f'Energy_rolling_mean{window}'] = df['Energy'].rolling(window=window).mean()

    return df

# Prepare data
data = create_features(data)
data = data.dropna()  

# Define features and target
X = data[['year', 'month', 'day_of_week', 'day_of_year',
           'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
           'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']]
y = data['Energy']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to create features for a new date based on historical data
def create_features_for_prediction(date, historical_data):
    
    pred_df = pd.DataFrame(index=[date])
    pred_df.index = pd.to_datetime(pred_df.index)  

    # Create basic datetime features
    pred_df['year'] = pred_df.index.year
    pred_df['month'] = pred_df.index.month
    pred_df['day_of_week'] = pred_df.index.dayofweek
    pred_df['day_of_year'] = pred_df.index.dayofyear

    
    for lag in [1, 7, 14, 30]:
        pred_df[f'Energy_lag{lag}'] = historical_data['Energy'].shift(lag).iloc[-1]

    # Calculate rolling mean features using the historical data
    for window in [7, 14, 30]:
        pred_df[f'Energy_rolling_mean{window}'] = historical_data['Energy'].rolling(window=window).mean().iloc[-1]

    return pred_df


def predict_for_date(date):
    
    pred_df = create_features_for_prediction(date, data)

    features = ['year', 'month', 'day_of_week', 'day_of_year',
                'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']

    # Make prediction
    prediction = model.predict(pred_df[features])

    return prediction[0]

# Function to get user input and make prediction
def get_prediction():
    current_date = datetime.now().date()  
    print(f"Current date is: {current_date}")  

    
    while True:
        date_str = input("Enter a date (YYYY-MM-DD) within the next 30 days (or type 'exit' to quit): ").strip()

       
        if date_str.lower() == 'exit':
            print("Exiting the prediction tool.")
            return

        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            # Check if the date is within the next 30 days
            if current_date <= input_date <= current_date + timedelta(days=30):
                break
            else:
                print("Please enter a date within the next 30 days.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Make prediction
    prediction = predict_for_date(input_date)
    print(f"Predicted electricity consumption on {input_date}: {prediction:.2f} MWh")

# Main execution
while True:
    get_prediction()
    if input("Do you want to make another prediction? (yes/no): ").lower() != 'yes':
        break

print("Thank you for using the electricity consumption prediction model!")



# Example DataFrame structure with an empty string in the Real_Estate column
data = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Real_Estate': ['Residential', 'Commercial', 'Industrial', '']  # Added empty string
})

# Updated function to get user input and make prediction
def get_prediction():
    current_date = datetime.now().date()
    print(f"Current date is: {current_date}")

    # Get location input from the user with clear instructions
    location = input("Enter the location (choose from: North, South, East, West, Residential, Commercial, Industrial): ").strip()

    # Ask for the date input
    while True:
        date_str = input("Enter a date (YYYY-MM-DD) within the next 30 days (or type 'exit' to quit): ").strip()

        if date_str.lower() == 'exit':
            print("Exiting the prediction tool.")
            return

        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if current_date <= input_date <= current_date + timedelta(days=30):
                break
            else:
                print("Please enter a date within the next 30 days.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Check if the location is found in the data
    if 'Region' in data.columns and 'Real_Estate' in data.columns:
        if location in data['Region'].unique() or location in data['Real_Estate'].unique():
            print(f"{location} is present in the data.")
            
            # Logic for actual prediction goes here (not shown here)
            # Assuming the prediction logic is handled elsewhere
            prediction = 150.0  # Placeholder for actual model prediction
        else:
            print(f"{location} is not found in either Region or Real Estate.")
            
            # Generate random features for prediction since location is not found
            X_random = pd.DataFrame({
                'year': [input_date.year],
                'month': [input_date.month],
                'day_of_week': [input_date.weekday()],
                'day_of_year': [input_date.timetuple().tm_yday],
                'Energy_lag1': [random.uniform(50, 200)],
                'Energy_lag7': [random.uniform(50, 200)],
                'Energy_lag14': [random.uniform(50, 200)],
                'Energy_lag30': [random.uniform(50, 200)],
                'Energy_rolling_mean7': [random.uniform(50, 200)],
                'Energy_rolling_mean14': [random.uniform(50, 200)],
                'Energy_rolling_mean30': [random.uniform(50, 200)],
                'Temperature': [random.uniform(20, 40)],  # Random temperature
            })

            # Create and fit the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_random, [random.uniform(50, 200)])  

            prediction = model.predict(X_random)

        # Combined output message
        if isinstance(prediction, (list, pd.Series, np.ndarray)):
            prediction_value = prediction[0]
        else:
            prediction_value = prediction

        print(f"Predicted electricity consumption on {input_date} for '{location}': {prediction_value:.2f} MWh")

    else:
        print("One or both columns 'Region' and 'Real_Estate' do not exist in the DataFrame.")

# Run the prediction function
get_prediction()


# Load the data
path = r"/content/Delhi_Energy_Data (1).xlsx"  # Ensure this is the correct path to your Excel file
data = pd.read_excel(path, parse_dates=["Date"])
data.columns = data.columns.str.strip()  # Remove any extra spaces

# Load real estate data
real_estate_data = pd.read_excel(r"/content/real_estate_1.xlsx")
real_estate_data.columns = real_estate_data.columns.astype(str).str.strip()  # Remove any extra spaces

# Print the columns of the real estate data for debugging
print("Columns in the real estate data:")
print(real_estate_data.columns.tolist())

# Check if 'Energy' and 'Region' columns exist in energy data
if 'Energy' not in data.columns or 'Region' not in data.columns:
    raise ValueError("The 'Energy' or 'Region' column is not present in the energy data.")

# Function to create features
def create_features(df):
    df = df.copy()

    # Ensure the index is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Create basic datetime features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear

    # Create lag features
    for lag in [1, 7, 14, 30]:
        df[f'Energy_lag{lag}'] = df['Energy'].shift(lag)

    # Create rolling mean features
    for window in [7, 14, 30]:
        df[f'Energy_rolling_mean{window}'] = df['Energy'].rolling(window=window).mean()

    return df

# Prepare data
data.set_index("Date", inplace=True)
data = create_features(data)
data = data.dropna()  # Drop rows with NaN values

# Define features and target for a specific region
def prepare_data_for_region(region):
    region_data = data[data['Region'] == region]

    if region_data.empty:
        raise ValueError(f"No data found for the region: {region}")

    X = region_data[['year', 'month', 'day_of_week', 'day_of_year',
                     'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                     'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']]
    y = region_data['Energy']

    return X, y

# Prepare data for real estate
def prepare_data_for_real_estate(location):
    location_data = real_estate_data[real_estate_data['Location'] == location]

    if location_data.empty:
        raise ValueError(f"No data found for the location: {location}")

    # Assuming real estate data has a 'date' column for date-related features
    if 'date' not in location_data.columns:
        raise ValueError("The 'date' column is not present in the real estate data.")

    location_data.set_index('date', inplace=True)
    location_data = create_features(location_data)

    # Define features and target
    X = location_data[['year', 'month', 'day_of_week', 'day_of_year',
                       'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                       'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']]
    y = location_data['Energy']

    return X, y

# Function to create features for a new date based on historical data for a specific region
def create_features_for_prediction(date, historical_data):
    pred_df = pd.DataFrame(index=[date])
    pred_df.index = pd.to_datetime(pred_df.index)

    pred_df['year'] = pred_df.index.year
    pred_df['month'] = pred_df.index.month
    pred_df['day_of_week'] = pred_df.index.dayofweek
    pred_df['day_of_year'] = pred_df.index.dayofyear

    for lag in [1, 7, 14, 30]:
        pred_df[f'Energy_lag{lag}'] = historical_data['Energy'].shift(lag).iloc[-1]

    for window in [7, 14, 30]:
        pred_df[f'Energy_rolling_mean{window}'] = historical_data['Energy'].rolling(window=window).mean().iloc[-1]

    return pred_df

# Function to predict for a specific date and region/location
def predict_for_date(date, region_or_location, model, is_real_estate=False):
    if is_real_estate:
        location_data = real_estate_data[real_estate_data['Location'] == region_or_location]

        if location_data.empty:
            raise ValueError(f"No data found for the location: {region_or_location}")

        pred_df = create_features_for_prediction(date, location_data.set_index('date'))

    else:
        region_data = data[data['Region'] == region_or_location]

        if region_data.empty:
            raise ValueError(f"No data found for the region: {region_or_location}")

        pred_df = create_features_for_prediction(date, region_data)

    features = ['year', 'month', 'day_of_week', 'day_of_year',
                'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']

    prediction = model.predict(pred_df[features])

    return prediction[0]

# Function to get user input and make prediction for a specific region or real estate
def get_prediction():
    current_date = datetime.now().date()
    print(f"Current date is: {current_date}")

    # Ask user if they want to predict based on region or real estate
    prediction_type = input("Do you want to predict energy consumption for 'Region' or 'Real Estate'? ").strip().lower()

    if prediction_type not in ['region', 'real estate']:
        print("Invalid input. Please enter 'Region' or 'Real Estate'.")
        return

    if prediction_type == 'real estate':
        # Get location input from the real estate data
        print("Available locations in real estate data:")

        if 'Location' not in real_estate_data.columns:
            print("The 'Location' column is not found in the real estate data. Please check the column names.")
            return

        print(real_estate_data['Location'].unique())  # Print unique locations for selection

        location = input("Enter the location of the real estate: ").strip()

        if location not in real_estate_data['Location'].values:
            print(f"No data found for the location: {location}")
            return

    else:
        # Get region input from user first
        region = input("Enter the region in Delhi where you want to predict electricity consumption: ").strip()

        if region not in data['Region'].unique():
            print(f"No data found for the region: {region}")
            return

    while True:
        # Get date input from user
        date_str = input("Enter a date (YYYY-MM-DD) within the next 30 days (or type 'exit' to quit): ").strip()

        if date_str.lower() == 'exit':
            print("Exiting the prediction tool.")
            return

        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if current_date <= input_date <= current_date + timedelta(days=30):
                break
            else:
                print("Please enter a date within the next 30 days.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Prepare data for prediction
    if prediction_type == 'real estate':
        X, y = prepare_data_for_real_estate(location)
        is_real_estate = True
    else:
        X, y = prepare_data_for_region(region)
        is_real_estate = False

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    prediction = predict_for_date(input_date, region if prediction_type == 'region' else location, model, is_real_estate)
    print(f"Predicted electricity consumption on {input_date} in {region if prediction_type == 'region' else location}: {prediction:.2f} MWh")

# Main execution
while True:
    get_prediction()
    if input("Do you want to make another prediction? (yes/no): ").lower() != 'yes':
        break

print("Thank you for using the electricity consumption prediction model!")


# Load the data
def load_data():
    try:
        energy_data_path = r"/content/Delhi_Energy_Data (1).xlsx"
        data = pd.read_excel(energy_data_path, parse_dates=["Date"])
        data.columns = data.columns.str.strip()  

        real_estate_data_path = r"/content/cleaned_real_estate_data.csv"
        real_estate_data = pd.read_excel(real_estate_data_path)
        real_estate_data.columns = real_estate_data.columns.str.strip()  

        return data, real_estate_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Create features
def create_features(df):
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear

    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'Energy_lag{lag}'] = df['Energy'].shift(lag)

    # Rolling mean features
    for window in [7, 14, 30]:
        df[f'Energy_rolling_mean{window}'] = df['Energy'].rolling(window=window).mean()

    return df

# Prepare data for a specific region
def prepare_data_for_region(data, region):
    region_data = data[data['Region'] == region]

    if region_data.empty:
        raise ValueError(f"No data found for the region: {region}")

    X = region_data[['year', 'month', 'day_of_week', 'day_of_year',
                     'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                     'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']]
    y = region_data['Energy']

    return X, y

# Prepare data for real estate
def prepare_data_for_real_estate(real_estate_data, location):
    location_data = real_estate_data[real_estate_data['Location'] == location]

    if location_data.empty:
        raise ValueError(f"No data found for the location: {location}")

    location_data = create_features(location_data.set_index('Date'))
    X = location_data[['year', 'month', 'day_of_week', 'day_of_year',
                       'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                       'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']]
    y = location_data['Energy']

    return X, y

# Create features for a new date based on historical data
def create_features_for_prediction(date, historical_data):
    pred_df = pd.DataFrame(index=[date])
    pred_df.index = pd.to_datetime(pred_df.index)

    pred_df['year'] = pred_df.index.year
    pred_df['month'] = pred_df.index.month
    pred_df['day_of_week'] = pred_df.index.dayofweek
    pred_df['day_of_year'] = pred_df.index.dayofyear

    # Use last available values for lags and rolling means
    for lag in [1, 7, 14, 30]:
        pred_df[f'Energy_lag{lag}'] = historical_data['Energy'].shift(lag).iloc[-1]

    for window in [7, 14, 30]:
        pred_df[f'Energy_rolling_mean{window}'] = historical_data['Energy'].rolling(window=window).mean().iloc[-1]

    return pred_df

# Predict for a specific date and region
def predict_for_date(date, region, model, data):
    region_data = data[data['Region'] == region]

    if region_data.empty:
        raise ValueError(f"No data found for the region: {region}")

    pred_df = create_features_for_prediction(date, region_data)

    features = ['year', 'month', 'day_of_week', 'day_of_year',
                'Energy_lag1', 'Energy_lag7', 'Energy_lag14', 'Energy_lag30',
                'Energy_rolling_mean7', 'Energy_rolling_mean14', 'Energy_rolling_mean30']

    prediction = model.predict(pred_df[features])

    return prediction[0]

# Load model from file
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Save model to file
def save_model(model, model_path):
    joblib.dump(model, model_path)

# Function to get user input and make prediction for a specific region or real estate
def get_prediction(data, real_estate_data):
    current_date = datetime.now().date()
    print(f"Current date is: {current_date}")

    prediction_type = input("Do you want to predict energy consumption for 'Region' or 'Real Estate'? ").strip().lower()

    if prediction_type not in ['region', 'real estate']:
        print("Invalid input. Please enter 'Region' or 'Real Estate'.")
        return

    if prediction_type == 'real estate':
        # List available locations
        print("Available locations in real estate data:")
        if 'Location' not in real_estate_data.columns:
            print("The 'Location' column is not found in the real estate data.")
            return

        print(real_estate_data['Location'].unique())
        location = input("Enter the location of the real estate: ").strip()

    else:
        region = input("Enter the region in Delhi where you want to predict electricity consumption: ").strip()

    # Validate location or region
    if prediction_type == 'real estate' and location not in real_estate_data['Location'].values:
        print(f"No data found for the location: {location}")
        return
    elif prediction_type == 'region' and region not in data['Region'].unique():
        print(f"No data found for the region: {region}")
        return

    while True:
        date_str = input("Enter a date (YYYY-MM-DD) within the next 30 days (or type 'exit' to quit): ").strip()

        if date_str.lower() == 'exit':
            print("Exiting the prediction tool.")
            return

        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if current_date <= input_date <= current_date + timedelta(days=30):
                break
            else:
                print("Please enter a date within the next 30 days.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Prepare data and model for prediction
    if prediction_type == 'real estate':
        X, y = prepare_data_for_real_estate(real_estate_data, location)
    else:
        X, y = prepare_data_for_region(data, region)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model for future use
    save_model(model, 'energy_prediction_model.joblib')

    prediction = predict_for_date(input_date, region if prediction_type == 'region' else location, model, data)
    print(f"Predicted electricity consumption on {input_date} in {region if prediction_type == 'region' else location}: {prediction:.2f} MWh")

# Main execution
def main():
    data, real_estate_data = load_data()
    while True:
        get_prediction(data, real_estate_data)

# Example DataFrame structure with an empty string in the Real_Estate column
data = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Real_Estate': ['Residential', 'Commercial', 'Industrial', '']  # Added empty string
})

# Updated function to get user input and make prediction
def get_prediction():
    current_date = datetime.now().date()
    print(f"Current date is: {current_date}")

    # Get location input from the user with clear instructions
    location = input("Enter the location (choose from: North, South, East, West, Residential, Commercial, Industrial): ").strip()

    # Ask for the date input
    while True:
        date_str = input("Enter a date (YYYY-MM-DD) within the next 30 days (or type 'exit' to quit): ").strip()

        if date_str.lower() == 'exit':
            print("Exiting the prediction tool.")
            return

        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if current_date <= input_date <= current_date + timedelta(days=30):
                break
            else:
                print("Please enter a date within the next 30 days.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

# Check if the location is fpound in the data
    if 'Region' in data.columns and 'Real_Estate' in data.columns:
        if location in data['Region'].unique() or location in data['Real_Estate'].unique():
            print(f"{location} is present in the data.")
            
            # Logic for actual prediction goes here (not shown here)
            # Assuming the prediction logic is handled elsewhere
            prediction = 150.0  # Placeholder for actual model prediction
        else:
            print(f"{location} is not found in either Region or Real Estate.")
            
            # Generate random features for prediction since location is not found
            X_random = pd.DataFrame({
                'year': [input_date.year],
                'month': [input_date.month],
                'day_of_week': [input_date.weekday()],
                'day_of_year': [input_date.timetuple().tm_yday],
                'Energy_lag1': [random.uniform(50, 200)],
                'Energy_lag7': [random.uniform(50, 200)],
                'Energy_lag14': [random.uniform(50, 200)],
                'Energy_lag30': [random.uniform(50, 200)],
                'Energy_rolling_mean7': [random.uniform(50, 200)],
                'Energy_rolling_mean14': [random.uniform(50, 200)],
                'Energy_rolling_mean30': [random.uniform(50, 200)],
                'Temperature': [random.uniform(20, 40)],  # Random temperature
            })
          # Create and fit the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_random, [random.uniform(50, 200)])  # Fit with random target for demonstration

            prediction = model.predict(X_random)

        # Combined output message
        if isinstance(prediction, (list, pd.Series, np.ndarray)):
            prediction_value = prediction[0]
        else:
            prediction_value = prediction

        print(f"Predicted electricity consumption on {input_date} for '{location}': {prediction_value:.2f} MWh")

    else:
        print("One or both columns 'Region' and 'Real_Estate' do not exist in the DataFrame.")

# Run the prediction function
get_prediction()





