# import openpyxl
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import requests
# from datetime import datetime, timedelta
 #Load data from Excel sheets
df_demand = pd.read_excel('Book2.xlsx', sheet_name='Electricity_demad')
df_weather = pd.read_excel('Book2.xlsx', sheet_name='Load_vs_temp_humidity')
df_area = pd.read_excel('Book2.xlsx', sheet_name='Real_estate')
print(df_demand.columns)
