import pandas as pd
import statsmodels.api as sm
import numpy as np

#==============================
# Update the Excel file path
excel_file_path = '/Users/User/Desktop/Highest Ranking Model/Research_Results_27.12.23.xlsm'

#==============================
# Read data from Excel file
df = pd.read_excel(excel_file_path)

#==============================
# Convert 'created_utc' to datetime format and handle time zone information
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', utc=True).dt.tz_convert(None)

#==============================
# Find the minimum timestamp and convert 'created_utc' to elapsed time
min_timestamp = df['created_utc'].min()
df['elapsed_time'] = (df['created_utc'] - min_timestamp).dt.total_seconds()

#==============================
# Keep time periods with seconds precision (no change here as per your request)
df['Time_Period'] = df['created_utc'].dt.to_period('S')  # Keep 'S' for second-level precision

#==============================
# Convert 'anon' to string to ensure consistent data type
df['anon'] = df['anon'].astype(str)

#==============================
# Drop rows with missing or NaN values in any column
df = df.dropna()

#==============================
# Create a new DataFrame with required columns
new_df = pd.DataFrame({
    'Predicted_Label': df['Predicted_Label'].astype(float),  # Ensure numeric data type
    'elapsed_time': df['elapsed_time'],
    'anon': df['anon']
})

#==============================
# Ensure the dependent variable (Predicted_Label) is binary
new_df['Predicted_Label'] = new_df['Predicted_Label'].astype(int)

#==============================
# Perform logistic regression (Generalized Linear Model with binomial family)
logit_model = sm.GLM(new_df['Predicted_Label'], 
                     sm.add_constant(new_df['elapsed_time']),  # Adding constant (intercept)
                     family=sm.families.Binomial(),
                     groups=new_df['anon'])
result_logit = logit_model.fit()

#==============================
# Display logistic regression results
print(result_logit.summary())
