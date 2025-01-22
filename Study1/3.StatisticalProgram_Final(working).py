import pandas as pd
import statsmodels.api as sm

#==============================
# Update the Excel file path
excel_file_path = '/Users/User/Desktop/Highest Ranking Model/Research_Results_27.12.23.xlsm'

#==============================
# Read data from Excel file
df = pd.read_excel(excel_file_path)

#==============================
# Remove rows with NaN values
df = df.dropna()

#==============================
# Convert 'created_utc' to datetime format
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)

#==============================
# Find the minimum timestamp and convert 'created_utc' to elapsed time
min_timestamp = df['created_utc'].min()
df['elapsed_time'] = (df['created_utc'] - min_timestamp).dt.total_seconds()

#==============================
# Define time periods (e.g., by day)
df['Time_Period'] = df['created_utc'].dt.to_period('D').astype(str)

#==============================
# Convert 'anon' to a consistent data type (string) to avoid TypeError
df['anon'] = df['anon'].astype(str)

#==============================
# Ensure 'Predicted_Label' is binary (0 or 1)
df['Predicted_Label'] = df['Predicted_Label'].astype(int)

#==============================
# Perform mixed-effects logistic regression using GLM (Generalized Linear Model) with binomial family
# Mixed effects models are used for repeated measures (grouped data)
# The 'anon' column is used as the group identifier
logit_model = sm.GLM(df['Predicted_Label'], 
                     sm.add_constant(df['elapsed_time']),  # Adding a constant to the model
                     family=sm.families.Binomial(), 
                     groups=df['anon'])
result_logit = logit_model.fit()

#==============================
# Display the logistic regression results
print(result_logit.summary())
