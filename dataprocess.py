import pandas as pd

# Opt into future behavior
pd.set_option('future.no_silent_downcasting', True)

# File paths
input_file = r"D:\cirrhosis\Cirrhosis-Patient-Survival-Prediction-main\cirrhosis1.csv"
output_file = r"D:\cirrhosis\Cirrhosis-Patient-Survival-Prediction-main\cirrhosis.csv"

# Load dataset as dataframe (csv file)
dataset = pd.read_csv(input_file)

# Remove the 'ID' column if it exists
if 'ID' in dataset.columns:
    dataset.drop(columns=['ID'], inplace=True)

# Cleaning null values for numeric columns
numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    dataset[col] = dataset[col].fillna(dataset[col].median())

# Cleaning null values for object columns
object_cols = dataset.select_dtypes(include=['object']).columns
for col in object_cols:
    dataset[col] = dataset[col].fillna(dataset[col].mode().values[0])

# Replace categorical values with numerical values
dataset["Sex"] = dataset["Sex"].replace({"M": 0, "F": 1})
dataset["Stage"] = dataset["Stage"].replace({4: 2, 1: 1, 2: 1, 3: 2})
dataset["Hepatomegaly"] = dataset["Hepatomegaly"].replace({"Y": 1, "N": 0})
dataset["Ascites"] = dataset["Ascites"].replace({"Y": 1, "N": 0})
dataset["Edema"] = dataset["Edema"].replace({"Y": 1, "N": 0, "S": 0})
dataset["Status"] = dataset["Status"].replace({"CL": 0, "C": 1, "D": 0})
dataset["Drug"] = dataset["Drug"].replace({"D-penicillamine": 1, "Placebo": 0})
dataset["Spiders"] = dataset["Spiders"].replace({"Y": 1, "N": 0})

# Save the cleaned and transformed dataset to a new CSV file
dataset.to_csv(output_file, index=False)

print(f"Cleaned and scaled dataset saved to {output_file}")