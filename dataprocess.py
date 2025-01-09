import pandas as pd
from sklearn.utils import shuffle

# Opt into future behavior
pd.set_option('future.no_silent_downcasting', True)

# File paths
input_file = r"D:\cirrhosis\Cirrhosis-Patient-Survival-Prediction-main\cirrhosis1.csv"
final_output_file = r"D:\cirrhosis\Cirrhosis-Patient-Survival-Prediction-main\balanced_cirrhosis.csv"

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

# Separate the stages
stage_1 = dataset[dataset['Stage'] == 1.0]
stage_2 = dataset[dataset['Stage'] == 2.0]

# Randomly sample Stage 2 to match Stage 1 count
stage_2_sampled = stage_2.sample(n=len(stage_1), random_state=42)

# Combine and shuffle the dataset
balanced_data = pd.concat([stage_1, stage_2_sampled], axis=0)
balanced_data = shuffle(balanced_data, random_state=42).reset_index(drop=True)

# Save the final balanced dataset
balanced_data.to_csv(final_output_file, index=False)

print(f"The final balanced dataset has been saved to {final_output_file}")

