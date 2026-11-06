
import pandas as pd

# Load the student placement dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

print("Dataset loaded successfully!")
print("Dataset size:", df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())
