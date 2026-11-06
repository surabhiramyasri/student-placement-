import pandas as pd

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Total students
total_students = len(df)

# Count placed and not placed students
placement_counts = df["status"].value_counts()

# Calculate placement percentage
placed_students = (df["status"].str.lower() == "placed").sum()
not_placed_students = total_students - placed_students

placement_percentage = (placed_students / total_students) * 100

print("===== STUDENT PLACEMENT ANALYSIS =====")

print("Total Students:", total_students)
print("Placed Students:", placed_students)
print("Not Placed Students:", not_placed_students)

print("Placement Percentage:", round(placement_percentage, 2), "%")

print("\nPlacement Status:")
print(placement_counts)

print("\nAverage Academic Percentages:")
print("SSC:", round(df["ssc_p"].mean(), 2))
print("Intermediate:", round(df["hsc_p"].mean(), 2))
print("Degree:", round(df["degree_p"].mean(), 2))