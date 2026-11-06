import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Graph 1: Placement Status
placement_counts = df["status"].value_counts()

placement_counts.plot(kind="bar")

plt.title("Student Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("placement_status.png")
plt.show()

# Graph 2: Average Academic Performance
subjects = ["SSC", "Intermediate", "Degree"]

averages = [
    df["ssc_p"].mean(),
    df["hsc_p"].mean(),
    df["degree_p"].mean()
]

plt.figure()
plt.bar(subjects, averages)

plt.title("Average Academic Performance")
plt.xlabel("Academic Level")
plt.ylabel("Average Percentage")
plt.tight_layout()
plt.savefig("academic_performance.png")
plt.show()

print("Both graphs created successfully!")