# wild_life_main.py

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('wild_life_data.csv')

# Encode 'Migrating' to binary (Yes=1, No=0)
df['Migrating'] = df['Migrating'].map({'Yes': 1, 'No': 0})

# Scale Health Index to % if in decimal
df['Health_Index'] = df['Health_Index'].apply(lambda x: x*100 if x < 1 else x)

# Define features and target
X = df[['Population_Count', 'Migrating', 'Avg_Temp_C', 'Rainfall_mm',
        'Wildfire_Incidents', 'Vegetation_Loss_%']]
y = df['Health_Index']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print("R² Score:", r2)
print("RMSE:", rmse)

# 1. Extinction Risk Calculation
df['Extinction_Risk_%'] = (
    (1 - df['Population_Count'] / df['Population_Count'].max()) * 30 +
    (df['Vegetation_Loss_%'] / 100) * 20 +
    (df['Wildfire_Incidents'] / df['Wildfire_Incidents'].max()) * 30 +
    (1 - df['Health_Index'] / 100) * 20
)

# Plot: Extinction Risk Distribution
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.histplot(df['Extinction_Risk_%'], bins=20, kde=True, color='darkred')
plt.title('Extinction Risk Distribution')
plt.xlabel('Estimated Extinction Risk (%)')
plt.tight_layout()
plt.show()

# 2. Population vs Health Index
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Population_Count', y='Health_Index',
                hue='Migrating', palette='Set1')
plt.title('Population Count vs Health Index')
plt.xlabel('Population Count')
plt.ylabel('Health Index (%)')
plt.tight_layout()
plt.show()

# 3. Wildfire Incidents vs Health Index
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Wildfire_Incidents', y='Health_Index', palette='Reds')
plt.title('Wildfire Incidents vs Health Index Distribution')
plt.xlabel('Number of Wildfires')
plt.ylabel('Health Index (%)')
plt.tight_layout()
plt.show()

# 4. Environmental Correlation Heatmap
plt.figure(figsize=(10, 6))
env_features = df[['Avg_Temp_C', 'Rainfall_mm', 'Wildfire_Incidents',
                   'Vegetation_Loss_%', 'Health_Index']]
corr = env_features.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Environmental Factors and Health')
plt.tight_layout()
plt.show()

# 5. Vegetation Loss vs Health Index (with Wildfire Incidents)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Vegetation_Loss_%', y='Health_Index',
                hue='Wildfire_Incidents', size='Wildfire_Incidents',
                palette='flare', sizes=(20, 200))
plt.title('Vegetation Loss vs Health Index (Wildfire Intensity)')
plt.xlabel('Vegetation Loss (%)')
plt.ylabel('Health Index (%)')
plt.tight_layout()
plt.show()

# 6. Rainfall and Health Index Boxplot (Quantiles)
df['Rainfall_Bins'] = pd.qcut(df['Rainfall_mm'], q=4)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Rainfall_Bins', y='Health_Index', palette='Blues')
plt.title('Health Index Across Rainfall Quantiles')
plt.xlabel('Rainfall Ranges (mm)')
plt.ylabel('Health Index (%)')
plt.tight_layout()
plt.show()
