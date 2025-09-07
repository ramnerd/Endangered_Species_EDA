# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder



# Load and preprocess data
df = pd.read_csv('climate_data.csv')

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Risk_Level':
        df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
X = df.drop('Risk_Level', axis=1)
y = df['Risk_Level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]


# Correlation Heatmap
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
numerical_df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
