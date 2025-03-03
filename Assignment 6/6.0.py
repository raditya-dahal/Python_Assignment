import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load dataset
df = pd.read_csv("bank.csv", sep=';')
print(df.info())  # Inspect column names and data types
print(df.head())

# Step 2: Select relevant columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Convert categorical variables to dummy variables
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

# Step 4: Heatmap of correlation coefficients
plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap of Correlation Coefficients")
plt.show()

"""
From the heatmap, we observe that:
- The correlation values are generally low, indicating that no single variable strongly predicts 'y'.
- Some variables show slight correlations with each other, likely due to categorical encoding.
- Multicollinearity is minimal, making logistic regression feasible.
"""

# Step 5: Define target variable and explanatory variables
y = df3['y'].map({'yes': 1, 'no': 0})  # Convert 'yes'/'no' to 1/0
X = df3.drop(columns=['y'])

# Step 6: Split dataset into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Train and evaluate logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Step 8: Confusion matrix and accuracy for logistic regression
cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Confusion Matrix:")
print(cm_log)
print(f"Logistic Regression Accuracy: {acc_log:.4f}")

# Step 9: Train and evaluate k-nearest neighbors model (k=3)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Confusion matrix and accuracy for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbors Confusion Matrix:")
print(cm_knn)
print(f"K-Nearest Neighbors Accuracy: {acc_knn:.4f}")

# Step 10: Compare the two models
"""
Comparison:
- Logistic Regression achieved an accuracy of {acc_log:.4f}, while KNN achieved {acc_knn:.4f}.
- Logistic Regression is generally preferred for binary classification due to its interpretability.
- KNN may be affected by the curse of dimensionality, making it less effective on high-dimensional data.
- Depending on the results, experimenting with different values of k in KNN might yield better performance.
"""
