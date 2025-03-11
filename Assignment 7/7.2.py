import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
file_path = "suv.csv"
df = pd.read_csv(file_path)

# Select features and target variable
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree with Entropy criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train_scaled, y_train)
y_pred_entropy = dt_entropy.predict(X_test_scaled)

# Evaluate Entropy model
print("Decision Tree with Entropy Criterion:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report:\n", classification_report(y_test, y_pred_entropy))

# Train Decision Tree with Gini criterion
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train_scaled, y_train)
y_pred_gini = dt_gini.predict(X_test_scaled)

# Evaluate Gini model
print("\nDecision Tree with Gini Criterion:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
print("Classification Report:\n", classification_report(y_test, y_pred_gini))
