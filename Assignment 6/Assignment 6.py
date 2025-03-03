import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("bank.csv", delimiter=';')


selected_columns = ['y', 'job', 'marital', 'default', 'housing', 'poutcome']
data = df[selected_columns]

data = pd.get_dummies(data, drop_first=True)

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

X = data.drop(columns=['y_yes'])
y = data['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Evaluate Logistic Regression model
log_conf_matrix = confusion_matrix(y_test, y_pred_log)
log_accuracy = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Confusion Matrix:")
print(log_conf_matrix)
print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")

# Train k-Nearest Neighbors model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluate KNN model
knn_conf_matrix = confusion_matrix(y_test, y_pred_knn)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\nK-Nearest Neighbors Confusion Matrix:")
print(knn_conf_matrix)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
