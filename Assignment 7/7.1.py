import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Step 0: Read data into a pandas dataframe
df = pd.read_csv("data_banknote_authentication.csv")

# Step 1: Define features and target variable
X = df.drop(columns=["class"])
y = df["class"]

# Step 2: Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Step 3: Train SVM with linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Step 4: Predict and evaluate linear kernel model
y_pred_linear = svm_linear.predict(X_test)
print("Confusion Matrix (Linear Kernel):\n", confusion_matrix(y_test, y_pred_linear))
print("Classification Report (Linear Kernel):\n", classification_report(y_test, y_pred_linear))

# Step 5: Train SVM with radial basis function (RBF) kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# Predict and evaluate RBF kernel model
y_pred_rbf = svm_rbf.predict(X_test)
print("Confusion Matrix (RBF Kernel):\n", confusion_matrix(y_test, y_pred_rbf))
print("Classification Report (RBF Kernel):\n", classification_report(y_test, y_pred_rbf))

# Step 6: Comparison
print("Comparison:")
print("- The linear kernel is typically better for linearly separable data.")
print("- The RBF kernel can handle more complex, non-linear decision boundaries.")
print("- Evaluating performance based on accuracy, precision, recall, and F1-score helps in choosing the best model.")
