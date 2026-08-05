# Iris Flower Classification using Naive Bayes

# Step 1: Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Iris dataset
iris = load_iris()

# Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
X = iris.data

# Target (Flower Species)
y = iris.target

# Step 3: Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 4: Create Naive Bayes Classifier
model = GaussianNB()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict the test data
y_pred = model.predict(X_test)

# Step 7: Display Results
print("Actual Classes:")
print(y_test)

print("\nPredicted Classes:")
print(y_pred)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Step 9: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 10: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 11: Predict a New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nSample Flower:", sample)
print("Predicted Species:", iris.target_names[prediction][0])
