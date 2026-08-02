# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load dataset (example: simple dataset)
data = {
    'Age': [25, 30, 45, 35, 22, 40, 28, 50],
    'Salary': [50000, 60000, 80000, 65000, 48000, 75000, 52000, 90000],
    'Purchased': [0, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Step 2: Split features and target
X = df[['Age', 'Salary']]
y = df['Purchased']

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Step 4: Apply Naïve Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Step 7: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
Confusion Matrix:
[[0 1]
 [0 1]]
Accuracy: 0.5
