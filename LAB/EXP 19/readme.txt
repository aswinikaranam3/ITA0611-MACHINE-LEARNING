# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Create Sample Bank Loan Dataset
data = {
    'Age':[25,35,45,20,35,52,23,40,60,48],
    'Income':[30000,50000,80000,25000,65000,90000,27000,70000,120000,85000],
    'Credit_Score':[650,700,750,600,720,800,620,760,820,780],
    'Employment':['No','Yes','Yes','No','Yes','Yes','No','Yes','Yes','Yes'],
    'Loan_Status':['No','Yes','Yes','No','Yes','Yes','No','Yes','Yes','Yes']
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Step 3: Convert Categorical Data into Numeric
encoder = LabelEncoder()

df['Employment'] = encoder.fit_transform(df['Employment'])
df['Loan_Status'] = encoder.fit_transform(df['Loan_Status'])

# Step 4: Split Features and Target
X = df[['Age','Income','Credit_Score','Employment']]
y = df['Loan_Status']

# Step 5: Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Step 6: Create Naive Bayes Model
model = GaussianNB()

# Step 7: Train the Model
model.fit(X_train, y_train)

# Step 8: Predict Test Data
y_pred = model.predict(X_test)

# Step 9: Results
print("\nActual Loan Status:")
print(y_test.values)

print("\nPredicted Loan Status:")
print(y_pred)

# Step 10: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy*100,2),"%")

# Step 11: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 12: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Predict New Applicant
new_customer = [[30,55000,710,1]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nLoan Prediction: Loan Approved")
else:
    print("\nLoan Prediction: Loan Rejected")
