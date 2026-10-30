# Implementation-of-Logistic-Regression-Using-Gradient-Descent
## Name: S Madhumitha
## Reg No:212225040217
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required Python libraries and load the `Placement_Data.csv` dataset into Google Colab.

2. Remove unnecessary columns, convert categorical data into numerical values, and separate input features and target variables.

3. Split the dataset into training and testing data, apply feature scaling, and train the Logistic Regression model using Gradient Descent.

4. Predict the student placement status and evaluate the model using accuracy score and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Madhumitha
RegisterNumber:212225040217
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('/content/Placement_Data (1).csv')

data = data.drop(['sl_no', 'salary'], axis=1)

data = pd.get_dummies(data, drop_first=True, dtype=int)

X = data.drop('status_Placed', axis=1).values
y = data['status_Placed'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m = np.zeros(X_train.shape[1])
c = 0
learning_rate = 0.05
epochs = 10000
n = len(y_train)

for i in range(epochs):

    z = np.dot(X_train, m) + c
    y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    dm = np.dot(X_train.T, (y_pred - y_train)) / n
    dc = np.sum(y_pred - y_train) / n

    m = m - learning_rate * dm
    c = c - learning_rate * dc

z = np.dot(X_test, m) + c
y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

y_pred = (y_pred >= 0.5).astype(int)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

if y_pred[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed") 
*/
```

## Output:
<img width="1920" height="1020" alt="Screenshot 2026-10-30 125346" src="https://github.com/user-attachments/assets/d1d25fe8-d3b3-4e8f-a991-ff56a3251582" />
<img width="1920" height="1020" alt="Screenshot 2026-10-30 125355" src="https://github.com/user-attachments/assets/b4562e44-cbd0-498e-9e6d-98857ad20062" />
<img width="1920" height="1020" alt="Screenshot 2026-10-30 125404" src="https://github.com/user-attachments/assets/d3301578-4c0e-4741-b6c9-223e658672a3" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

