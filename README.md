# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
NAME:  SANTHABABU  G


REGISTER NUMBER:   212224040292

## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:


1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: 

Import Required Libraries

Import libraries for numerical computation (numpy), data handling (pandas), plotting (matplotlib), and performance metrics from sklearn.

Step 2:

Load Dataset

Read the CSV file student_scores.csv using pandas.read_csv().

x stores the input features (hours studied).

y stores the target variable (exam scores).

Step 3:

Split Data into Training and Testing Sets

Use train_test_split() to split x and y into:

x_train, y_train: for training

x_test, y_test: for evaluation

Use a test size of 1/3 and a fixed random_state for reproducibility.

Step 4:

Train the Linear Regression Model

Create a LinearRegression model instance.

Fit the model to the training data using regressor.fit(x_train, y_train).

Step 5:

Make Predictions

Use regressor.predict() on x_test to get predicted scores y_pred.

Step 6: 

Evaluate the Model

Calculate evaluation metrics:

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Print the values of MSE, MAE, and RMSE.

Step 7: 

Visualize the Results

Training Set Plot:

Scatter plot of actual training data (x_train, y_train).

Regression line based on predictions from training data.

Test Set Plot:

Scatter plot of actual test data (x_test, y_test).

Regression line based on predicted test data (y_pred).

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="green")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


```

## Output:

HEAD:

<img width="203" height="307" alt="image" src="https://github.com/user-attachments/assets/644ad0c7-6217-43bd-8e6c-0d4be6bc9f17" />

TAIL:

<img width="188" height="179" alt="image" src="https://github.com/user-attachments/assets/eb3b01e0-3a6f-401b-8265-f7b6793eed89" />

X VALUE:

<img width="204" height="444" alt="image" src="https://github.com/user-attachments/assets/8c3e761a-e8c1-4d14-9ab5-84a6f4f7cedd" />

Y VALUE:

<img width="611" height="49" alt="image" src="https://github.com/user-attachments/assets/5d46f93b-0705-49e0-8afc-894f3d9192c9" />

Y PREDICT:

<img width="564" height="61" alt="image" src="https://github.com/user-attachments/assets/32ec1b10-788c-40ee-9434-d6458ef7c451" />

Y TEST:

<img width="467" height="41" alt="image" src="https://github.com/user-attachments/assets/1f64b315-8837-4040-a20b-a01baece55b5" />



MSE,MAE,RMSE:


<img width="250" height="69" alt="image" src="https://github.com/user-attachments/assets/d646a1c9-b0a2-4266-9def-6bb010421fc7" />




GRAPH:

<img width="756" height="571" alt="image" src="https://github.com/user-attachments/assets/a92cc075-3be3-4558-9dea-d35880b37a94" />


<img width="740" height="573" alt="image" src="https://github.com/user-attachments/assets/313f8fbd-769e-4169-b2e8-4cf47bfd8690" />

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
