import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from regression import Lasso, predict
import os
print(os.getcwd())
df = pd.read_csv("salary_data.csv")
scaler = MinMaxScaler()
X = df.iloc[:,:-1].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size= 1/3, random_state= 1)

epochs = 1000
learning_rate = 0.005
l1_penalty = 500

model_Lasso = Lasso(epochs,learning_rate,l1_penalty)
model_Lasso.fit(X_train,Y_train)

Y_pred_Lasso = predict(X_test,model_Lasso.theta,model_Lasso.b)

print(Y_pred_Lasso)
print(Y_test)