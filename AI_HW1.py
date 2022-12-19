import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MachineLearning.KNN as KNN
from sklearn.datasets import load_iris

iris = load_iris()

iris_type = {0 : "setosa", 1 : "versicolor", 2 : "virginica"}
X = iris.data
y = iris.target

k = 10

print(X)
print(y)
testset = np.array([]) # test를 위한 데이터 빼서 저장


loop = len(y)
i = 0
while i < loop: #매 15번째 데이터 testset에 넣은 후 dataset에서 삭제
    if i % 14 == 0:
        np.append(X[i],y[i])
        np.append(testset,X[i])
        print(testset)
        loop -= 1
    i += 1
         
for i in range(len(X)): # data의 feature 마지막에 type 삽입
    X[i].append(y[i])

correct = 0
wrong = 0
for i in range(len(testset)):
    predict = KNN.get_nearest(X,testset[i],k)
    real = testset[i][4]
    print("{} Computed class: {}, True class: {}".format(i,iris_type[predict],iris_type[real]))
    if(predict == real):
        correct += 1
    else:
        wrong += 1

accuracy = correct / len(testset)
print("accuracy : {}%".format(accuracy * 100))
