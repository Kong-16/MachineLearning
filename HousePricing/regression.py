import numpy as np
import random

def predict(X, theta, b):
    return np.dot(X, theta) + b

class Lasso():
    def __init__(self, epochs, learning_rate, penalty):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.penalty = penalty

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        data_num, feature_num = X.shape
        # initialize weights & bias
        self.b = 0
        self.theta = np.random.rand(feature_num, 1)
        theta_tmp = self.theta
        # repeat "epochs" time
        for j in range(self.epochs):
            # X = n(data_num) * m(feature_num) matrix
            # theta = m * 1 matrix
            # err = m * 1 matrix
            pred = predict(X, self.theta, self.b)
            err = y - pred
            for i in range(feature_num):
                # lasso penalty = abs(penalty * theta)
                if self.theta[i] > 0:
                    theta_tmp[i] = ((2 * X[:, i].T.dot(err)) + self.penalty) / data_num
                elif self.theta[i] < 0:
                    theta_tmp[i] = ((2 * X[:, i].T.dot(err)) - self.penalty) / data_num
                else:
                    theta_tmp[i] = ((2 * X[:, i].T.dot(err)) + random.uniform(-self.penalty,self.penalty)) / data_num
            b_tmp = -2 * np.sum(err) / data_num
            self.theta = self.theta - (self.learning_rate * theta_tmp)
            self.b = self.b - (self.learning_rate * b_tmp)
        return self

class Ridge():
    def __init__(self, epochs, learning_rate, penalty):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.penalty = penalty
    def fit(self, X, y):
        y = y.reshape(-1, 1)
        data_num, feature_num = X.shape
        # initialize weights
        self.b = 0
        self.theta = np.random.rand(feature_num, 1)
        theta_tmp = self.theta
        # repeat "epochs" time
        for j in range(self.epochs):
            # X = n(data_num) * m(feature_num) matrix
            # theta = m * 1 matrix
            # err = m * 1 matrix
            pred = predict(X, self.theta, self.b)
            err = y - pred
            # Ridge penalty = penalty * (theta ^ 2)
            theta_tmp = (-(2 * X.T.dot(err)) + (2 * self.penalty * self.theta)) / data_num
            #theta_tmp = np.linalg.inv(X.T.dot(X) + (np.identity(feature_num) * self.penalty)).dot(X.T.dot(y))
            b_tmp = -2 * np.sum(err) / data_num
            self.theta = self.theta - (self.learning_rate * theta_tmp)
            self.b = self.b - (self.learning_rate * b_tmp)
        return self

class Elasticnet():
    def __init__(self, epochs, learning_rate, lasso_penalty, ridge_penalty):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lasso_penalty = lasso_penalty
        self.ridge_penalty = ridge_penalty
    def fit(self, X, y):
        y = y.reshape(-1, 1)
        data_num, feature_num = X.shape
        # initialize weights
        self.b = 0
        self.theta = np.random.rand(feature_num, 1)
        theta_tmp = self.theta
        # repeat "epochs" time
        for j in range(self.epochs):
            # X = n(data_num) * m(feature_num) matrix
            # theta = m * 1 matrix
            # err = m * 1 matrix
            pred = predict(X, self.theta, self.b)
            err = y - pred
            # lasso + ridge
            for i in range(feature_num):
                # lasso penalty = abs(penalty * theta)
                if self.theta[i] > 0:
                    theta_tmp[i] = (-(2 * X[:, i].T.dot(err)) +
                                      self.lasso_penalty + (2 * self.ridge_penalty * self.theta[i])) / data_num
                elif self.theta[i] < 0:
                    theta_tmp[i] = (-(2 * X[:, i].T.dot(err)) -
                                      self.lasso_penalty + (2 * self.ridge_penalty * self.theta[i])) / data_num
                else:
                    theta_tmp[i] = (-(2 * X[:, i].T.dot(err)) +
                                      random.uniform(-self.lasso_penalty,self.lasso_penalty) + (2 * self.ridge_penalty * self.theta[i])) / data_num
            b_tmp = -2 * np.sum(err) / data_num
            self.theta = self.theta - (self.learning_rate * theta_tmp)
            self.b = self.b - (self.learning_rate * b_tmp)
        return self