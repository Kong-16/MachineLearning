import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import warnings

warnings.filterwarnings("ignore")

# Load Train and Test dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# train 데이터에만 있는 가격 정보 제거
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)

# Id 제거
all_features = pd.concat([train_features, test]).reset_index(drop=True)
del all_features['Id']

# 넘파이 연산을 위해 'object' 타입 변수들 One-hot Encoding
train_categorical = all_features.select_dtypes(include='object')

for col in train_categorical.columns:
    del all_features[col]

train_categorical = pd.get_dummies(train_categorical, dummy_na=True, drop_first=True)

all_features = pd.concat([all_features, train_categorical], axis=1)  # normalize된 데이터

# Handling missing values
all_features.fillna(0, inplace=True)

ntrain = train.shape[0]
X_train = all_features[:ntrain].values
y_train = train_labels.values
X_test = all_features[ntrain:].values


# 오차 계산
import math

y_test = pd.read_csv('sample_submission.csv')
del y_test['Id']
y_test.values

def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)

    result = 0
    l = len(actual)
    for i in range(l):
        result += (actual[i] - predict[i]) ** 2
    result = math.sqrt(result / l)

    return result


# 학습
from regression import Lasso, Ridge, Elasticnet, predict

# 1) 직접 구현한 회귀 모델 사용
epochs = 10000
learning_rate = 0.005
ls_penalty = 0.9
rd_penalty = 0.1

print("직접 구현한 개별 모델의 RMSE")
# (1) Lasso
my_ls = Lasso(epochs, learning_rate, ls_penalty)
my_ls.fit(X_train, y_train)
my_ls_pred = predict(X_test, my_ls.theta, my_ls.b)
print(my_ls_pred)
print("직접 구현한 Lasso의 RSME:", rmse(y_test, my_ls_pred))

# (2) Ridge
my_rd = Ridge(epochs, learning_rate, rd_penalty)
my_rd.fit(X_train, y_train)
my_rd_pred = predict(X_test, my_rd.theta, my_rd.b)
print("직접 구현한 Ridge의 RSME:", rmse(y_test, my_rd_pred))

# (3) Elastic net
my_en = Elasticnet(epochs, learning_rate, ls_penalty, rd_penalty)
my_en.fit(X_train, y_train)
my_en_pred = predict(X_test, my_en.theta, my_en.b)
print("직접 구현한 Elasticnet의 RSME: ", rmse(y_test, my_en_pred))

# 2) 사이킷런 회귀 모델
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("\n사이킷런 개별 모델의 RMSE")
# 각각의 모델을 통해 학습했을 때 결과
ls = Lasso()
ls.fit(X_train, y_train)
ls_pred = ls.predict(X_test).reshape(-1, 1)
print("사이킷런 Lasso의 RSME: ", rmse(y_test, ls_pred))

rd = Ridge()
rd.fit(X_train, y_train)
rd_pred = rd.predict(X_test).reshape(-1, 1)
print("사이킷런 Ridge의 RSME: ", rmse(y_test, rd_pred))

en = ElasticNet()
en.fit(X_train, y_train)
en_pred =en.predict(X_test).reshape(-1, 1)
print("사이킷런 ElasticNet의 RSME: ", rmse(y_test, en_pred))

# 2) 교차검증 및 앙상블 후 결과
def k_fold(model):
    k = 5

    num = X_train.shape[0] // k
    fold_train_predict = np.zeros((X_train.shape[0], 1))
    fold_test_predict = np.zeros((X_test.shape[0], k))

    for i in range(k):
        fold_X_test = X_train[i * num: (i + 1) * num]

        fold_X_train = np.concatenate([X_train[:i * num], X_train[(i + 1) * num:]], axis=0)
        fold_y_train = np.concatenate([y_train[:i * num], y_train[(i + 1) * num:]], axis=0)

        model.fit(fold_X_train, fold_y_train)

        #if model == my_ls:
        #    fold_train_predict[i * num: (i + 1) * num, :] = predict(fold_X_test, my_ls.theta, my_ls.b).reshape(-1, 1)
        #    fold_test_predict[:, i] = predict(X_test, my_ls.theta, my_ls.b)[:, 0]
        #elif model == my_rd:
        #    fold_train_predict[i * num: (i + 1) * num, :] = predict(fold_X_test, my_rd.theta, my_rd.b).reshape(-1, 1)
        #    fold_test_predict[:, i] = predict(X_test, my_rd.theta, my_rd.b)[:, 0]
        #elif model == my_en:
        #    fold_train_predict[i * num: (i + 1) * num, :] = predict(fold_X_test, my_en.theta, my_en.b).reshape(-1, 1)
        #    fold_test_predict[:, i] = predict(X_test, my_en.theta, my_en.b)[:, 0]
        #else:
        fold_train_predict[i * num: (i + 1) * num, :] = model.predict(fold_X_test).reshape(-1, 1)
        fold_test_predict[:, i] = model.predict(X_test)

    test_predict_mean = np.mean(fold_test_predict, axis=1).reshape(-1, 1)

    return fold_train_predict, test_predict_mean


svr = SVR()
rf = RandomForestRegressor()
xgb = XGBRegressor()
gbm = LGBMRegressor()

ls_train, ls_test = k_fold(ls)
rd_train, rd_test = k_fold(rd)
en_train, en_test = k_fold(en)
svr_train, svr_test = k_fold(svr)
rf_train, rf_test = k_fold(rf)
xgb_train, xgb_test = k_fold(xgb)
gbm_train, gbm_test = k_fold(gbm)

new_X_train = np.concatenate([rd_train, ls_train, en_train, svr_train, rf_train, xgb_train, gbm_train], axis=1)
new_X_test = np.concatenate([rd_test, ls_test, en_test, svr_test, rf_test, xgb_test, gbm_test], axis=1)

ls.fit(new_X_train, y_train)
ls_pred = ls.predict(new_X_test)
print("\n스태킹 앙상블 후 RSME: ", rmse(y_test, ls_pred))
