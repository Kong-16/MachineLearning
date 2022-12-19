import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import warnings
warnings.filterwarnings("ignore")

# 1. Load Train and Test dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Feature Engineering
# 1) Numeric feature
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'hasgarage', 'hasbsmt',
                 'hasfireplace']:  # 이거는 사실상 Categorical Feature
            pass
        else:
            numeric.append(i)

# Id 제거
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Remove Outliers 1
train.drop(train[(train['OverallQual'] < 5) & (train['SalePrice'] > 200000)].index, inplace=True)
train.drop(train[(train['GrLivArea'] > 4100) & (train['SalePrice'] < 300000)].index, inplace=True)
train.drop(train[(train['TotalBsmtSF'] > 4000) & (train['SalePrice'] < 300000)].index, inplace=True)
train.drop(train[(train['LotArea'] > 110000) & (train['SalePrice'] < 500000)].index, inplace=True)
train.drop(train[(train['LotFrontage'] > 300) & (train['SalePrice'] < 300000)].index, inplace=True)

# Split features and labels
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# concat train and test features in order to apply the feature transformations to entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Handling missing values
def fill_missing(features):
    for col in ('GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if i in ['GarageYrBlt', 'YrSold']:
            continue
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features['GarageYrBlt'].fillna(features['GarageYrBlt'].median()))
    features.update(features[numeric].fillna(0))
    return features


all_features = fill_missing(all_features)

# Fix skewed features
# Fetch all numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if i in ['MSSubClass', 'OverallQual', 'OverallCond']:
        pass
    elif all_features[i].dtype in numeric_dtypes:
        numeric.append(i)

skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))

skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index


def log_scaling(res, ls):
    m = res.shape[1]
    for l in ls:
        res[l] = np.log(1.01 + res[l])
    return res


log_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

all_features = log_scaling(all_features, log_features)

# 2) Categorial Feature
train_categorical = all_features.select_dtypes(include='object')  # categorical dtype 분류
train_categorical['MoSold'] = all_features['MoSold'].astype(str)
train_categorical['YrSold'] = all_features['YrSold'].astype(str)
train_categorical['MSSubClass'] = all_features['MSSubClass'].astype(str)
train_categorical['OverallQual'] = all_features['OverallQual'].astype(str)
train_categorical['OverallCond'] = all_features['OverallCond'].astype(str)  # Numerical Feature으로 인식되지만 사실상 Categorial Feature로 작동하는 5가지 feature 추가

for col in train_categorical.columns:
    del all_features[col]

# 3) 데이터 정규화 - numerical features에 대해서만 정규화 진행하기 위해 이곳에 배치
nor_all_features = all_features
for col in all_features.columns:
    nor_all_features[col] = (all_features[col] - all_features[col].min()) / (all_features[col].max() - all_features[col].min())

# Categofical Feature 삭제 및 변환
# 학습에 적합한 Features
categorical_strong_features = ['MSZoning', 'Street', 'Neighborhood', 'Condition1', 'HouseStyle', 'Exterior1st',
                               'Exterior2nd', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond',
                               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual',
                               'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition',
                               'OverallQual', 'OverallCond', 'MSSubClass']

for column in train_categorical.columns:
    if column not in categorical_strong_features:
        del train_categorical[column]

# 결측치 채우기
for columns in train_categorical.columns:
    train_categorical[columns].fillna('None', inplace=True)

# 회귀모델이 학습할 수 있도록 숫자형태로 변환
train_categorical_enc = pd.get_dummies(train_categorical, dummy_na=True, drop_first=True)

nor_all_features = pd.concat([nor_all_features, train_categorical_enc], axis=1) # normalize된 데이터

ntrain = train.shape[0]
nor_X_train = nor_all_features[:ntrain].values
y_train = np.log1p(train_labels.values)
nor_X_test = nor_all_features[ntrain:].values

# 3. 학습
# 1) 직접 구현한 회귀 모델 사용
from regression import Lasso, Ridge, Elasticnet, predict

epochs = 10000
learning_rate = 0.005
ls_penalty = 0.9
rd_penalty = 0.1

# Lasso
my_ls = Lasso(epochs, learning_rate, ls_penalty)
my_ls.fit(nor_X_train, y_train)
my_ls_pred = predict(nor_X_test, my_ls.theta, my_ls.b)
my_ls_pred = np.expm1(my_ls_pred)

y_Id = pd.read_csv('test.csv')['Id']
my_ls_pred = pd.DataFrame(my_ls_pred, columns = ['SalePrice'])
submit = pd.concat([y_Id, my_ls_pred], axis = 1)
submit.to_csv('my_lasso.csv', index=False)


# Ridge
my_rd = Ridge(epochs, learning_rate, rd_penalty)
my_rd.fit(nor_X_train, y_train)
my_rd_pred = predict(nor_X_test,my_rd.theta, my_rd.b)
my_rd_pred = np.expm1(my_rd_pred)

SalePrice = pd.DataFrame(my_rd_pred)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('my_Ridge.csv', index=False)


# Elastic net
my_en = Elasticnet(epochs, learning_rate, ls_penalty, rd_penalty)
my_en.fit(nor_X_train, y_train)
my_en_pred = predict(nor_X_test, my_en.theta, my_en.b)
my_en_pred = np.expm1(my_en_pred)

SalePrice = pd.DataFrame(my_en_pred)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('my_Elasticnet.csv', index=False)

# 2) 사이킷런 회귀 모델
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

# 각각의 모델을 통해 학습했을 때 결과
# Lasso
start = time.time()
ls = Lasso()
ls.fit(nor_X_train, y_train)
ls_pred = ls.predict(nor_X_test).reshape(-1, 1)
ls_pred = np.expm1(ls_pred)
print("사이킷런 Ridge의 실행 시간: ", time.time() - start)

SalePrice = pd.DataFrame(ls_pred)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('Lasso.csv', index=False)

# Ridge
start = time.time()
rd = Ridge()
rd.fit(nor_X_train, y_train)
rd_pred = rd.predict(nor_X_test).reshape(-1, 1)
rd_pred = np.expm1(rd_pred)
print("사이킷런 Ridge의 실행 시간: ", time.time() - start)

SalePrice = pd.DataFrame(rd_pred)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('Ridge.csv', index=False)

# Elastic Net
start = time.time()
en = ElasticNet()
en.fit(nor_X_train, y_train)
en_pred = en.predict(nor_X_test).reshape(-1, 1)
en_pred = np.expm1(en_pred)
print("사이킷런 Ridge의 실행 시간: ", time.time() - start)

SalePrice = pd.DataFrame(en_pred)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('ElasticNet.csv', index=False)

# 3) 교차검증 및 앙상블 후 결과
def k_fold(model):
    k = 5

    num = nor_X_train.shape[0] // k
    fold_train_predict = np.zeros((nor_X_train.shape[0], 1))
    fold_test_predict = np.zeros((nor_X_test.shape[0], k))

    for i in range(k):
        fold_X_test = nor_X_train[i * num: (i + 1) * num]

        fold_X_train = np.concatenate([nor_X_train[:i * num], nor_X_train[(i + 1) * num:]], axis=0)
        fold_y_train = np.concatenate([y_train[:i * num], y_train[(i + 1) * num:]], axis=0)

        model.fit(fold_X_train, fold_y_train)

        if model == my_ls:
            fold_train_predict[i * num: (i + 1) * num, :] = predict(fold_X_test, my_ls.theta, my_ls.b).reshape(-1, 1)
            fold_test_predict[:, i] = predict(nor_X_test, my_ls.theta, my_ls.b)[:, 0]
        elif model == my_rd:
            fold_train_predict[i * num: (i + 1) * num, :] = predict(fold_X_test, my_rd.theta, my_rd.b).reshape(-1, 1)
            fold_test_predict[:, i] = predict(nor_X_test, my_rd.theta, my_rd.b)[:, 0]
        elif model == my_en:
            fold_train_predict[i * num: (i + 1) * num, :] = predict(fold_X_test, my_en.theta, my_en.b).reshape(-1, 1)
            fold_test_predict[:, i] = predict(nor_X_test, my_en.theta, my_en.b)[:, 0]
        else:
            fold_train_predict[i * num: (i + 1) * num, :] = model.predict(fold_X_test).reshape(-1, 1)
            fold_test_predict[:, i] = model.predict(nor_X_test)

    test_predict_mean = np.mean(fold_test_predict, axis=1).reshape(-1, 1)

    return fold_train_predict, test_predict_mean

# 앙상블에 사용할 회귀 모델
svr = SVR()
rf = RandomForestRegressor()
xgb = XGBRegressor()
gbm = LGBMRegressor()

my_ls_train, my_ls_test = k_fold(my_ls)
my_rd_train, my_rd_test = k_fold(my_rd)
my_en_train, my_en_test = k_fold(my_en)
ls_train, ls_test = k_fold(ls)
rd_train, rd_test = k_fold(rd)
en_train, en_test = k_fold(en)
svr_train, svr_test = k_fold(svr)
rf_train, rf_test = k_fold(rf)
xgb_train, xgb_test = k_fold(xgb)
gbm_train, gbm_test = k_fold(gbm)

my_new_X_train = np.concatenate([my_rd_train, my_ls_train, my_en_train, svr_train, rf_train, xgb_train, gbm_train], axis=1)
my_new_X_test = np.concatenate([my_rd_test, my_ls_test, my_en_test, svr_test, rf_test, xgb_test, gbm_test], axis=1)
new_X_train = np.concatenate([rd_train, ls_train, en_train, svr_train, rf_train, xgb_train, gbm_train], axis=1)
new_X_test = np.concatenate([rd_test, ls_test, en_test, svr_test, rf_test, xgb_test, gbm_test], axis=1)

print(my_new_X_train)
print(new_X_train)
ls.fit(my_new_X_train, y_train)
ls.fit(new_X_train, y_train)
my_ensemble_predict = ls.predict(my_new_X_test)
ensemble_predict = ls.predict(new_X_test)
my_ensemble_predict = np.expm1(my_ensemble_predict)
ensemble_predict = np.expm1(ensemble_predict)

SalePrice = pd.DataFrame(my_ensemble_predict)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('my_ensemble.csv', index=False)

SalePrice = pd.DataFrame(ensemble_predict)
submit = pd.concat([y_Id, SalePrice], axis = 1)
submit.columns.values[1] = 'SalePrice'
submit.to_csv('ensemble.csv', index=False)

