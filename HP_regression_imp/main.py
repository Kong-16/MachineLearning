import numpy as np
import pandas as pd
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import warnings
from regression import Ridge, Lasso, Elasticnet
warnings.filterwarnings("ignore")

# Load Train and Test dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Feature Engineering
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


# 결측치 채우기
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

# fix skewed features
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

'''
# 결측치 채우기
for columns in train_categorical.columns:
    all_features[columns].fillna('None', inplace=True)

# 학습에 사용할 수 있도록 값 조정
for col in all_features:
    if col in train_categorical:
        del all_features[col]

# Feature 삭제 및 변환
categorical_strong_features = ['MSZoning', 'Street', 'Neighborhood', 'Condition1', 'HouseStyle', 'Exterior1st',
                               'Exterior2nd', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond',
                               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual',
                               'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition',
                               'OverallQual', 'OverallCond', 'MSSubClass']

train_categorical_enc = pd.get_dummies(train_categorical, dummy_na=True, drop_first=True)

all_features = pd.concat([all_features, train_categorical_enc], axis=1)
'''


# 임시 - 너무 많은 데이터를 사용하면 느려져서 numerical value만 가지고 테스트해보려고 추가한 부분
for col in all_features:
    if col in train_categorical:
        del all_features[col]

ntrain = train.shape[0]
X_train = all_features[:ntrain].to_numpy()
y_train = train_labels.to_numpy()
X_test = all_features[ntrain:].to_numpy()

# 학습
from regression import Lasso, predict
import math

# 오차 계산
y_test = pd.read_csv('sample_submission.csv')
del y_test['Id']
y_test = y_test.to_numpy()

def rmse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    result = 0
    l = len(actual)
    for i in range(l):
        result += (actual[i] - predict[i]) ** 2
    result = math.sqrt(result / l)
    
    return result


# 1) 직접 구현한 Lasso
epochs = 1
learning_rate = 0.005
l1_penalty = 100000000
l2_penalty = 5
ls = Lasso(epochs, learning_rate, l1_penalty)
ls.fit(X_train, y_train)
ls_pred = predict(X_test, ls.theta, ls.b)

'''
# 2) 사이킷런 Lasso
from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(X_train, y_train)
ls_pred = ls.predict(X_test).reshape(-1,1)
'''

print(ls_pred)
print(y_test)
print(rmse(y_test, ls_pred))