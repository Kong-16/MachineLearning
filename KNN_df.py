import numpy as np
import pandas as pd
import math


#두 데이터 간의 거리를 구하는 함수
def get_distance(point1, point2): # (pd.series, pd.series)
    distance = pow(point1.sub(point2,axis="index"),2)
    distance = math.sqrt(distance.sum())
    return distance

def get_nearest(testdata, dataset, type, k): # (pd.series, pd.df, class, num of K) 
    df = dataset.apply(get_distance,axis=1,args=(testdata,)) #apply로 모든 dataset의 row에 대해 get_distance 실행
    distance = pd.DataFrame([], columns=["distance","class"]) #거리와 class로 이루어진 dataframe 생성
    distance["distance"] = df 
    distance["class"] = type 
    distance = distance.sort_values(by="distance") #거리 순으로 정렬
    distance = distance[:k] # k개 만큼 슬라이싱
    freq = distance["class"].value_counts() 
    return freq[:1].index[0]