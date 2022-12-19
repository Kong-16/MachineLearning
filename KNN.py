import numpy as np
import pandas as pd
import math


#두 데이터 간의 거리를 구하는 함수
def get_distance(point1, point2):
    distance = 0
    featureNum = len(point1) - 1 #데이터의 feature 수를 구함, 마지막 원소는 타입이므로 거리구하지 않음. 
    for i in range(featureNum):
        distance += pow(point1[i] - point2[i],2)
    return math.sqrt(distance)

def get_nearest(dataset, testdata, k, classnum):
    distance_arr = []
    for data in dataset:
        tmp_list = [get_distance(testdata,data),data[len(data)-1]]
        distance_arr.append(tmp_list) # 거리, 타입 으로 이루어진 리스트 삽입
    distance_arr.sort(key = lambda x:x[0]) # 거리순으로 정렬
    type_arr = [0 for i in range(classnum)] # type 개수를 담을 리스트
    for i in range(k): # 가장 가까운 k개의 데이터의 타입 개수 측정
        idx = distance_arr[i][1] # distance_arr의 타입 = index
        type_arr[idx] += 1
    big = max(type_arr)
    return type_arr.index(big) # 가장 많은 수가 나온 타입의 인덱스 = 타입. 해당 타입으로 예상.

def get_nearest_np(dataset, testdata, k, classnum):
    distance_arr = np.array([]) 
    for data in dataset:
        tmp_list = np.array([get_distance(testdata,data),data[len(data)-1]])
        np.append(distance_arr,tmp_list) # 거리, 타입 으로 이루어진 리스트 삽입
    distance_arr.sort(key = lambda x:x[0]) # 거리순으로 정렬
    type_arr = [0 for i in range(classnum)] # type 개수를 담을 리스트
    for i in range(k): # 가장 가까운 k개의 데이터의 타입 개수 측정
        idx = distance_arr[i][1] # distance_arr의 타입 = index
        type_arr[idx] += 1
    big = np.max(type_arr)
    return type_arr.index(big) # 가장 많은 수가 나온 타입의 인덱스 = 타입. 해당 타입으로 예상.