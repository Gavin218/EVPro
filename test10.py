import pandas as pd
import datetime
import pickle
import numpy as np
import math

dataset = pd.read_csv("D:/Backup/桌面/EVC/Building_CS_Order.csv")
startTimeList = dataset["充电开始时间"]
chargingTimeList = dataset["充电时长（分）"]
dataList = []
for i in range(len(dataset)):
    startTime = startTimeList[i]
    chargingTime = chargingTimeList[i]
    if type(startTime) == str:
        dataList.append([datetime.datetime.strptime(startTime, "%Y/%m/%d %H:%M"), chargingTime])
print(len(dataList))
with open("D:/Backup/桌面/EVC/startAndChargeTime", 'wb') as f:
    pickle.dump(dataList, f)
y = pd.read_pickle("D:/Backup/桌面/EVC/startAndChargeTime")
y1 = y[8]
y11 = y1[0]
y12 = y1[1]

