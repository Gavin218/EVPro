# from Pretreatment import readXlsxShowClusterNum
# inputPath = 'D:/Backup/桌面/EVdata/hf/label/cluster6.xlsx'
# sheet = 0
# readXlsxShowClusterNum(inputPath, sheet, 6)


# from Pretreatment import drawPicturesAndSaveByDifferentClass2
# inputPath = 'D:/Backup/桌面/EVdata/gm/loadData.xlsx'
# sheet = 0
# labelPath = 'D:/Backup/桌面/EVdata/gm/label/cluster7.xlsx'
# k = 7
# outputPath = 'D:/Backup/桌面/EVdata/gm/label/7/test/'
# drawPicturesAndSaveByDifferentClass2(inputPath, sheet, labelPath, k, outputPath)

# x = [800, 551, 898, 950, 1035, 491]
# y = [57, 80, 63, 62, 58, 39]
# t = []
# for i in range(6):
#     t.append(x[i]/y[i])
# print(t)

import time
# import datetime
# x = 5
# x = datetime.time(minute=(x*15)%60, hour=int(x*15/60))
# y = str(x)[0:5]
# t = 3


# orderPath = 'D:/Backup/桌面/EVdata/hf/startTimeAndChargingOrderByCluster6'
# sheet = 5

import pandas as pd



# orderDataList = pd.read_pickle(orderPath)[sheet]
# dataset = []
# minTime = 10000000
# maxTime = 0
# for orderData in orderDataList:
#     # startTime = orderData[0].hour * 60 + orderData[0].minute
#     chargingTime = orderData[1]
#     # dataset.append([startTime, chargingTime])
#     if chargingTime > maxTime:
#         maxTime = chargingTime
#     if chargingTime < minTime:
#         minTime = chargingTime
# print(minTime, maxTime)


# orderClassPath = 'D:/Backup/桌面/EVdata/gm/startTimeAndChargingOrderByCluster7_2'
# outputPath = 'D:/Backup/桌面/EVdata/gm/statisPrice.xlsx'
# from Pretreatment import statisticsElePriceByClass
# statisticsElePriceByClass(orderClassPath, outputPath)
# from Pretreatment import analyzeDifferentPrices
# import datetime
# dataPath = 'D:/Backup/桌面/EVdata/gm/loadData.xlsx'
# dataSheet = 0
# datePath = dataPath
# dateSheet = 1
# cutDate = datetime.date(year=2022, month=2, day=1)
# analyzeDifferentPrices(dataPath, dataSheet, datePath, dateSheet, cutDate)

# from Pretreatment import excel_to_matrix
# dataset = excel_to_matrix("D:/Backup/桌面/EVC/loadData.xlsx", 0)
# l = []
# for i in range(len(dataset)):
#     for j in range(len(dataset[i])):
#         l.append(dataset[i][j])
# l.sort()
# print(l)

import pandas as pd
# import xlrd
# # 使用pandas库读取csv文件
# dataset = pd.read_csv(".../.../file.csv")
# # 使用xlrd库读取xlsx文件
# data_table = xlrd.open_workbook(".../.../file.xlsx").sheets()[i]
# # 注：由于具体库的版本不同，具体的读取方式可能有所差异

# new_path = 0
#
#
# import openpyxl
# import pickle
# # 将数据存储为xlsx格式文件
# wb = openpyxl.Workbook()
# wb.save(".../.../file.xlsx")
# # 将数据存储为pickle格式文件
# pickle.dump(new_path, open(".../.../output_path"))


# inp = "D:/Backup/桌面/EVdata/gm/gmsouth_load.xlsx"
# outp = "D:/Backup/桌面/EVdata/gm/dtwdis"
# from EV_Functions import DTWDistanceIO
# DTWDistanceIO(inp, outp)
# t = pd.read_pickle(outp)
# tt = 3

import pandas as pd
 # 用于内部手册的范例
# from EV_Functions import distanceMatToPredByKMedoidsIO
# inputPath = "D:/Backup/桌面/EVdata/gm/dtwdis"
# k = 7
# labeloutputPath = "D:/Backup/桌面/EVdata/gm/labelList"
# medoidoutputPath = "D:/Backup/桌面/EVdata/gm/medoidsList"
# distanceMatToPredByKMedoidsIO(inputPath, k, labeloutputPath, medoidoutputPath)


inputPath = "D:/Backup/桌面/EVdata/gm/gmsouth_load.xlsx"
labelPath = "D:/Backup/桌面/EVdata/gm/labelList"
medoidPath = "D:/Backup/桌面/EVdata/gm/medoidsList"
k = 7
outputPath = "D:/Backup/桌面/testPic/"
# from EV_Functions import drawPicturesAndSaveByDifferentClass
# drawPicturesAndSaveByDifferentClass(inputPath, labelPath, medoidPath, k, outputPath)
featureOutputPath = "D:/Backup/桌面/EVdata/gm/featureOfgm.xlsx"
from EV_Functions import tsfelForTypicalDays
tsfelForTypicalDays(inputPath,  medoidPath, featureOutputPath)
# import pandas as pd
# import datetime
# inputPath = "D:/Backup/桌面/EVdata/hf/hf_load-场站5.xlsx"
# dateset = pd.read_excel(inputPath).values[:, 0]
# y = dateset[4]
# x = y.date()
# t = 3


# from EV_Functions import labelToLists
# loadDataPath = "D:/Backup/桌面/EVdata/gm/gmsouth_load.xlsx"
# daysLabelPath = "D:/Backup/桌面/EVdata/gm/labelList"
# k_days = 7
# outputPath = "D:/Backup/桌面/EVdata/gm/datelabelList"
# labelToLists(loadDataPath, daysLabelPath, k_days, outputPath)