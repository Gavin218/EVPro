from Pretreatment import clusterDatasetByLabel
import pandas as pd
inputPath = "D:/Backup/桌面/EVC/startAndChargeTime"
labelPath = "D:/Backup/桌面/EVC/AEKMlabels.xlsx"
labelSheet = 0
datePath = "D:/Backup/桌面/EVC/loadData.xlsx"
dateSheet = 2
k = 6
outputPath = "D:/Backup/桌面/EVC/startAndChargeTimeByCluster"
# clusterDatasetByLabel(inputPath, labelPath, labelSheet, datePath, dateSheet, k, outputPath)
ori_data = pd.read_pickle("D:/Backup/桌面/EVC/startAndChargeTime")
new_data = pd.read_pickle("D:/Backup/桌面/EVC/startAndChargeTimeByCluster")
allT = 0
for one in new_data:
    allT += len(one)
    print(len(one))
print(allT)
print(len(ori_data))