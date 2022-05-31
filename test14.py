
from Pretreatment import onlyKeepMinutes
import pandas as pd
# k1 = 1
# k2 = 10
#
# x = pd.read_pickle("D:/Backup/桌面/EVC/startAndChargeTimeByCluster")
# xx = x[0][0][0]
# y = xx.time()
# y1 = y.minute
# print("Hello world")
# onlyKeepMinutes("D:/Backup/桌面/EVC/startAndChargeTimeByCluster", "D:/Backup/桌面/EVC/startAndChargeTimeForDrawing")
# data = pd.read_pickle("D:/Backup/桌面/EVC/startAndChargeTimeForDrawing")
# data1 = data[0]
# data11 = data1[0]
# t = 2

# from Pretreatment import drawElbowPicture
# import pandas as pd
#
# dataset = pd.read_pickle("D:/Backup/桌面/EVK/gmsouth/lowDemData2")
# k_min = 1
# k_max = 15
# drawElbowPicture(dataset, k_min, k_max)


inputPath = "D:/Backup/桌面/EVK/hf/lowDemData"
k = 3
labelPath = "D:/Backup/桌面/EVK/hf/AEKMlabel.xlsx"
sheet = 0
# from Pretreatment import kMeansAndSaveLabelsIO
# kMeansAndSaveLabelsIO(inputPath, k, labelPath)
outputPath = "D:/Backup/桌面/EVK/hf/pic3/"
# #
from Pretreatment import drawPicturesAndSaveByDifferentClass
from Pretreatment import drawOnePictureToShowDifferentClass
# # #
inputPath = "D:/Backup/桌面/EVK/hf/loadData.xlsx"
drawPicturesAndSaveByDifferentClass(inputPath, sheet, labelPath, k, outputPath)
drawOnePictureToShowDifferentClass(inputPath, sheet, labelPath)
