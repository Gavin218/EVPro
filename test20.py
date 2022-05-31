from Pretreatment import excel_to_matrix
import datetime
# dataset = excel_to_matrix("D:/Backup/桌面/EVK/gmsouth/loadData.xlsx", 1)
# x = dataset[0][0]
# t = 3
# from Pretreatment import orderCsvToDataframeIO
# orderCsvToDataframeIO("D:/Backup/桌面/EVK/gmsouth/orderData.xlsx", 0, "D:/Backup/桌面/EVK/gmsouth/timeOforderData")
datePath = "D:/Backup/桌面/EVK/gmsouth/loadData.xlsx"
dateSheet = 1
labelPath = "D:/Backup/桌面/EVK/gmsouth/AEKMlabel2.xlsx"
labelSheet = 0
k = 3
orderDataPath = "D:/Backup/桌面/EVK/gmsouth/timeOforderData"
outputPath = "D:/Backup/桌面/EVK/gmsouth/orderDataByCluster2"

from Pretreatment import clusterOrderDataByLabels
# clusterOrderDataByLabels(datePath, dateSheet, labelPath, labelSheet, k, orderDataPath, outputPath)

# from Pretreatment import orderKmeansAndDrawElbow
# orderKmeansAndDrawElbow(outputPath, 2, 1, 12)
# import matplotlib.pyplot as plt
# x = [1, 2, 4]
# y = [2, 3, 1]
# plt.scatter(x, y, color="red", s=10)
# plt.show()


from Pretreatment import orderKmeansAndDrawScatter
orderKmeansAndDrawScatter(orderPath=outputPath, sheet=0, k=3)


