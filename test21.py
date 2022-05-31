# from Pretreatment import DTWDistanceIO
# inputPath = "D:/Backup/桌面/EVdata/hf/loadData.xlsx"
# sheet = 0
# outputPath = "D:/Backup/桌面/EVdata/hf/distanceMat.xlsx"
# DTWDistanceIO(inputPath, sheet, outputPath)


# from Pretreatment import distanceMatToPredByKMedoidsIO
# distancePath = "D:/Backup/桌面/EVdata/hf/distanceMat.xlsx"
# outputPath = "D:/Backup/桌面/EVdata/hf/label/"
# for k in range(2, 11):
#     distanceMatToPredByKMedoidsIO(distancePath, 0, k, outputPath + "cluster{i}.xlsx".format(i=k))



# from Pretreatment import drawOnePictureToShowDifferentClass3
# from Pretreatment import drawPicturesAndSaveByDifferentClass3
# inputPath = "D:/Backup/桌面/EVdata/gm/loadData.xlsx"
# sheet = 0
# labelPath = "D:/Backup/桌面/EVdata/gm/label/cluster7.xlsx"
# k = 7
# outputPath1 = "D:/Backup/桌面/EVdata/hf/pic/"
# outputPath2 = "D:/Backup/桌面/EVdata/hf/total"
# drawPicturesAndSaveByDifferentClass3(inputPath, sheet, labelPath, k, outputPath1)
# drawOnePictureToShowDifferentClass3(inputPath, sheet, labelPath, outputPath2)

# from Pretreatment import drawOnePictureToShowDifferentClass2
# from Pretreatment import drawPicturesAndSaveByDifferentClass2
#
# inputPath = "D:/Backup/桌面/EVdata/hf/loadData.xlsx"
# sheet = 0
#
#
# for k in range(2, 11):
#     labelPath = "D:/Backup/桌面/EVdata/hf/label/cluster{i}.xlsx".format(i=k)
#     outputPath1 = "D:/Backup/桌面/EVdata/hf/label/{i}/".format(i=k)
#     outputPath2 = "D:/Backup/桌面/EVdata/hf/label/cluster{i}".format(i=k)
#     drawPicturesAndSaveByDifferentClass2(inputPath, sheet, labelPath, k, outputPath1)
#     drawOnePictureToShowDifferentClass2(inputPath, sheet, labelPath, outputPath2)

# from Pretreatment import readCsvAndOutput
# inputPath = "D:/Backup/桌面/EVdata/gm/shenzhenweather.csv"
# datePath = "D:/Backup/桌面/EVdata/hf/loadData.xlsx"
# sheet = 1
# outputPath = "D:/Backup/桌面/EVdata/hf/needDate.xlsx"
# readCsvAndOutput(inputPath, datePath, sheet, outputPath)


# from Pretreatment import showStatistics
# inputPath = "D:/Backup/桌面/EVdata/hf/correlation.xlsx"
# outputPath = "D:/Backup/桌面/EVdata/hf/detailInfomation.xlsx"
# sheet = 1
# showStatistics(inputPath, sheet, outputPath)

# from Pretreatment import csvCleanIO2
# csvCleanIO2('D:/Backup/桌面/EVdata/hf/hf_order.csv', 'D:/Backup/桌面/EVdata/hf/order2.xlsx')


# from Pretreatment import orderToDataframeIO2
# inputPath = 'D:/Backup/桌面/EVdata/hf/order2.xlsx'
# sheet = 0
# outputPath = 'D:/Backup/桌面/EVdata/hf/startTimeAndChargingOrder2'
# orderToDataframeIO2(inputPath, sheet, outputPath)

# from Pretreatment import clusterOrderDataByLabels2
# datePath = 'D:/Backup/桌面/EVdata/gm/loadData.xlsx'
# dateSheet = 1
# labelPath = 'D:/Backup/桌面/EVdata/gm/label/cluster7.xlsx'
# labelSheet = 0
# k = 7
# orderDataPath = 'D:/Backup/桌面/EVdata/gm/startTimeAndChargingOrder2'
# outputPath = 'D:/Backup/桌面/EVdata/gm/startTimeAndChargingOrderByCluster7_2'
# clusterOrderDataByLabels2(datePath, dateSheet, labelPath, labelSheet, k, orderDataPath, outputPath)

# from Pretreatment import orderOpticsAndDrawScatter
# orderPath = 'D:/Backup/桌面/EVdata/hf/startTimeAndChargingOrderByCluster6'
# sheet = 1
# minSample = 36
# orderOpticsAndDrawScatter(orderPath, sheet, minSample)


# X = [[1,2],[2,1]]
# from sklearn.cluster import OPTICS
# clustering = OPTICS(min_samples=1).fit(X)
# labels = clustering.labels_
# y = max(labels)
# print(y)




