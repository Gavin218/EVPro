import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import pandas as pd


# dataPath = "D:/Backup/桌面/EVC/lowDemData"
#
# dataset = pd.read_pickle(dataPath)
# '利用SSE选择k'
# SSE = []  # 存放每次结果的误差平方和
#
# for k in range(4, 15):
#     # km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
#     km = KMeans(n_clusters=k).fit(dataset)
#     SSE.append(km.inertia_)  # estimator.inertia_获取聚类准则的总和
# X = range(4, 15)
# print(SSE)
# plt.xlabel('k')
# plt.ylabel('SSE')
# plt.plot(X, SSE, 'o-')
# plt.show()

# from Pretreatment import kMeansAndSaveLabelsIO
# kMeansAndSaveLabelsIO("D:/Backup/桌面/EVC/lowDemData", 6, "D:/Backup/桌面/EVC/AEKMlabels.xlsx")

from Pretreatment import drawPicturesAndSaveByDifferentClass
drawPicturesAndSaveByDifferentClass("D:/Backup/桌面/EVC/loadData.xlsx", 1, "D:/Backup/桌面/EVC/AEKMlabels.xlsx", 6, "D:/Backup/桌面/EVC/pic3/")