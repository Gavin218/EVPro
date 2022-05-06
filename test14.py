
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

from Pretreatment import drawElbowPicture

drawElbowPicture()