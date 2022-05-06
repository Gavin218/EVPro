from Pretreatment import excel_to_matrix
import pandas as pd
import datetime
# labelList = excel_to_matrix("D:/Backup/桌面/EVC/AEKMlabels.xlsx", 0)
# dateList = excel_to_matrix("D:/Backup/桌面/EVC/loadData.xlsx", 2)
# y = dateList[0]
# t = 3
data2 = pd.read_pickle("D:/Backup/桌面/EVC/startAndChargeTime")
x = data2[19][0]
originDay = datetime.date(year=1899, month=12, day=30)
y = (x.date() - originDay).days
x = 43810.0
