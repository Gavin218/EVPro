from Pretreatment import kMeansTSENPlot

filePath = "D:/Backup/桌面/EVK/hf/lowDemData"
sheet = -1
clusterNum = 3
# kMeansTSENPlot(filePath, sheet, clusterNum)

# from Pretreatment import excel_to_matrix
# import datetime
# originDay = datetime.date(year=1899, month=12, day=30)
# t = excel_to_matrix("D:/Backup/桌面/EVK/loadData.xlsx", 1)[0][0]
#
# nowday = originDay + datetime.timedelta(days=t)
# y = nowday.weekday()
# t = 3

from Pretreatment import dateToWeekIO
dateToWeekIO("D:/Backup/桌面/EVK/hf/loadData.xlsx", 1, "D:/Backup/桌面/EVK/hf/isWeek.xlsx")