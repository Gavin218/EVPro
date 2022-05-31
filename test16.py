# import pandas as pd
# import datetime
inputPath = "D:/Backup/桌面/EVK/gmsouth/深圳市“新冠肺炎”-每日确诊病例统计_2920001503670.csv"
existPath = "D:/Backup/桌面/EVK/hf/loadData.xlsx"
sheet = 1
outputPath = "D:/Backup/桌面/EVK/hf/patient.xlsx"
# dataset = pd.read_csv(input_file)
# originDay = datetime.date(year=1899, month=12, day=30)
# t = dataset["jzrq"]
# tt = t[8]
# tt = "1958年8月24日"
# y = datetime.datetime.strptime(tt, "%Y年%m月%d日")
# nowday = originDay + datetime.timedelta(days=21421)
# date22 = y.date()
# t = date22 == nowday
# tttt = 2

from Pretreatment import readCsvToExistDateDataIO
readCsvToExistDateDataIO(inputPath, existPath, sheet, outputPath)