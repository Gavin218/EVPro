from Pretreatment import excel_to_matrix

# dataset = excel_to_matrix("D:/Backup/桌面/EV/minle.xlsx", 4)
# data = dataset[14]
# t = 1
#
# dateList = excel_to_matrix("D:/Backup/桌面/EV/minle.xlsx", 5)

# x = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26]
# i1 = 0
# i2 = i1 + 7
#
#
# x_list = []
# while i2 < len(x):
#     if x[i2] - x[i1] == 7:
#         x_list.append([x[i1:i2], x[i2]])
#     i1 += 1
#     i2 += 1
# t = 0

# x = [5,2,76]
# x2 = [7,27,2]
# x3 = x + x2
# t = 1

from Pretreatment import daysToDayIO
from Pretreatment import daysToDayAndOthersIO
import pandas as pd

input_path = "D:/Backup/桌面/EV/minle.xlsx"
sheet1 = 1
sheet2 = 2
daysNum = 7
output_path = "D:/Backup/桌面/EV/dataConOthers"
# daysToDayIO(inputPath=input_path, sheetData=sheet1, sheetDate=sheet2, daysNum=daysNum, outputPath=output_path)
daysToDayAndOthersIO(inputPath=input_path, sheetData=sheet1, sheetDate=sheet2, daysNum=daysNum, outputPath=output_path)
data = pd.read_pickle(output_path)
data1 = data[2]
print(len(data), len(data1), len(data1[0]), len(data1[1]))
t = 2

