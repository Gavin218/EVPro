from Pretreatment import excel_to_matrix
from Pretreatment import writeToExcel

input_path = "D:/Backup/桌面/EV/minle.xlsx"
existDate = excel_to_matrix(input_path, 3)
xinguan = excel_to_matrix(input_path, 7)
outData = []
dict = {}
for data in xinguan:
    dict[data[0]] = data[1]
for data in existDate:
    if dict.get(data[0]) is None:
        outData.append([data[0], 0])
    else:
        outData.append([data[0], dict[data[0]]])
writeToExcel("D:/Backup/桌面/EV/xinguandata.xlsx", outData)

# dict = {1:3, 2:8, 98:1}
# print(dict[1])
# print(dict[98])
# print(dict.get(1))
# print(dict.get(3))