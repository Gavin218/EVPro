# 该类用于存储数据预处理函数

#写入文件为list格式,也可为array格式
def writeToExcel(file_path, new_list):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Feature'
    for r in range(len(new_list)):
        for c in range(len(new_list[r])):
            ws.cell(r + 1, c + 1).value = new_list[r][c]
    # excel中的行和列是从1开始计数的，所以需要+1
    wb.save(file_path) # 注意，写入后一定要保存
    print("成功写入文件: " + file_path + " !")
    return 1

def excel_to_matrix(path, i):
    import xlrd
    table = xlrd.open_workbook(path).sheets()[i]#获取第i+1个sheet表
    row = table.nrows  # 行数
    # col = table.ncols  # 列数
    datamatrix = []#生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(row):
        rows = table.row_values(x)  # 把list转换为矩阵进行矩阵操作
        rows = [x for x in rows if x != '']
        datamatrix.append(rows) # 按列把数据存进矩阵中
    return datamatrix


# 输入为数据路径和日期路径，将合规的seq2seq写入指定路径
def daysToDayIO(inputPath, sheetData, sheetDate, daysNum, outputPath):
    import pickle
    dataset = excel_to_matrix(inputPath, sheetData)
    date = excel_to_matrix(inputPath, sheetDate)
    i1 = 0
    i2 = i1 + daysNum
    daysAndDayList = []
    while i2 < len(date):
        if date[i2][0] - date[i1][0] == daysNum:
            daysList = []
            for i in range(daysNum):
                daysList += dataset[i1 + i]
            daysAndDayList.append([daysList, dataset[i2]])
        i1 += 1
        i2 += 1
    with open(outputPath, 'wb') as f:
        pickle.dump(daysAndDayList, f)
    print("已成功存储到指定路径！")
    return 0

def drawPicture(arr):
    import matplotlib.pyplot as plt
    plt.plot(arr)
    plt.show()
    plt.close()
    return 0

def drawPictures(arr):
    import matplotlib.pyplot as plt
    for one in arr:
        plt.plot(list(range(len(one))), one)
    plt.show()
    plt.close()
    return 0

def drawPicturesAndSave(arr, outPath):
    import matplotlib.pyplot as plt
    for one in arr:
        plt.plot(list(range(len(one))), one)
    plt.savefig(outPath)
    plt.close()
    return 0