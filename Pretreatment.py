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

def weatherNumToDayIO(weatherDataPath, inputPath, sheetData, sheetDate, daysNum, outputPath):
    import pickle
    import pandas as pd
    weatherData = pd.read_pickle(weatherDataPath)
    dataset = excel_to_matrix(inputPath, sheetData)
    date = excel_to_matrix(inputPath, sheetDate)
    i1 = 0
    i2 = i1 + daysNum
    weatherAndDayList = []
    while i2 < len(date):
        if date[i2][0] - date[i1][0] == daysNum:
            daysList = []
            for i in range(daysNum):
                daysList += weatherData[i1 + i]
            weatherAndDayList.append([daysList, dataset[i2]])
        i1 += 1
        i2 += 1
    with open(outputPath, 'wb') as f:
        pickle.dump(weatherAndDayList, f)
    print("已成功存储到指定路径！")
    return 0

# 输入为数据路径和日期路径(包含周末，季节以及新冠数据)，将合规的seq2seq写入指定路径
def daysToDayAndOthersIO(inputPath, sheetData, sheetDate, daysNum, outputPath):
    import pickle
    dataset = excel_to_matrix(inputPath, sheetData)
    date = excel_to_matrix(inputPath, sheetDate)
    i1 = 0
    i2 = i1 + daysNum
    daysAndDayList = []
    while i2 < len(date):
        if date[i2][0] - date[i1][0] == daysNum:
            daysList = []
            xinGuan = []
            for i in range(daysNum):
                daysList += dataset[i1 + i]
                xinGuan.append(date[i][3])
            daysAndDayList.append([daysList, dataset[i2], xinGuan, date[i2][1], date[i2][2]])
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

def kMeansAndSaveLabelsIO(inputPath, k, labelPath):
    from sklearn.cluster import KMeans
    import pandas as pd
    if inputPath[len(inputPath) - 4 : len(inputPath)] == 'xlsx': 
        dataset = excel_to_matrix(inputPath, 0)
    else:
        dataset = pd.read_pickle(inputPath)
    km = KMeans(n_clusters=k).fit(dataset)
    labels = km.labels_
    labelsList = [[one] for one in labels]
    writeToExcel(labelPath, labelsList)
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

def drawPicturesAndSaveByDifferentClass(inputPath, sheet, labelPath, k, outputPath):
    import matplotlib.pyplot as plt
    import pandas as pd
    if sheet >= 0:
        dataset = excel_to_matrix(inputPath, sheet)
    else:
        dataset = pd.read_pickle(inputPath)
    labelList = excel_to_matrix(labelPath, 0)
    clusterList = []
    for i in range(k):
        clusterList.append([])
    for i in range(len(dataset)):
        label = labelList[i][0]
        clusterList[int(label)].append(dataset[i])
    lengthList = []
    for i in range(k):
        arr = clusterList[i]
        lengthList.append(len(arr))
        for one in arr:
            plt.plot(list(range(len(one))), one)
        plt.savefig(outputPath + "class{num}".format(num=i+1))
        plt.close()
    print(lengthList)
    return 0

def drawPicturesAndSaveByDifferentClass2(inputPath, sheet, labelPath, k, outputPath):
    import matplotlib.pyplot as plt
    import pandas as pd
    if sheet >= 0:
        dataset = excel_to_matrix(inputPath, sheet)
    else:
        dataset = pd.read_pickle(inputPath)
    labelList = excel_to_matrix(labelPath, 0)
    clusterList = []
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    for i in range(k):
        clusterList.append([])
    for i in range(len(dataset)):
        label = labelList[i][0]
        clusterList[int(label)].append(dataset[i])
    lengthList = []
    for i in range(k):
        arr = clusterList[i]
        lengthList.append(len(arr))
        for one in arr:
            plt.plot(list(range(len(one))), one, color=colorList[i])
        plt.savefig(outputPath + "class{num}".format(num=i+1))
        plt.close()
    print(lengthList)
    return 0

# 转换坐标格式为时间格式
def drawPicturesAndSaveByDifferentClass3(inputPath, sheet, labelPath, k, outputPath):
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd

    if sheet >= 0:
        dataset = excel_to_matrix(inputPath, sheet)
    else:
        dataset = pd.read_pickle(inputPath)
    labelList = excel_to_matrix(labelPath, 0)
    clusterList = []
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    for i in range(k):
        clusterList.append([])
    for i in range(len(dataset)):
        label = labelList[i][0]
        clusterList[int(label)].append(dataset[i])
    lengthList = []
    length = 96
    X_label = list(range(96))

    X_label_str = []
    for i in range(96):
        if X_label[i] * 15 % 240 == 0:
            X_label_str.append(str(datetime.time(minute=(X_label[i]*15)%60, hour=int(X_label[i]*15/60)))[0:5])
        else:
            X_label_str.append('')
    X_label_str[95] = "24:00"
    for i in range(k):
        arr = clusterList[i]
        lengthList.append(len(arr))
        for one in arr:
            plt.plot(list(range(len(one))), one, color=colorList[i])
        plt.xticks(list(range(length)), X_label_str)

        plt.savefig(outputPath + "class{num}".format(num=i+1))
        plt.close()
    print(lengthList)
    return 0

def drawOnePictureToShowDifferentClass(inputPath, sheet, labelPath):
    import matplotlib.pyplot as plt
    dataset = excel_to_matrix(inputPath, sheet)
    length = len(dataset[0])
    labelList = excel_to_matrix(labelPath, 0)
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    for i in range(len(dataset)):
        plt.plot(list(range(length)), dataset[i], colorList[int(labelList[i][0])])
    plt.show()
    plt.close()
    return 0

def drawOnePictureToShowDifferentClass2(inputPath, sheet, labelPath, outputPath):
    import matplotlib.pyplot as plt
    import pandas as pd
    dataset = excel_to_matrix(inputPath, sheet)
    length = len(dataset[0])
    labelList = excel_to_matrix(labelPath, 0)
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    for i in range(len(dataset)):
        plt.plot(list(range(length)), dataset[i], colorList[int(labelList[i][0])])
    plt.savefig(outputPath)
    plt.close()
    return 0

def drawOnePictureToShowDifferentClass3(inputPath, sheet, labelPath, outputPath):
    import matplotlib.pyplot as plt
    import datetime
    dataset = excel_to_matrix(inputPath, sheet)
    length = len(dataset[0])
    labelList = excel_to_matrix(labelPath, 0)
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    length = 96
    X_label = list(range(96))
    X_label_str = []
    for i in range(96):
        if X_label[i] * 15 % 240 == 0:
            X_label_str.append(str(datetime.time(minute=(X_label[i] * 15) % 60, hour=int(X_label[i] * 15 / 60)))[0:5])
        else:
            X_label_str.append('')
    X_label_str[95] = "24:00"
    for i in range(len(dataset)):
        plt.plot(list(range(length)), dataset[i], colorList[int(labelList[i][0])])
    plt.xticks(list(range(length)), X_label_str)
    plt.savefig(outputPath)
    plt.close()
    return 0

def drawElbowPicture(dataset, k_min, k_max):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(k_min, k_max):
        # km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
        km = KMeans(n_clusters=k).fit(dataset)
        SSE.append(km.inertia_)  # estimator.inertia_获取聚类准则的总和
    X = range(k_min, k_max)
    print(SSE)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()
    return 0

# 该函数未完成！（好像也不需要完成了）
def drawElbowPictures(dataPath, k_min, k_max, picturesSsvePath):
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    dataset = pd.read_pickle(dataPath)
    '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(k_min, k_max):
        # km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
        km = KMeans(n_clusters=k).fit(dataset)
        SSE.append(km.inertia_)  # estimator.inertia_获取聚类准则的总和
    X = range(k_min, k_max)
    print(SSE)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()
    return 0

# 读取为dataframe格式，输出仍为dataframe格式
def clusterDatasetByLabel(inputPath, labelPath, labelSheet, datePath, dateSheet, k, outputPath):
    import pandas as pd
    import datetime
    import pickle
    dataset = pd.read_pickle(inputPath)
    labelList = excel_to_matrix(labelPath, labelSheet)
    dateList = excel_to_matrix(datePath, dateSheet)
    datasetAfterClustering = []   # 该数组用于存放分类好的dataset
    setList = []    # 该列表用于存放不同类别中的日期
    for i in range(k):
        datasetAfterClustering.append([])
        setList.append(set())
    # 用于将日期放入对应set中
    for i in range(len(labelList)):
        label = int(labelList[i][0])
        setList[label].add(dateList[i][0])
    # 将数据放入对应列表中
    originDay = datetime.date(year=1899, month=12, day=30)
    for i in range(len(dataset)):
        data = dataset[i]
        date = data[0].date()
        dateNum = (date - originDay).days
        for j in range(len(setList)):
            oneSet = setList[j]
            if dateNum in oneSet:
                datasetAfterClustering[j].append(data)
                break
    with open(outputPath, 'wb') as f:
        pickle.dump(datasetAfterClustering, f)
    return 0

def onlyKeepMinutes(inputPath, outputPath):
    import pandas as pd
    import pickle
    dataset = pd.read_pickle(inputPath)
    k = len(dataset)
    newDataset = []
    for i in range(k):
        newDataset.append([])
    for i in range(k):
        oneClassData = dataset[i]
        for j in range(len(oneClassData)):
            data = oneClassData[j]
            newDataset[i].append([data[0].time().minute + data[0].time().hour * 60, data[1]])
    with open(outputPath, 'wb') as f:
        pickle.dump(newDataset, f)
    return 0

# 读取日期及起始充电时间，充电时长的Excel，转换为dataframe进行保存
def orderToDataframeIO(inputPath, sheet, outputPath):
    import pickle
    import datetime
    dataset = excel_to_matrix(inputPath, sheet)
    newList = []
    # i = 0
    for data in dataset:
        # print(i)
        wholeDateStr = data[0]
        chargingTime = data[1]
        wholeDate = datetime.datetime.strptime(wholeDateStr, "%Y-%m-%d %H:%M:%S")
        newList.append([wholeDate.date(), wholeDate.time(), chargingTime])
        # i += 1
    with open(outputPath, 'wb') as f:
        pickle.dump(newList, f)
    return 0

# 读取日期及起始充电时间，充电时长以及充电金额的Excel，转换为dataframe进行保存
def orderToDataframeIO2(inputPath, sheet, outputPath):
    import pickle
    import datetime
    dataset = excel_to_matrix(inputPath, sheet)
    newList = []
    # i = 0
    for data in dataset:
        # print(i)
        wholeDateStr = data[0]
        chargingTime = data[1]
        chargingMoney = data[2]
        chargingEle = data[3]
        # wholeDate = datetime.datetime.strptime(wholeDateStr, "%Y/%m/%d %H:%M")
        wholeDate = datetime.datetime.strptime(wholeDateStr, "%Y-%m-%d %H:%M:%S")
        newList.append([wholeDate.date(), wholeDate.time(), chargingTime, chargingMoney, chargingEle])
        # i += 1
    with open(outputPath, 'wb') as f:
        pickle.dump(newList, f)
    return 0

def clusterOrderDataByLabels(datePath, dateSheet, labelPath, labelSheet, k, orderDataPath, outputPath):
    import pandas as pd
    import datetime
    import pickle
    dateList = excel_to_matrix(datePath, dateSheet)
    labelList = excel_to_matrix(labelPath, labelSheet)
    orderDataList = pd.read_pickle(orderDataPath)
    outputList = []
    for i in range(k):
        outputList.append([])
    dict = {}
    for i in range(len(dateList)):
        dict[dateList[i][0]] = labelList[i][0]  # 这儿的键虽然是日期，但只为浮点数，并非日期格式
    originDay = datetime.date(year=1899, month=12, day=30)
    for orderData in orderDataList:
        theDate = orderData[0]
        dateFloat = (theDate - originDay).days
        if dateFloat in dict:
            label = int(dict[dateFloat])
            outputList[label].append([orderData[1], orderData[2]])
    lengthList = []
    for i in range(k):
        lengthList.append(len(outputList[i]))
    with open(outputPath, 'wb') as f:
        pickle.dump(outputList, f)
    print("各类的订单数目为", lengthList)
    return 0

# 加上电价这一变量
def clusterOrderDataByLabels2(datePath, dateSheet, labelPath, labelSheet, k, orderDataPath, outputPath):
    import pandas as pd
    import datetime
    import pickle
    dateList = excel_to_matrix(datePath, dateSheet)
    labelList = excel_to_matrix(labelPath, labelSheet)
    orderDataList = pd.read_pickle(orderDataPath)
    outputList = []
    for i in range(k):
        outputList.append([])
    dict = {}
    for i in range(len(dateList)):
        dict[dateList[i][0]] = labelList[i][0]  # 这儿的键虽然是日期，但只为浮点数，并非日期格式
    originDay = datetime.date(year=1899, month=12, day=30)
    for orderData in orderDataList:
        theDate = orderData[0]
        dateFloat = (theDate - originDay).days
        if dateFloat in dict and orderData[4] > 0:
            label = int(dict[dateFloat])
            outputList[label].append([orderData[1], orderData[2], orderData[3], orderData[4]])
    lengthList = []
    for i in range(k):
        lengthList.append(len(outputList[i]))
    with open(outputPath, 'wb') as f:
        pickle.dump(outputList, f)
    print("各类的订单数目为", lengthList)
    return 0

def orderKmeansAndDrawElbow(orderPath, sheet, k_min, k_max):
    import pandas as pd
    orderDataList = pd.read_pickle(orderPath)[sheet]
    dataset = []
    for orderData in orderDataList:
        dataset.append([orderData[0].hour * 60 + orderData[0].minute, orderData[1]])
    drawElbowPicture(dataset, k_min=k_min, k_max=k_max)
    return 0

def orderKmeansAndDrawScatter(orderPath, sheet, k):
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    orderDataList = pd.read_pickle(orderPath)[sheet]
    dataset = []
    for orderData in orderDataList:
        startTime = orderData[0].hour * 60 + orderData[0].minute
        chargingTime = orderData[1]
        dataset.append([startTime, chargingTime])
    km = KMeans(n_clusters=k).fit(dataset)
    labels = km.labels_
    colorList = ['blue', 'red', 'green', 'black', 'orange', 'brown', 'purple', 'yellow']
    clusterList = []
    for i in range(k):
        clusterList.append([])    # 每类一个列表
        clusterList[i].append([]) # X坐标一个列表
        clusterList[i].append([]) # Y坐标一个列表
    for i in range(len(dataset)):
        clusterList[labels[i]][0].append(dataset[i][0])
        clusterList[labels[i]][1].append(dataset[i][1])
    for i in range(k):
        plt.scatter(clusterList[i][0], clusterList[i][1], color=colorList[i], s=10)
    plt.show()
    return 0

def opticsCluster(arr, minSample):
    from sklearn.cluster import OPTICS
    clustering = OPTICS(min_samples=minSample).fit(arr)
    result = clustering.labels_
    out, leng = [], []
    for j in range(max(result) + 1):
        outi = [i for i, x in enumerate(result) if x == j]
        out.append(outi)
        leng.append(len(outi))
    return result, out, leng

def opticsClusterIO(oriPath, sheet, minSample, newPath):
    data = excel_to_matrix(oriPath, sheet)
    opticsResult = opticsCluster(data, minSample)
    doc = open(newPath, 'w')
    print(opticsResult, file=doc)
    print("聚类结果已输出至指定路径！")
    doc.close()
    return 0

def orderOpticsAndDrawScatter(orderPath, sheet, minSample):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import OPTICS
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    orderDataList = pd.read_pickle(orderPath)[sheet]
    dataset = []
    for orderData in orderDataList:
        startTime = orderData[0].hour * 60 + orderData[0].minute
        chargingTime = orderData[1]
        dataset.append([startTime, chargingTime])
    dataset = scaler.fit_transform(dataset)   # 进行归一化
    clustering = OPTICS(min_samples=minSample).fit(dataset)
    labels = clustering.labels_
    clusterNum = max(labels) + 1
    print("聚类数目为", clusterNum)
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    clusterList = []
    for i in range(clusterNum + 1):
        clusterList.append([])    # 每类一个列表
        clusterList[i].append([]) # X坐标一个列表
        clusterList[i].append([]) # Y坐标一个列表
    for i in range(len(dataset)):
        if labels[i] > -1:
            clusterList[labels[i]][0].append(dataset[i][0])
            clusterList[labels[i]][1].append(dataset[i][1])
        else:
            clusterList[clusterNum][0].append(dataset[i][0])
            clusterList[clusterNum][1].append(dataset[i][1])
    lengthList = []
    for i in range(clusterNum):
        # plt.scatter(clusterList[i][0], clusterList[i][1], color=colorList[i], s=10)
        plt.scatter(clusterList[i][0], clusterList[i][1], s=5)
        lengthList.append(len(clusterList[i][0]))
    # plt.scatter(clusterList[clusterNum][0], clusterList[clusterNum][0], s=10, color="grey")
    print("各类的聚类数目为", lengthList)
    plt.show()
    return 0


# 输入格式为float, 输出1为周末，0为非周末
def dateToWeekIO(inputPath, sheet, outputPath):
    import datetime
    originDay = datetime.date(year=1899, month=12, day=30)
    dateSet = excel_to_matrix(inputPath, sheet)
    weekList = []
    for date in dateSet:
        nowday = originDay + datetime.timedelta(days=date[0])
        week = nowday.weekday()
        weekList.append([1]) if week >= 5 else weekList.append([0])
    writeToExcel(outputPath, weekList)
    return 0

def readXlsxShowClusterNum(inputPath, sheet, k):
    labelList = excel_to_matrix(inputPath, sheet)
    result = [0] * k
    for label in labelList:
        result[int(label[0])] += 1
    print(result)
    return 0



def readCsvToExistDateDataIO(inputPath, existPath, sheet, outputPath):
    import pandas as pd
    import datetime
    dataset = pd.read_csv(inputPath)
    dateset = excel_to_matrix(existPath, sheet)
    patientsList = dataset["xzqz"]
    timeLengthList = dataset["jzsj"]
    dateList = dataset["jzrq"]
    dict = {}
    originDay = datetime.date(year=1899, month=12, day=30)
    for i in range(len(dataset)):
        if timeLengthList[i] == "24时":
            theDate = datetime.datetime.strptime(dateList[i], "%Y年%m月%d日").date()
            dict[theDate] = patientsList[i]
    newPatientsList = []
    for date in dateset:
        nowDate = originDay + datetime.timedelta(days=date[0])
        newPatientsList.append([dict[nowDate]]) if nowDate in dict else newPatientsList.append([0])
    writeToExcel(outputPath, newPatientsList)
    return 0


# 若所读取源文件中不存在多组数据，则取sheet = -1
def kMeansTSENPlot(filePath, sheet, clusterNum):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    colorList = ['blue', 'green', 'red', 'yellow', 'black', 'orange', 'brown', 'purple']
    if sheet == -1:
        dataset = pd.read_pickle(filePath)
    else:
        dataset = pd.read_pickle(filePath)[sheet]
    km = KMeans(n_clusters=clusterNum).fit(dataset)
    kmResult = km.labels_
    tsne = TSNE(n_components=2)
    embeddedData = tsne.fit_transform(dataset)
    for i in range(clusterNum):
        rowIndex, colIndex = [], []
        for j in range(len(kmResult)):
            print(j)
            if kmResult[j] == i:
                rowIndex.append(embeddedData[j][0])
                colIndex.append(embeddedData[j][1])
        plt.scatter(rowIndex, colIndex, s=1, c=colorList[i])
    plt.show()
    return 0

# 用于读取现有数据集，并根据所设阈值进行切分，所设定cut为训练集所占比例
def cutTrainAndTestIO(input_path, cut, train_data_outputPath, test_data_outputPath):
    import pandas as pd
    import numpy as np
    import pickle
    data = pd.read_pickle(input_path)
    newData_train = []
    newData_test = []
    for i in range(len(data)):
        if np.random.rand() <= cut:
            newData_train.append(data[i])
        else:
            newData_test.append(data[i])
    with open(train_data_outputPath, 'wb') as f:
        pickle.dump(newData_train, f)
    with open(test_data_outputPath, 'wb') as f:
        pickle.dump(newData_test, f)
    print(len(newData_test))
    print(len(newData_train))
    print(len(newData_train) / len(data))
    print("已随机切分完毕并存储至指定路径！")
    return 0

def readXlsxWritePdIO(input_path, sheet, output_path):
    import pickle
    dataset = excel_to_matrix(input_path, sheet)
    datasetList = []
    for data in dataset:
        datasetList.append(data[0])
    with open(output_path, 'wb') as f:
        pickle.dump(datasetList, f)
    print("已成功存储至指定路径！")
    return 0

def sentenceToWordsList(sentence):
    import jieba
    result_list = []
    redundantList = ['，','。','》','《','；','：','’','【','】','{','}',
                     '！','#','%','&','*','-','_','+','=','0','1','2','3','4','5','6','7','8','9','(',')','（','）',
                     '“','”','‘','’','℃','、']
    seg = jieba.cut(sentence)
    for one in seg:
        if one not in redundantList:
            try:
                num = int(one)
            except:
                result_list.append(one)
    return result_list

def sentencesToWordsListsIO(input_path, output_path):
    import pandas as pd
    import pickle
    dataset = pd.read_pickle(input_path)
    newDataset = []
    for data in dataset:
        newDataset.append(sentenceToWordsList(data))
    with open(output_path, 'wb') as f:
        pickle.dump(newDataset, f)
    print("已成功切分并存储至指定路径！")
    return 0


# word2vec模型的保存，注：output需要以.model结尾
def word2VecModelIO(input_path, model_output_path):
    from gensim.models import word2vec
    import pandas as pd
    sentences = pd.read_pickle(input_path)
    model = word2vec.Word2Vec(sentences, min_count=1)
    model.save(model_output_path)
    print("模型已成功存储至指定路径！")
    return 7

def wordsListSetToVec(dataset, model):
    newDataset = []
    for data in dataset:
        newData = []
        for one in data:
            newData += model.wv[one].tolist()
        newDataset.append(newData)
    return newDataset

def wordsListSetToVecIO(input_path, model_path, output_path):
    import pandas as pd
    from gensim.models import Word2Vec
    import pickle
    dataset = pd.read_pickle(input_path)
    model = Word2Vec.load(model_path)
    newDataset = wordsListSetToVec(dataset=dataset, model=model)
    with open(output_path, 'wb') as f:
        pickle.dump(newDataset, f)
    print("已完成全部词向量转换并存储至指定路径！")
    return 0

def DTWDistanceIO(excelPath, sheet, outputPath):
    import tslearn.metrics as metrics
    # 读取数据
    X = excel_to_matrix(excelPath, sheet)
    # 采用tslearn中的DTW系列及变种算法计算相似度，生成距离矩阵dists
    dists = metrics.cdist_dtw(X)  # dba + dtw
    # dists = metrics.cdist_soft_dtw_normalized(X,gamma=0.5) # softdtw
    writeToExcel(outputPath, dists)
    return 0

def DTWDistanceIO2(excelPath, outputPath):
    import tslearn.metrics as metrics
    import pandas as pd
    import pickle
    # 读取数据
    dataset = pd.read_excel(excelPath).values
    loadData = dataset[:, 1:97:1]
    # dateList = dataset[:, 0]
    # 采用tslearn中的DTW系列及变种算法计算相似度，生成距离矩阵dists
    dists = metrics.cdist_dtw(loadData)
    with open(outputPath, 'wb') as f:
        pickle.dump(dists, f)
    return 0

def distanceMatToPredByKMedoidsIO(distanceMatPath, distanceSheet, k, outputPath):
    from sklearn_extra.cluster import KMedoids
    distanceMat = excel_to_matrix(distanceMatPath, distanceSheet)
    km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
    y_pred = km.fit_predict(distanceMat)
    y_pred = [[one] for one in y_pred]
    writeToExcel(outputPath, y_pred)
    return 0

def distanceMatToPredByKMedoidsIO2(distanceMatPath, k, outputPath):
    from sklearn_extra.cluster import KMedoids
    import pandas as pd
    import pickle
    distanceMat = pd.read_pickle(distanceMatPath)
    km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
    y_pred = km.fit_predict(distanceMat)
    with open(outputPath, 'wb') as f:
        pickle.dump(y_pred, f)
    return 0

def DTWkMedoidsIO(excelPath, sheet, k):
    import numpy as np
    from sklearn_extra.cluster import KMedoids
    import tslearn.metrics as metrics

    # 读取数据
    X = excel_to_matrix(excelPath, 1)
    # 读取标签
    # 声明precomputed自定义相似度计算方法
    km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
    # 采用tslearn中的DTW系列及变种算法计算相似度，生成距离矩阵dists
    dists = metrics.cdist_dtw(X)  # dba + dtw
    # dists = metrics.cdist_soft_dtw_normalized(X,gamma=0.5) # softdtw
    y_pred = km.fit_predict(dists)
    return y_pred



def showDetailList(clusterList, k, isweekList, seasonList, patientList, lowList, highList, isRainList):
    newList = []
    existDataList = [isweekList, seasonList, patientList, lowList, highList, isRainList]
    for i in range(k):
        newList.append([])
        for j in range(len(existDataList)):
            newList[i].append([0, 0])   # 第一个0为数值和，第二个为数目
    for i in range(len(clusterList)):
        label = int(clusterList[i])
        for j in range(len(existDataList)):
            newList[label][j][0] += existDataList[j][i]
            newList[label][j][1] += 1
    for i in range(k):
        for j in range(len(existDataList)):
            if newList[i][j][1] > 0:
                newList[i][j] = newList[i][j][0] / newList[i][j][1]
            else:
                newList[i][j] = None
    return newList

def showStatistics(inputPath, sheet, outputPath):
    import numpy as np
    dataset = np.array(excel_to_matrix(inputPath, sheet))
    clusterNum = dataset.shape[1]
    isweekList = dataset[:,0]
    seasonList = dataset[:,1]
    patientList = dataset[:,2]
    lowList = dataset[:,3]
    highList = dataset[:,4]
    isRainList = dataset[:,5]
    totalList = []
    for i in range(2, clusterNum - 4):
        clusterList = dataset[:,i + 4]
        detailList = showDetailList(clusterList, i, isweekList, seasonList, patientList, lowList, highList, isRainList)
        totalList.append(detailList)
    print(totalList)
    newList = []
    for i in range(len(totalList)):
        for data in totalList[i]:
            newList.append(data)
    writeToExcel(outputPath, newList)
    return 0

def readCsvAndOutput(inputPath, datePath, dateSheet, outputPath):
    import pandas as pd
    import datetime
    csvData = pd.read_csv(inputPath)
    dateData = excel_to_matrix(datePath, dateSheet)
    needDate = set()
    for date in dateData:
        needDate.add(date[0])
    existDateList = csvData["date"]
    existLowList = csvData["low"]
    existHighList = csvData["high"]
    existIsRainList = csvData["type"]
    originDay = datetime.date(year=1899, month=12, day=30)
    outputList = []
    for i in range(len(existDateList)):
        oneDate = datetime.datetime.strptime(existDateList[i], "%Y/%m/%d")
        dateNum = (oneDate.date() - originDay).days
        if dateNum in needDate:
            outputList.append([existDateList[i], existLowList[i], existHighList[i], existIsRainList[i]])
    writeToExcel(outputPath, outputList)
    return 0

def csvCleanIO(inputPath, outputPath):
    import pandas as pd
    import csv
    dataset = pd.read_csv(inputPath)

    # dataset = pd.read_csv('Building_CS_Order.csv')
    # dataset = pd.read_csv('hf_order.csv')

    dataset1 = dataset.drop(dataset.loc[dataset['充电时长（分）'] < 6, '充电时长（分）'].index)
    dataset1 = dataset1.reset_index(drop=True)
    newList = []
    startTimeList = dataset1['充电开始时间']
    chargingTimeList = dataset1['充电时长（分）']
    for i in range(len(dataset1)):
        newList.append([startTimeList[i], chargingTimeList[i]])
    writeToExcel(outputPath, newList)
    return 0


# 加上了充电费（元）和充电电量（度）这一列
def csvCleanIO2(inputPath, outputPath):
    import pandas as pd
    import csv
    dataset = pd.read_csv(inputPath)

    # dataset = pd.read_csv('Building_CS_Order.csv')
    # dataset = pd.read_csv('hf_order.csv')

    dataset1 = dataset.drop(dataset.loc[dataset['充电时长（分）'] < 6, '充电时长（分）'].index)
    dataset1 = dataset1.reset_index(drop=True)
    newList = []
    startTimeList = dataset1['充电开始时间']
    chargingTimeList = dataset1['充电时长（分）']
    chargingEle = dataset1['充电电量（度）']
    chargingMoneyList = dataset1['充电费（元）']
    for i in range(len(dataset1)):
        newList.append([startTimeList[i], chargingTimeList[i], chargingMoneyList[i], chargingEle[i]])
    writeToExcel(outputPath, newList)
    return 0

def statisticsElePriceByClass(orderClassPath, outputPath):
    import pandas as pd
    orderClass = pd.read_pickle(orderClassPath)
    totalList = []  # 用于存储所有类电价结果的列表
    for orderList in orderClass:
        oneClassList = [0] * 24
        for oneOrder in orderList:
            oneClassList[oneOrder[0].hour] += oneOrder[2] / oneOrder[3]
        oneClassList = [one / len(orderList) for one in oneClassList]
        totalList.append(oneClassList)
    writeToExcel(outputPath, totalList)
    return 0

# cutdate也为datetime.date格式
def analyzeDifferentPricesByLabel(dataPath, dataSheet, classPath, classSheet, label, datePath, dateSheet, cutDate):
    import datetime
    dataset = excel_to_matrix(dataPath, dataSheet)
    classSet = excel_to_matrix(classPath, classSheet)
    dateSet = excel_to_matrix(datePath, dateSheet)
    originDay = datetime.date(year=1899, month=12, day=30)
    for i in range(len(dataset)):
        if classSet[i] == label:
            return 1
    return 0

def analyzeDifferentPrices(dataPath, dataSheet, datePath, dateSheet, cutDate):
    import datetime
    import matplotlib.pyplot as plt
    dataset = excel_to_matrix(dataPath, dataSheet)
    dateSet = excel_to_matrix(datePath, dateSheet)
    originDay = datetime.date(year=1899, month=12, day=30)
    totalList = []
    beforeDaysList = []
    afterDaysList = []
    cutDateNum = (cutDate - originDay).days
    for i in range(len(dataset)):
        if dateSet[i][0] < cutDateNum:
            beforeDaysList.append(dataset[i])
        else:
            afterDaysList.append(dataset[i])
    # drawPictures(beforeDaysList)
    # drawPictures(afterDaysList)
    totalList = [beforeDaysList, afterDaysList]
    for oneList in totalList:
        X_label = list(range(96))
        X_label_str = []
        length = 96
        for i in range(96):
            if X_label[i] * 15 % 240 == 0:
                X_label_str.append(str(datetime.time(minute=(X_label[i] * 15) % 60, hour=int(X_label[i] * 15 / 60)))[0:5])
            else:
                X_label_str.append('')
        X_label_str[95] = "24:00"
        for i in range(len(oneList)):
            plt.plot(list(range(length)), oneList[i])
            plt.xticks(list(range(length)), X_label_str)
        plt.show()
        plt.close()
    return 0