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
