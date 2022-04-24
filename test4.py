
import pandas as pd
import pickle
import numpy as np

# input_path = "D:/Backup/桌面/EV/dataConOthers"
# train_data_outputPath = "D:/Backup/桌面/EV/train_data2"
# test_data_outputPath = "D:/Backup/桌面/EV/test_data2"
# cut = 0.7
#
# data = pd.read_pickle(input_path)
# newData_train = []
# newData_test = []
# for i in range(len(data)):
#     if np.random.rand() <= cut:
#         newData_train.append(data[i])
#     else:
#         newData_test.append(data[i])
# data = newData_train
# with open(train_data_outputPath, 'wb') as f:
#     pickle.dump(newData_train, f)
# with open(test_data_outputPath, 'wb') as f:
#     pickle.dump(newData_test, f)
# print(len(newData_test))
# print(len(newData_train))

# input_path = "D:/Backup/桌面/EV/weatherInformation2"
# output_path = "D:/Backup/桌面/EV/weatherInformation3"
# dataset = pd.read_pickle(input_path)
# newDataset = []
# for data in dataset:
#     index = data.find('深圳市气象台')
#     if index == -1:
#         newDataset.append(data)
#     else:
#         newDataset.append(data[0 : index])
# with open(output_path, 'wb') as f:
#     pickle.dump(newDataset, f)
# print(dataset[13])
# print(newDataset[13])
# print(dataset[113])
# print(newDataset[113])


# from Pretreatment import sentencesToWordsListsIO
# import pandas as pd
# input_path = "D:/Backup/桌面/EV/weatherInformation3"
# output_path = "D:/Backup/桌面/EV/weatherInformation4"
# sentencesToWordsListsIO(input_path, output_path)
#
# x1 = pd.read_pickle(input_path)
# x2 = pd.read_pickle(output_path)
# print(x1[100])
# print(x2[100])
# print(x1[200])
# print(x2[200])

# from Pretreatment import word2VecModelIO
# word2VecModelIO("D:/Backup/桌面/EV/weatherInformation4", "D:/Backup/桌面/EV/word2VetModel.model")
# model_path = "D:/Backup/桌面/EV/word2VecModel.model"
# from gensim.models import Word2Vec
# model = Word2Vec.load(model_path)
# y = model.wv["晴"]
# print(y)

from Pretreatment import excel_to_matrix
# import pandas as pd
# import datetime
# import pickle
# train_data_date = "D:/Backup/桌面/EV/每日天气信息_2920000903461.xlsx"
# weatherData = pd.read_pickle("D:/Backup/桌面/EV/weatherInformation4")
# output_path = "D:/Backup/桌面/EV/need_weatherInformation4"
# dataset_date = excel_to_matrix(train_data_date, 2)
# dict = {}
# newDate_list = []
# originDay = datetime.date(year=1899, month=12, day=30)
#
# for i in range(len(dataset_date)):
#     strDate = dataset_date[i][0]
#     year = int(strDate[0:4])
#     month = int(strDate[5:7])
#     day = int(strDate[8:10])
#     theNum = (datetime.date(year=year, month=month, day=day) - originDay).days
#     if dict.get(theNum) is None:
#         dict[theNum] = weatherData[i]
#     else:
#         dict[theNum] += weatherData[i]
#
#
#
# need_weather_list = []
# need_date = excel_to_matrix("D:/Backup/桌面/EV/minle.xlsx", 3)
# for data in need_date:
#     if dict.get(data[0]) is None:
#         need_weather_list.append([])
#     else:
#         need_weather_list.append(dict[data[0]])
# with open(output_path, 'wb') as f:
#     pickle.dump(need_weather_list, f)

# import pandas as pd
# x = pd.read_pickle("D:/Backup/桌面/EV/need_weatherInformation4")
# y = excel_to_matrix("D:/Backup/桌面/EV/minle.xlsx", 3)
# print(len(x), len(y))

# from Pretreatment import wordsListSetToVecIO
# import pandas as pd
# wordsListSetToVecIO("D:/Backup/桌面/EV/need_weatherInformation4", "D:/Backup/桌面/EV/word2VecModel.model", "D:/Backup/桌面/EV/wordVec_weatherInformation")
# x1 = pd.read_pickle("D:/Backup/桌面/EV/need_weatherInformation4")
# x2 = pd.read_pickle("D:/Backup/桌面/EV/wordVec_weatherInformation")
# t = 2
weatherDataPath = "D:/Backup/桌面/EV/wordVec_weatherInformation"
inputPath = "D:/Backup/桌面/EV/minle.xlsx"
sheetData = 1
sheetDate = 3
daysNum = 7
outputPath = "D:/Backup/桌面/EV/toTrain_wordVec_weatherInformation"
from Pretreatment import weatherNumToDayIO
# weatherNumToDayIO(weatherDataPath, inputPath, sheetData, sheetDate, daysNum, outputPath)
import pandas as pd
x = pd.read_pickle(outputPath)
# x1 = x[2]
# x2 = x1[0]
# x3 = x1[1]
# t = 2
