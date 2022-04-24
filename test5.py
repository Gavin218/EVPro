
import pandas as pd
import pickle


# minle = pd.read_pickle("D:/桌面/relatedFile/minleDays")
# weather_path = "D:/桌面/relatedFile/toTrain_wordVec_weatherInformation.weatherInformation"
new_weather_path = "D:/桌面/relatedFile/two_factors_data"
train_data_outputPath = "D:/桌面/relatedFile/train_data_twoFactors"
test_data_outputPath = "D:/桌面/relatedFile/test_data_twoFactors"
# cons = 42600
# dataset = pd.read_pickle(weather_path)
# for i in range(len(dataset)):
#     t = len(dataset[i][0])
#     need_list = [0] * (cons - t)
#     dataset[i][0] = need_list + dataset[i][0]
#
# with open(new_weather_path, 'wb') as f:
#     pickle.dump(dataset, f)
#
# c = 0
# for i in dataset:
#     if len(i[0]) == cons:
#         c += 1
# print(len(dataset), c)


from Pretreatment import cutTrainAndTestIO
cutTrainAndTestIO(new_weather_path, 0.7, train_data_outputPath, test_data_outputPath)