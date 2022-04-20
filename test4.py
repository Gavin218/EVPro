
import pandas as pd
import pickle
import numpy as np

input_path = "D:/Backup/桌面/EV/dataConOthers"
train_data_outputPath = "D:/Backup/桌面/EV/train_data2"
test_data_outputPath = "D:/Backup/桌面/EV/test_data2"
cut = 0.7

data = pd.read_pickle(input_path)
newData_train = []
newData_test = []
for i in range(len(data)):
    if np.random.rand() <= cut:
        newData_train.append(data[i])
    else:
        newData_test.append(data[i])
data = newData_train
with open(train_data_outputPath, 'wb') as f:
    pickle.dump(newData_train, f)
with open(test_data_outputPath, 'wb') as f:
    pickle.dump(newData_test, f)
print(len(newData_test))
print(len(newData_train))