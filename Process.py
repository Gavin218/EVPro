# 该类用于存储整个执行过程
from train import train2
from train import train1
import AllModel

input_path = "D:/Backup/桌面/EV/train_data"
lr = 0.001
modelClass = AllModel.LSTMModel1
modelClass1 = AllModel.EVModel21
modelClass2 = AllModel.EVModel22
lossOutPath = "D:/Backup/桌面/EV/test5.xlsx"
modelOutPath = "D:/Backup/桌面/EV/lstmModel"


def func1():
    train2(input_path, lr, modelClass1, modelClass2, lossOutPath, modelOutPath)
def func2():
    train1(input_path, lr, modelClass, lossOutPath, modelOutPath)
    
func2()