# 该类用于存储整个执行过程
from train import train2
from train import train1
from train import train3
import AllModel

input_path = "D:/Backup/桌面/EV/train_data"
lr = 0.001
modelClass = AllModel.EVModel3
modelClass1 = AllModel.EVModel21
modelClass2 = AllModel.EVModel31
modelClass3 = AllModel.EVModel32
lossOutPath = "D:/Backup/桌面/EV/test5.xlsx"
modelOutPath = "D:/Backup/桌面/EV/lstmModel"

input_path = "D:/桌面/relatedFile/train_data_twoFactors"
lossOutPath = "D:/桌面/relatedFile/loss2.xlsx"
modelOutPath = "D:/桌面/relatedFile/two_factors_model/"




def func1():
    train2(input_path, lr, modelClass1, modelClass2, lossOutPath, modelOutPath)
def func2():
    train1(input_path, lr, modelClass, lossOutPath, modelOutPath)
def func3():
    train3(input_path, lr, modelClass1, modelClass2, modelClass3, lossOutPath, modelOutPath)

    
func3()