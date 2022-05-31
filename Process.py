# 该类用于存储整个执行过程
from train import train2
from train import train1
from train import train3
from train import AEtrain
import AllModel

input_path = "D:/Backup/桌面/EV/train_data"
lr = 0.001
modelClass = AllModel.EVModelAE2
modelClass1 = AllModel.EVModel21
modelClass2 = AllModel.EVModel31
modelClass3 = AllModel.EVModel32
lossOutPath = "D:/Backup/桌面/EV/test5.xlsx"
modelOutPath = "D:/Backup/桌面/EV/lstmModel"

input_path = "D:/桌面/relatedFile/train_data_twoFactors"
lossOutPath = "D:/桌面/relatedFile/loss2.xlsx"
modelOutPath = "D:/桌面/relatedFile/two_factors_model/"

excelPath = "D:/Backup/桌面/EVK/hf/loadData.xlsx"
batchsz = 128
epochNum = 100
recordOutputPath = "D:/Backup/桌面/EVK/hf/lossRecord.xlsx"
modelSavePath = "D:/Backup/桌面/EVK/hf/AEmodel"


def func1():
    train2(input_path, lr, modelClass1, modelClass2, lossOutPath, modelOutPath)
def func2():
    train1(input_path, lr, modelClass, lossOutPath, modelOutPath)
def func3():
    train3(input_path, lr, modelClass1, modelClass2, modelClass3, lossOutPath, modelOutPath)
def func4():
    AEtrain(modelClass, excelPath, lr, batchsz, epochNum, recordOutputPath, modelSavePath)
    
func4()