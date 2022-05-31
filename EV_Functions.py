# 步骤2所需代码
def DTWDistanceIO(excelPath, outputPath):
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
    print("距离矩阵已存储至指定路径！")
    return 0

# 步骤3所需代吗
def distanceMatToPredByKMedoidsIO(distanceMatPath, k, labelOutputPath, medoidsOutputPath):
    from sklearn_extra.cluster import KMedoids
    import pandas as pd
    import pickle
    distanceMat = pd.read_pickle(distanceMatPath)
    km = KMedoids(n_clusters=k, random_state=0, metric="precomputed")
    y_pred = km.fit_predict(distanceMat)
    with open(labelOutputPath, 'wb') as f:
        pickle.dump(y_pred, f)
    print("聚类标签已存储至指定路径！")
    with open(medoidsOutputPath, 'wb') as f:
        pickle.dump(km.medoid_indices_, f)
    print("聚类中心点对应序号已存储至指定路径！")
    return 0

# 步骤4所需代码
def drawPicturesAndSaveByDifferentClass(inputPath, labelPath, medoidPath, k, outputPath):
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    dataset = pd.read_excel(inputPath).values[:, 1:97:1]
    labelList = pd.read_pickle(labelPath)
    medoidList = pd.read_pickle(medoidPath)
    clusterList = []
    # colorList用于对不同类别施以不同颜色，若k值大于10，则第11种与第1种相同。
    colorList = ['blue', 'red', 'green', 'black', 'orange',
                 'brown', 'purple', 'yellow', 'pink', "aqua"]
    for i in range(k):
        clusterList.append([])
    for i in range(len(dataset)):
        label = labelList[i]
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
    print("各类别总负荷曲线已存储至指定文件夹！")
    for i in range(k):
        plt.plot(list(range(96)), dataset[medoidList[i]], color=colorList[i])
        plt.xticks(list(range(length)), X_label_str)
        plt.savefig(outputPath + "_medoid_class{num}".format(num=i + 1))
        plt.close()
    print("各典型日负荷曲线已存储至指定文件夹！")
    print("各类聚类数目为", lengthList)
    return 0


def tsfelForTypicalDays(inputPath,  medoidPath, featureOutputPath):
    import pandas as pd
    import tsfel
    dataset = pd.read_excel(inputPath).values[:, 1:97:1]
    medoidList = pd.read_pickle(medoidPath)
    typeDaysList = [dataset[i] for i in medoidList]
    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain()

    # Extract features
    df_feature = tsfel.time_series_features_extractor(cfg, typeDaysList)
    df_feature.to_excel(featureOutputPath)
    return 0

def labelToLists(loadDataPath, daysLabelPath, k_days, outputPath):
    import pandas as pd
    import pickle
    dateset = pd.read_excel(loadDataPath).values[:, 0]
    daysLabelList = pd.read_pickle(daysLabelPath)
    dateListByClass = []
    for i in range(k_days):
        dateListByClass.append([])
    for i in range(len(daysLabelList)):
        dateListByClass[int(daysLabelList[i])].append(dateset[i].date())
    with open(outputPath, 'wb') as f:
        pickle.dump(dateListByClass, f)
    return 0

def func333(loadDataPath, daysLabelPath, k_days):
    import pandas as pd
    import pickle
    dateset = pd.read_excel(loadDataPath).values[:, 0]
    daysLabelList = pd.read_pickle(daysLabelPath)
    dateListByClass = []
    for i in range(k_days):
        dateListByClass.append([])
    for i in range(len(daysLabelList)):
        dateListByClass[int(daysLabelList[i])].append(dateset[i].date())
    return dateListByClass