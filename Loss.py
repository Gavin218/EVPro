import pandas as pd

def calculateMAPE(oriData, modelPath):
    import numpy as np
    import tensorflow as tf
    model = tf.saved_model.load(modelPath)
    theLen = len(oriData[0][0])
    loss_list = []
    for index in range(len(oriData)):
        print(index)
        X_test = tf.cast(oriData[index][0], dtype=tf.float32)
        X_test = tf.reshape(X_test, [1, theLen, 1])
        Y_pre = model(X_test)
        Y_true = tf.cast(oriData[index][1], dtype=tf.float32)
        Y_true = tf.reshape(Y_true, [1, 96]).numpy()
        Y_pre = tf.reshape(Y_pre, [1, 96]).numpy()
        one_loss = np.mean(np.abs((Y_true - Y_pre) / Y_true))
        if one_loss != np.inf:
            loss_list.append(np.mean(np.abs((Y_true - Y_pre) / Y_true)))
    return np.mean(loss_list)

def calculateMAPE_two_factors(oriData, modelPath):
    import tensorflow as tf
    import numpy as np
    model1 = tf.saved_model.load(modelPath + "model1")
    model2 = tf.saved_model.load(modelPath + "model2")
    model3 = tf.saved_model.load(modelPath + "model3")
    theLen1 = len(oriData[0][0])
    theLen2 = len(oriData[0][1])
    loss_list = []
    for index in range(len(oriData)):
        print(index)
        X_test1 = tf.cast(oriData[index][0], dtype=tf.float32)
        X_test1 = tf.reshape(X_test1, [1, theLen1, 1])
        low_data1 = model1(X_test1)

        X_test2 = tf.cast(oriData[index][1], dtype=tf.float32)
        X_test2 = tf.reshape(X_test2, [1, theLen2, 1])
        low_data2 = model2(X_test2)

        low_data = tf.concat([low_data1, low_data2], axis=1)
        Y_pre = model3(low_data)

        Y_true = tf.cast(oriData[index][2], dtype=tf.float32)
        Y_true = tf.reshape(Y_true, [1, 96]).numpy()
        Y_pre = tf.reshape(Y_pre, [1, 96]).numpy()
        one_loss = np.mean(np.abs((Y_true - Y_pre) / Y_true))
        if one_loss != np.inf:
            loss_list.append(np.mean(np.abs((Y_true - Y_pre) / Y_true)))
    return np.mean(loss_list)

result1 = calculateMAPE(pd.read_pickle("D:/Backup/桌面/EV/test_data"), "D:/Backup/桌面/EV/model2")
result2 = calculateMAPE_two_factors(pd.read_pickle("D:/Backup/桌面/EV/test_data_twoFactors"), "D:/Backup/桌面/EV/two_factors_model/")
print(result1, result2)
