import pandas as pd

def drawComparisonPic(oriData, modelPath, outputPath):
    from Pretreatment import drawPicturesAndSave
    import tensorflow as tf
    model = tf.saved_model.load(modelPath)
    theLen = len(oriData[0][0])
    for index in range(100):
        print(index)
        X_test = tf.cast(oriData[index][0], dtype=tf.float32)
        X_test = tf.reshape(X_test, [1, theLen, 1])
        Y_pre = model(X_test)
        Y_true = tf.cast(oriData[index][1], dtype=tf.float32)
        Y_true = tf.reshape(Y_true, [1, 96])
        Y_pre = tf.reshape(Y_pre, [1, 96])
        # print(float(tf.losses.mean_squared_error(Y_true, Y_pre)))
        theLines = [Y_true[0].numpy(), Y_pre[0].numpy()]
        drawPicturesAndSave(theLines, outPath=outputPath + "pic{i}".format(i=index))
    return 0

def drawComparisonPic2(oriData, modelPath, outputPath):
    from Pretreatment import drawPicturesAndSave
    import tensorflow as tf
    model1 = tf.saved_model.load(modelPath + "model1")
    model2 = tf.saved_model.load(modelPath + "model2")
    model3 = tf.saved_model.load(modelPath + "model3")
    theLen1 = len(oriData[0][0])
    theLen2 = len(oriData[0][1])
    for index in range(100):
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
        Y_true = tf.reshape(Y_true, [1, 96])
        Y_pre = tf.reshape(Y_pre, [1, 96])
        # print(float(tf.losses.mean_squared_error(Y_true, Y_pre)))
        theLines = [Y_true[0].numpy(), Y_pre[0].numpy()]
        drawPicturesAndSave(theLines, outPath=outputPath + "pic{i}".format(i=index))
    return 0


# oriData = pd.read_pickle("D:/Backup/桌面/EV/train_data")
modelPath = "D:/Backup/桌面/EV/lstmModel"
outputPath = "D:/Backup/桌面/EV/lstm_test_pictures/"

# oriData = pd.read_pickle("D:/桌面/relatedFile/test_data_wordVet")
modelPath = "D:/桌面/relatedFile/model1"
outputPath = "D:/桌面/relatedFile/word2Vec_test/"

# drawComparisonPic(oriData=oriData, modelPath=modelPath, outputPath=outputPath)

oriData = pd.read_pickle("D:/桌面/relatedFile/test_data_twoFactors")
modelPath = "D:/桌面/relatedFile/two_factors_model/"
outputPath = "D:/桌面/relatedFile/two_factors_test_pic/"
drawComparisonPic2(oriData, modelPath, outputPath)
