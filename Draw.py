import pandas as pd

def drawComparisonPic(oriData, modelPath, outputPath):
    from Pretreatment import drawPicturesAndSave
    import tensorflow as tf
    model = tf.saved_model.load(modelPath)
    for index in range(100):
        print(index)
        X_test = tf.cast(oriData[index][0], dtype=tf.float32)
        X_test = tf.reshape(X_test, [1, 672, 1])
        Y_pre = model(X_test)
        Y_true = tf.cast(oriData[index][1], dtype=tf.float32)
        Y_true = tf.reshape(Y_true, [1, 96])
        Y_pre = tf.reshape(Y_pre, [1, 96])
        # print(float(tf.losses.mean_squared_error(Y_true, Y_pre)))
        theLines = [Y_true[0].numpy(), Y_pre[0].numpy()]
        drawPicturesAndSave(theLines, outPath=outputPath + "pic{i}".format(i=index))
    return 0


oriData = pd.read_pickle("D:/Backup/桌面/EV/train_data")
modelPath = "D:/Backup/桌面/EV/lstmModel"
outputPath = "D:/Backup/桌面/EV/lstm_test_pictures/"

drawComparisonPic(oriData=oriData, modelPath=modelPath, outputPath=outputPath)


