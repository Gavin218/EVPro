# 该类用于存储训练函数
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import pickle

# data = []
# for i in range(10):
#     data.append([np.random.rand(672), np.random.rand(96)])

# 单纯使用seq2seq的预测函数，输入为处理好的pd文件
def train1(input_path, lr, modelClass, lossOutPath, modelOutPath):
    from Pretreatment import writeToExcel
    data = pd.read_pickle(input_path)
    theLen = len(data[0][0])
    record = []
    model = modelClass()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(100):
        for step, x in enumerate(data):

            x_origin = tf.constant(x[0], tf.float32)
            x_origin = tf.reshape(x_origin, [1, theLen, 1])
            y_real = tf.constant(x[1], tf.float32)
            y_real = tf.reshape(y_real, [1, 96, 1])

            with tf.GradientTape() as tape:
                y_pred = model(x_origin)
                y_pred = tf.reshape(y_pred, [1, 96, 1])
                ori_loss = tf.losses.mean_squared_error(y_pred=y_pred, y_true=y_real)
                rec_loss = tf.reduce_mean(ori_loss)
            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step%100 == 0:
                record.append([float(rec_loss)])

    writeToExcel(lossOutPath, record)
    model.save(modelOutPath)

# 联合其他因素的seq2seq的预测函数，（初步思路为编码后进行拼接）（此函数还未完成）
def train2(input_path, lr, modelClass1, modelClass2, lossOutPath, modelOutPath):
    from Pretreatment import writeToExcel
    data = pd.read_pickle(input_path)


    record = []
    model1 = modelClass1()
    model2 = modelClass2()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(100):
        for step, x in enumerate(data):

            x_origin = tf.constant(x[0], tf.float32)
            x_origin = tf.reshape(x_origin, [1, 672, 1])
            y_real = tf.constant(x[1], tf.float32)
            y_real = tf.reshape(y_real, [1, 96, 1])

            with tf.GradientTape() as tape:
                y_pred = model1(x_origin)
                y_pred = tf.reshape(y_pred, [1, 96, 1])
                ori_loss = tf.losses.mean_squared_error(y_pred=y_pred, y_true=y_real)
                rec_loss = tf.reduce_mean(ori_loss)
            grads = tape.gradient(rec_loss, model1.trainable_variables)
            optimizer.apply_gradients(zip(grads, model1.trainable_variables))
            if step%100 == 0:
                record.append([float(rec_loss)])

    writeToExcel(lossOutPath, record)
    model1.save(modelOutPath)

# 两个使用seq2seq的预测函数，在低维状态下进行拼接，输入为处理好的pd文件
def train3(input_path, lr, modelClass1, modelClass2, modelClass3, lossOutPath, modelOutPath):
    from Pretreatment import writeToExcel
    data = pd.read_pickle(input_path)
    theLen1 = len(data[0][0])
    theLen2 = len(data[0][1])

    model1 = modelClass1()
    model2 = modelClass2()
    model3 = modelClass3()
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    record = []
    for epoch in range(100):
        for step, x in enumerate(data):

            x_origin1 = tf.constant(x[0], tf.float32)
            x_origin1 = tf.reshape(x_origin1, [1, theLen1, 1])

            x_origin2 = tf.constant(x[1], tf.float32)
            x_origin2 = tf.reshape(x_origin2, [1, theLen2, 1])

            y_real = tf.constant(x[2], tf.float32)
            y_real = tf.reshape(y_real, [1, 96, 1])

            with tf.GradientTape(persistent=True) as tape:
                low_data1 = model1(x_origin1)
                low_data2 = model2(x_origin2)

                low_data = tf.concat([low_data1, low_data2], axis=1)

                y_pred = model3(low_data)
                y_pred = tf.reshape(y_pred, [1, 96, 1])
                ori_loss = tf.losses.mean_squared_error(y_pred=y_pred, y_true=y_real)
                rec_loss = tf.reduce_mean(ori_loss)
            grads1 = tape.gradient(rec_loss, model1.trainable_variables)
            optimizer.apply_gradients(zip(grads1, model1.trainable_variables))
            grads2 = tape.gradient(rec_loss, model2.trainable_variables)
            optimizer.apply_gradients(zip(grads2, model2.trainable_variables))
            grads3 = tape.gradient(rec_loss, model3.trainable_variables)
            optimizer.apply_gradients(zip(grads3, model3.trainable_variables))
            del tape
            if step%100 == 0:
                record.append([float(rec_loss)])

    writeToExcel(lossOutPath, record)
    model1.save(modelOutPath + "model1")
    model2.save(modelOutPath + "model2")
    model3.save(modelOutPath + "model3")

def AEtrain(modelClass, excelPath, lr, batchsz, epochNum, recordOutputPath, modelSavePath):
    from Pretreatment import writeToExcel
    from Pretreatment import excel_to_matrix
    import time
    dataList = excel_to_matrix(excelPath, 0)
    length = len(dataList[0])
    tf_data = tf.cast(dataList, dtype=tf.float32)
    train_db = tf.data.Dataset.from_tensor_slices(tf_data)
    train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
    model = modelClass()
    optimizer = keras.optimizers.Adam(lr=lr)
    record = []
    startTime = time.time()
    for epoch in range(epochNum):
        for step, x in enumerate(train_db):
            with tf.GradientTape() as tape:
                x_rec_logits = model(x)
                # rec_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
                # rec_loss = tf.reduce_mean(rec_loss1)
                orirec_loss = tf.losses.mean_squared_error(x, x_rec_logits)
                rec_loss = tf.reduce_mean(orirec_loss)
            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                # 间隔性打印训练误差
                print(epoch, step, float(rec_loss))
                record.append([epoch, step, float(rec_loss)])
    endTime = time.time()
    print("训练总用时{second}秒".format(second = endTime - startTime))
    writeToExcel(recordOutputPath, record)
    print("误差记录已存至指定路径！")
    model.save(modelSavePath)
    print("模型已存至指定路径")
    return 0