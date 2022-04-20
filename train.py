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
def train1(input_path, cut, train_data_outputPath, test_data_outputPath, lr, modelClass, lossOutPath, modelOutPath):
    from Pretreatment import writeToExcel
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
    record = []
    model = modelClass()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(100):
        for step, x in enumerate(data):

            x_origin = tf.constant(x[0], tf.float32)
            x_origin = tf.reshape(x_origin, [1, 672, 1])
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

# 联合其他因素的seq2seq的预测函数，（初步思路为编码后进行拼接）
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

