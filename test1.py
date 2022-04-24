from tensorflow import keras
import tensorflow as tf
import numpy as np


# 用时序卷积构建自编码器
class EVModel3(keras.Model):
    def __init__(self):
        super(EVModel3, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Conv1D(32, 5, padding='causal', dilation_rate=1, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=10),
            keras.layers.Conv1D(32, 5, padding='causal', dilation_rate=2, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=10),
            keras.layers.Conv1D(64, 5, padding='causal', dilation_rate=4, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=5),
            keras.layers.Conv1D(64, 5, padding='causal', dilation_rate=4, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=5),
            keras.layers.Flatten(),
            # keras.layers.Dense(500, activation='relu'),
            # keras.layers.Dense(10, activation='relu')
        ])

        # self.decoder = keras.Sequential([
        #     keras.layers.Dense(324, activation='relu'),
        #     keras.layers.Dense(768, activation='relu'),
        #     keras.layers.Reshape((1, 12, 64)),
        #     keras.layers.Conv2DTranspose(32, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
        #     keras.layers.Reshape((24, 32)),
        #     keras.layers.UpSampling1D(size=2),
        #     keras.layers.Reshape((1, 48, 32)),
        #     keras.layers.Conv2DTranspose(1, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
        # ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        # h = self.decoder(h)
        return h



def classTest(CNNclass, dataSet):
    model = CNNclass()
    model.build(input_shape=(1, 42600, 1))
    # model.build(input_shape=(1, 140, 1))
    model.summary()
    Y_total = []
    for oneData in dataSet:
        X_test = tf.cast(oneData, dtype=tf.float32)
        X_test = tf.reshape(X_test, [1, 42600, 1])
        Y_test = model(X_test)
        Y_total.append(Y_test)
        shape_list = [X_test.shape, Y_test.shape]
        print(Y_test)
        print(X_test.shape, Y_test.shape)
        break;
data = []
for i in range(10):
    data.append(np.random.rand(42600))
classTest(EVModel3, data)
