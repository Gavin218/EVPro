# 该py文件用于存储用到的模型

from tensorflow import keras


# 用时序卷积构建自编码器
class EVModel1(keras.Model):
    def __init__(self):
        super(EVModel1, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Conv1D(32, 2, padding='causal', dilation_rate=1, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=3),
            keras.layers.Conv1D(32, 2, padding='causal', dilation_rate=2, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=3),
            keras.layers.Conv1D(64, 2, padding='causal', dilation_rate=4, activation='relu'),
            keras.layers.AveragePooling1D(),
            keras.layers.Conv1D(64, 2, padding='causal', dilation_rate=4, activation='relu'),
            keras.layers.AveragePooling1D(),
            keras.layers.Flatten(),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(10, activation='relu')
        ])

        self.decoder = keras.Sequential([
            keras.layers.Dense(324, activation='relu'),
            keras.layers.Dense(768, activation='relu'),
            keras.layers.Reshape((1, 12, 64)),
            keras.layers.Conv2DTranspose(32, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
            keras.layers.Reshape((24, 32)),
            keras.layers.UpSampling1D(size=2),
            keras.layers.Reshape((1, 48, 32)),
            keras.layers.Conv2DTranspose(1, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        h = self.decoder(h)
        return h



# 用时序卷积构建的编码器
class EVModel21(keras.Model):
    def __init__(self):
        super(EVModel21, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Conv1D(32, 2, padding='causal', dilation_rate=1, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=3),
            keras.layers.Conv1D(32, 2, padding='causal', dilation_rate=2, activation='relu'),
            keras.layers.AveragePooling1D(pool_size=3),
            keras.layers.Conv1D(64, 2, padding='causal', dilation_rate=4, activation='relu'),
            keras.layers.AveragePooling1D(),
            keras.layers.Conv1D(64, 2, padding='causal', dilation_rate=4, activation='relu'),
            keras.layers.AveragePooling1D(),
            keras.layers.Flatten(),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(10, activation='relu')
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        return h

# 用时序卷积构建的解码器
class EVModel22(keras.Model):
    def __init__(self):
        super(EVModel22, self).__init__()
        self.decoder = keras.Sequential([
            keras.layers.Dense(324, activation='relu'),
            keras.layers.Dense(768, activation='relu'),
            keras.layers.Reshape((1, 12, 64)),
            keras.layers.Conv2DTranspose(32, [1, 2], data_format='channels_last', strides=[1, 2],
                                         activation='relu'),
            keras.layers.Reshape((24, 32)),
            keras.layers.UpSampling1D(size=2),
            keras.layers.Reshape((1, 48, 32)),
            keras.layers.Conv2DTranspose(1, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
        ])

    def call(self, inputs, training=None):
        h = self.decoder(inputs)
        return h

class LSTMModel1(keras.Model):
    def __init__(self):
        super(LSTMModel1, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.LSTM(96)
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        return h

# 用时序卷积构建自编码器（训练数据为wordVec）
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
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(10, activation='relu')
        ])

        self.decoder = keras.Sequential([
            keras.layers.Dense(324, activation='relu'),
            keras.layers.Dense(768, activation='relu'),
            keras.layers.Reshape((1, 12, 64)),
            keras.layers.Conv2DTranspose(32, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
            keras.layers.Reshape((24, 32)),
            keras.layers.UpSampling1D(size=2),
            keras.layers.Reshape((1, 48, 32)),
            keras.layers.Conv2DTranspose(1, [1, 2], data_format='channels_last', strides=[1, 2], activation='relu'),
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        h = self.decoder(h)
        return h