import numpy as np
import pandas as pd
import csv
from keras.layers import Input, Dense, Flatten, LSTM, Conv1D, Layer
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard


def write_log(callback, names, logs, batch_no):  # 写tensorboard
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def MaxMinNormalization(x):
    Min = K.min(x)
    Max = K.max(x)
    if Max == Min:
        return x - Min
    x = (x - Min) / (Max - Min)
    return x


def npMaxMinNormalization(x):
    Min = np.min(x)
    Max = np.max(x)
    if Max == Min:
        return x - Min
    x = (x - Min) / (Max - Min)
    return x


def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y-y_pred)))


# 自定义层，把Generator生成的(20天MidPrice平均值减第十天MidPrice的差)和(前十天的MidPrice做差的序列)拼在一起
# 作为Discriminator的输入
class to_sequence(Layer):
    def __init__(self, **kwargs):
        super(to_sequence, self).__init__(**kwargs)

    def build(self, input_shape=[(10, 12), (1,)]):  # input_shape[0]: (10, 12), input_shape[1]: (1,)
        assert isinstance(input_shape, list)
        super(to_sequence, self).build(input_shape)

    def call(self, inputs, **kwargs):  # inputs[0] -- input_features, inputs[1] -- pred
        assert isinstance(inputs, list)
        previous = inputs[0][:, :, 0][:, 1:]-inputs[0][:, :, 0][:, :-1]  # MidPrice_diff
        sequence = K.concatenate((previous, inputs[1]), axis=1)
        sequence = K.expand_dims(sequence, axis=2)
        sequence = MaxMinNormalization(sequence)
        # print(sequence.shape)
        return sequence

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], 10)


class GAN():
    def __init__(self):
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])


        # Build the generator
        self.generator = self.build_generator()

        input_features = Input(shape=(10, 12))
        nonscale_input = Input(shape=(10, 12))
        pred = self.generator(input_features)
        to_sequence.trainable = False
        sequence = to_sequence()([nonscale_input, pred])
        self.discriminator.trainable = False  # For the combined model we will only train the generator
        validity = self.discriminator(sequence)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([input_features, nonscale_input], [pred, validity])
        self.combined.compile(loss=[rmse, 'binary_crossentropy'], optimizer=optimizer, loss_weights=[1, 1])


    def build_generator(self):

        model = Sequential()

        model.add(LSTM(64, input_shape=(10, 12), return_sequences=True))
        model.add(LSTM(64, input_shape=(10, 12), return_sequences=False))
        model.add(Dense(20, activation='linear'))
        model.add(Dense(1, activation='linear'))

        model.summary()

        input = Input(shape=(10, 12))
        pred_Y = model(input)

        return Model(input, pred_Y)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv1D(filters=32, kernel_size=4, strides=1, input_shape=(10, 1)))
        model.add(Conv1D(filters=64, kernel_size=4, strides=1))
        model.add(Conv1D(filters=128, kernel_size=4, strides=1))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Flatten())

        model.summary()

        input = Input(shape=(10, 1))
        validity = model(input)

        return Model(input, validity)

    def train(self, epochs, batch_size=128):

        # Load the dataset
        X_train = np.load('Fin_X_GAN.npy')
        X_train_nonscale = np.load('Fin_X_GAN_nonscale.npy')
        Y_train = np.load('Fin_Y_GAN.npy')
        Y_pre = np.load('Fin_Y_pre.npy')

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        callback_discriminator = TensorBoard('./finallog')
        callback_discriminator.set_model(self.discriminator)
        callback_combined = TensorBoard('./finallog')
        callback_combined.set_model(self.combined)

        for epoch in range(epochs):

            names = ['d_loss', 'd_accuracy', '?', 'RMSE', 'G loss']

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            input_features = X_train[idx]
            nonscale_input = X_train_nonscale[idx]
            true_diff = Y_train[idx]
            true_diff = true_diff.reshape((-1, 1))
            previous = Y_pre[idx]
            true_sequence = np.concatenate((previous, true_diff), axis=1)
            for i in range(true_sequence.shape[0]):
                true_sequence[i] = npMaxMinNormalization(true_sequence[i])
            true_sequence = true_sequence[:, :, np.newaxis]

            # Generate a batch of predictions
            pred = self.generator.predict(input_features)
            pred.reshape((-1, 1))
            pred_sequence = np.concatenate((previous, pred), axis=1)
            for i in range(pred_sequence.shape[0]):
                pred_sequence[i] = npMaxMinNormalization(pred_sequence[i])
            pred_sequence = pred_sequence[:, :, np.newaxis]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(true_sequence, valid)
            d_loss_fake = self.discriminator.train_on_batch(pred_sequence, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([input_features, nonscale_input], [Y_train[idx], valid])

            # Print the progress
            print("%d [D loss: %.5f, acc.: %.2f%%] [G RMSE loss: %.5f, G loss: %.5f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
            loss = np.concatenate((d_loss, g_loss), axis=0)
            write_log(callback_combined, names, loss, epoch)


gan = GAN()
gan.train(epochs=20000, batch_size=1000)
X_test = np.load('Fin_test_GAN.npy')
test_data = np.array(pd.read_csv('test_data.csv').drop(['Date', 'Time'], axis=1).iloc[:, 1:])
pred = gan.generator.predict(X_test)
with open('GAN_Fin_features.csv', 'w') as fout:
    fieldnames=['caseid', 'midprice']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(142, 1000):
            writer.writerow({'caseid': str(i+1), 'midprice': str(pred[i][0]+np.mean(test_data[i*10:i*10+10, 0]))})
