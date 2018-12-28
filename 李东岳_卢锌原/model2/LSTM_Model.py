import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from tensorflow import keras
import tensorflow as tf

class Model():
    """A class for an building and inferencing an lstm model"""
    
    def __init__(self, name):
        self.model = keras.models.Sequential()
        self.name = name

    def save_model(self):
        self.model.save(self.name + ".h5")

    def load_model(self):
        self.model = keras.models.load_model(self.name + ".h5")
        self.model.summary()

    def build_model(self):
        self.model.add(keras.layers.LSTM(
            units = 10, 
            input_shape = (10, 7), 
            return_sequences = True,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.LSTM(
            units = 10,
            return_sequences = True,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.LSTM(
            units = 10,
            return_sequences = False,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(
            10, 
            activation = "relu",
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(1, activation = "linear"))

        self.model.compile(
            loss="mse", 
            optimizer="adam",
            metrics=['accuracy', 'mse']
        )



    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            keras.callbacks.ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        
        # learning rate schedule
        # def step_decay(epoch):
        #     initial_lrate = 0.1
        #     drop = 0.5
        #     epochs_drop = 10.0
        #     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        #     return lrate

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            # keras.callbacks.LearningRateScheduler(step_decay),
            keras.callbacks.ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )
        
        print('[Model] Training Completed. Model saved as %s' % save_fname)


    def predict_data(self, data):
        predicted = self.model.predict(data)
        return predicted