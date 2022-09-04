#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D

def nn(num_classes):
    """Multi-layer perceptron.
    """
    model = Sequential()
    model.add(Dense(256, input_dim=193))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def cnn(num_classes):
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten

    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    model = Sequential()
    model.add(Conv1D(64, 3, input_shape=(193, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 3))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                    optimizer='SGD',
                    metrics=['acc'])

    return model





def cnn2d(num_classes):
    from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten
    model = Sequential()
    # Conv Layer #1
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=(193, 193, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis=1))
    # Conv Layer #2
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3,3))
    # Conv Layer #3
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3,3))
    # Conv Layer #4
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3,3))
    # Flatten
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])

    model.summary()
    
    return model
# %%
