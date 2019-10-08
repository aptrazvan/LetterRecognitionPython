import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D # to add pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np


def convolutional_model():
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(26, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

# create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/', batch_size=64)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('data/validation/', batch_size=64)
# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/', batch_size=64)

predict_it = datagen.flow_from_directory('data/predict/', batch_size=64)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(train_it)

model = convolutional_model()
model.fit_generator(train_it, steps_per_epoch=10, validation_data=val_it, validation_steps=8)
loss = model.evaluate_generator(test_it, steps=24)

print(loss)

# make a prediction
yhat = model.predict_generator(predict_it, steps=24)

print(yhat.shape)

for i in range(yhat.shape[0]):
	print(np.argmax(yhat[i]))
