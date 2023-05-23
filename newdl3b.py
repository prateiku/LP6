# Importing the required libraries
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow import keras
import numpy as np

# Loading the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Visualizing the first image in the training set
plt.imshow(x_train[1])

# Visualizing the second image in the training set
plt.imshow(x_train[0])

# Preprocessing the data
x_train = x_train.astype('float32') / 255.0 
x_test = x_test.astype('float32') / 255.0

# Reshaping the input images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Checking the shape of the data arrays
x_train.shape  # (60000, 28, 28)
x_test.shape   # (10000, 28, 28, 1)
y_train.shape  # (60000,)
y_test.shape   # (10000,)

# Defining the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), 
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128, (3,3), activation='relu'), 
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax')
])

# Printing the summary of the model
model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
