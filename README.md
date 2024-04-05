# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model
![image](https://github.com/Pravinrajj/mnist-classification/assets/117917674/c8ed1138-8e1f-489b-96eb-a89cd5b81918)

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.
### STEP 2:
Download and load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP 5:
Split the data into train and test
### STEP 6:
Build the convolutional neural network model
### STEP 7:
Train the model with the training data
### STEP 8:
Plot the performance plot
### STEP 9:
Evaluate the model with the testing data
### STEP 10:
Fit the model and predict the single input


## PROGRAM

```
### Name: YOHESH KUMAR R.M
### Register Number: 212222240118
```
```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
     
(X_train, y_train), (X_test, y_test) = mnist.load_data()
     
X_train.shape
     
X_test.shape
     
single_image= X_train[0]
     
single_image.shape
     
plt.imshow(single_image,cmap='gray')
     
y_train.shape

X_train.min()
     
X_train.max()
     
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
     
X_train_scaled.min()
     
X_train_scaled.max()
     
y_train[0]
     
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
     
type(y_train_onehot)
     
y_train_onehot.shape
     
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
     
y_train_onehot[500]
     
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add (layers.Input (shape=(28,28,1)))
model.add (layers.Conv2D (filters=32, kernel_size=(3,3), activation='relu'))
model.add (layers.MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers.Dense (32, activation='relu'))
model.add (layers.Dense (25, activation='relu'))
model.add (layers.Dense (10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("YOHESH KUMAR R.M : 212222240118")
print(confusion_matrix(y_test,x_test_predictions))
print("YOHESH KUMAR R.M : 212222240118")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('two.png')
type(img)
img = image.load_img('two.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print("YOHESH KUMAR R.M : 212222240118")
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-04-05 132253](https://github.com/yoheshkumar/mnist-classification/assets/119393568/9c5e583e-508b-4101-81e4-b58b4e741b5b)
![Screenshot 2024-04-05 132302](https://github.com/yoheshkumar/mnist-classification/assets/119393568/4fbb74bf-c9d4-42b9-a6ad-2f6e2cb235ff)




### Classification Report
![Screenshot 2024-04-05 132651](https://github.com/yoheshkumar/mnist-classification/assets/119393568/1581b5f9-b071-4abf-a1dd-c0f070fe8ff3)

### Confusion Matrix
![Screenshot 2024-04-05 132819](https://github.com/yoheshkumar/mnist-classification/assets/119393568/17ca62c8-83a0-42d9-ae95-fa0e095387c8)

### New Sample Data Prediction
#### Input:
![image8](https://github.com/yoheshkumar/mnist-classification/assets/119393568/374c5f73-5c68-4468-a058-48d6c226363c)

#### Output:
![Screenshot 2024-04-05 133658](https://github.com/yoheshkumar/mnist-classification/assets/119393568/076486d3-1c32-49f0-9529-7b4eaa8f5592)


![Screenshot 2024-04-05 133745](https://github.com/yoheshkumar/mnist-classification/assets/119393568/ecc02009-d84c-4a02-a112-6bb2a22f0efb)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
