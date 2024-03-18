# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

## Neural Network Model

![image](https://github.com/praveenst13/mnist-classification/assets/118787793/08209516-2c8f-43e1-a5d5-354ff6e009c9)


## DESIGN STEPS

### STEP 1: Load the dataset from the tensorflow library.
### STEP 2:Preprocess the dataset. MNIST dataset using to classify handwritten written digit.
### STEP 3: Create and train your model.
### STEP 4: plot the training loss, validation loss vs iteration plot.
### STEP 5: Test the model for your handwritten scanned images.

## PROGRAM

### Name:Praveen s
### Register Number:212222240077

```python
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
from tensorflow.keras.layers import Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape

X_test.shape

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
inputs1=keras.Input(shape=(28,28,1))
model.add(inputs1)
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=6,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)

metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Name:PRAVEEN S")
print("Reg no:212222240077")
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('1.png')

img = image.load_img('1.png')
img = image.load_img('1.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)



print(x_single_prediction)


plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

print("Name:PRAVEEN S")
print("Reg no:212222240077")





```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/praveenst13/mnist-classification/assets/118787793/c0a56197-1220-457e-ab0f-499beb770534)

![image](https://github.com/praveenst13/mnist-classification/assets/118787793/5059047e-fa89-49ef-a773-a9c57c567b02)


### Classification Report
![image](https://github.com/praveenst13/mnist-classification/assets/118787793/f540ee16-484b-4bea-a8a3-11bfd8efadd1)


### Confusion Matrix
![image](https://github.com/praveenst13/mnist-classification/assets/118787793/92591f1d-92e4-4f64-b284-54fe18288c8c)


### New Sample Data Prediction
![1](https://github.com/praveenst13/mnist-classification/assets/118787793/9386a720-d4b1-44a0-b87e-3356e2a3d73b)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
