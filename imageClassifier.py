#start
import tensorflow as tf
import numpy as np
from tensorflow import keras

#printing stuff
import matplotlib.pyplot as plt

#load a pre defined dataset
fashion_mnist= keras.datasets.fashion_mnist

#pull out data from dataset
(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()

#Show data
#print(train_labels[0])
#print(train_images[0])
#plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
#plt.show()

#Define our neural net structure
model= keras.Sequential([
    #input 28x28 flattened into 784x1
    keras.layers.Flatten(input_shape=(28,28)),

    #hidden layer is 128 deep, relu returns the value, or 0
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    #output 0-10 depending on clothing
    keras.layers.Dense(units=10, activation=tf.nn.softmax)

])
#Compile our model
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train our model, using our training data
model.fit(train_images,train_labels,epochs=5)

#test model
test_loss= model.evaluate(test_images,test_labels)

plt.imshow(test_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

#make predictions
predictions= model.predict(test_images)
print(predictions[0])

print(list(predictions[0]).index(max(predictions[0])))


print("What's up")