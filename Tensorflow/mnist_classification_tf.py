# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#print(tf.reduce_sum(tf.random.normal([100, 100])))
#print(tf.random.normal([100, 100]))

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Scaling 
x_train, x_test = x_train / 255.0, x_test / 255.0

#N,D = x_train ; Nx28x28


# Make the data compitable with the model

#X = x_train.reshape(28*28)
#Y = y_train # can we convert this into one-hot encoder?

# Define the model
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
        ])

# define the cost
#model.compile(tf.keras.optimizers.Adam, # selection of learning rate 
#              tf.keras.losses.CategoricalCrossentropy, # multiple categories.; sparse_categorical_crossengtropy?
#              tf.keras.metrics.Accuracy # 
#              )

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# training the model
model.fit(x_train,
          y_train,
          validation_data=(x_test,y_test),
          epochs=5
          )

# evaluate the model.
test_loss, test_acc = model.evaluate(x_test,y_test)

# display the classification score
print("Test Loss:",test_loss)
print("Test Accuracy:",test_acc)


# predict the model

predictions = model.predict(x_test)

for i in range(5):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel("Actual Label"+str(y_test[i]))
    plt.title("Prediction:"+str(np.argmax(predictions[i])))
    plt.show()



