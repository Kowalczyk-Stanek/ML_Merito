import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import numpy as np


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = (train_images // 16).astype("uint8")
test_images = (test_images // 16).astype("uint8")

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 15 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 15


def create_model(dropout_rate):
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(28 * 28,)),
        layers.Dropout(dropout_rate),  
        layers.Dense(10, activation='softmax')
    ])
    return model


dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for dropout_rate in dropout_rates:
    print(f"Dropout: {dropout_rate * 100}%")
    

    model = create_model(dropout_rate)
    

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    

    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    results.append(test_acc)
    
    print(f"Średnia dokładność: {test_acc * 100:.2f}%")


