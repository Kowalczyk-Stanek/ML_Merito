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


activation_functions = ['tanh', 'relu', 'elu', 'leaky_relu', 'sigmoid']


def train_model(activation):
    model = models.Sequential([
        layers.Dense(32, activation=activation, input_shape=(28 * 28,)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return test_loss, test_acc


results = {}

for activation in activation_functions:
    test_loss, test_acc = train_model(activation)
    results[activation] = {'test_loss': test_loss, 'test_acc': test_acc}


print("Wyniki testowania dla różnych funkcji aktywacji:")
for activation, result in results.items():
    print(f"Funkcja aktywacji: {activation}")
    print(f"Średnia dokładność: {result['test_acc'] * 100:.2f}%")
    print(f"Średnie odchylenie standardowe: {result['test_loss']:.4f}\n")
