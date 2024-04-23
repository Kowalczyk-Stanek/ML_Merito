import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, optimizers
import numpy as np
import time


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


def train_and_evaluate(optimizer_name, optimizer):
    print(f"Optymalizator: {optimizer_name}")
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(28 * 28,)),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()
    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    end_time = time.time()


    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Dokładność: {test_acc * 100:.2f}%")
    print(f"Czas: {end_time - start_time:.2f} sekundy")


optimizers_to_test = {
    'Adam': optimizers.Adam(),
    'RMSprop': optimizers.RMSprop(),
    'SGD with Momentum': optimizers.SGD(momentum=0.9),
    'SGD': optimizers.SGD(),
    'AdaGrad': optimizers.Adagrad()
}

for optimizer_name, optimizer in optimizers_to_test.items():
    train_and_evaluate(optimizer_name, optimizer)
