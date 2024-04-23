import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import numpy as np


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

activation_functions = ['tanh', 'relu', 'elu', 'leaky_relu', 'sigmoid']

for activation in activation_functions:
    print(f"Testing with {activation} activation:")
    

    model = models.Sequential([
        layers.Dense(32, activation=activation, input_shape=(28 * 28,)),
        layers.Dense(10, activation='softmax')
    ])


    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)


    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
   


    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)


    correct_predictions = (predicted_labels == test_labels)
    accuracy = np.mean(correct_predictions)
    print(f"Średnia dokładność: {accuracy * 100:.2f}%")

    std_deviation = np.std(correct_predictions)
    print(f"Średnie Odchylenie standardowe: {std_deviation:.4f}\n")
