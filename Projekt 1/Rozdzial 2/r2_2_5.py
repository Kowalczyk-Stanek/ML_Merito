import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, optimizers
import numpy as np
import time

# Definicja optymalizatorów do przetestowania
optimizers_to_test = {
    'Adam': optimizers.Adam(),
    'RMSprop': optimizers.RMSprop(),
    'SGD with Momentum': optimizers.SGD(momentum=0.9),
    'SGD': optimizers.SGD(),
    'AdaGrad': optimizers.Adagrad()
}

# Załadowanie danych MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Redukcja poziomów szarości z 256 na 16
train_images = (train_images // 16).astype("uint8")
test_images = (test_images // 16).astype("uint8")

# Przygotowanie danych treningowych
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 15  # Normalizacja do przedziału [0, 1]
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 15  # Normalizacja do przedziału [0, 1]

# Definicja modelu sieci neuronowej
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(10, activation='softmax')
])

# Dla każdego optymalizatora testujemy szybkość, dokładność i czas treningu
for optimizer_name, optimizer in optimizers_to_test.items():
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Pomiar czasu treningu
    start_time = time.time()
    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Ocena modelu na zbiorze testowym
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Dokładność optymlizatora {optimizer_name}: {test_acc * 100:.2f}%")
    print(f"Czas: {training_time:.2f} sekund")
    print("---------------------------------------")

