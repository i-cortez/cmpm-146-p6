from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np

# Load the features you created in feature_extraction.py
train_features = np.load("results/train_features.npy")
train_labels = np.load("results/train_labels.npy")
validation_features = np.load("results/validation_features.npy")
validation_labels = np.load("results/validation_labels.npy")
test_features = np.load("results/test_features.npy")
test_labels = np.load("results/test_labels.npy")

# Train the classifier from the features
model = models.Sequential()
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

history = model.fit(
    train_features,
    train_labels,
    epochs=10,
    batch_size=20,
    validation_data=(validation_features, validation_labels)
)

model.summary()

model.save('cats_and_dogs_small_best.h5')

# Plot the results
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()