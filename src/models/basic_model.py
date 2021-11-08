from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...
# A CNN using some number of convolutional and maxpooling
model.add(layers.Conv2D(32, (7, 7), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, (7, 7), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (7, 7), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# A single flatten layer
model.add(layers.Flatten())
model.add(layers.Dense(64))
# A hidden densely connected layer
model.add(layers.Activation("relu"))
# A final densely connected layer that outputs a single number
model.add(layers.Dense(1))
model.add(layers.Activation("sigmoid"))


# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py