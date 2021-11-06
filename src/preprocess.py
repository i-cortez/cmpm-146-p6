from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt

train_dir = 'cats_and_dogs_small/train/'
test_dir = 'cats_and_dogs_small/test/'
validation_dir = 'cats_and_dogs_small/validation/'

# Create the data generators (each should be an instance of ImageDataGenerator)
# Rescale all images from the [0...255] range to the [0...1] range
# Use the ImageDataGenerator class to intialize two generators
train_datagen = ImageDataGenerator(
    # Original images consist in RGB coefficients in the 0-255 range
    # target values between 0 and 1 by scaling with a 1./255 factor
    rescale = 1./255
)
test_datagen = ImageDataGenerator(
    # Original images consist in RGB coefficients in the 0-255 range
    # target values between 0 and 1 by scaling with a 1./255 factor
    rescale = 1./255
)

# Call flow_from_directory on each of your datagen objects
train_generator = train_datagen.flow_from_directory(
    train_dir, # target directory
    target_size = (150, 150), # resize images
    batch_size = 20,
    class_mode='binary' # specify binary lables
)
test_generator = test_datagen.flow_from_directory(
    test_dir, # target directory
    target_size = (150, 150), # resize images
    batch_size = 20,
    class_mode='binary' # specity binary lables
)
validation_generator = None  # TODO: Student

# Usage Example:
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


def show_example_images(datagen):
    """
    Use this function to show some example images from the data generator.

    :param datagen: The data generator.
    """
    fnames = [os.path.join(train_cats_dir, fname) for
        fname in os.listdir(train_cats_dir)]
    img_path = fnames[3]            # Chooses one image to augment
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)     # Converts it to a Numpy array with shape (150, 150, 3) 
    x = x.reshape((1,) + x.shape)   # Reshapes it to (1, 150, 150, 3)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()