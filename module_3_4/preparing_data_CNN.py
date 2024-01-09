import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

pixel_width = 28
pixel_height = 28

num_of_classes = 10

(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

features_train = features_train.reshape(features_train.shape[0], pixel_width, pixel_height, 1)
features_test = features_test.reshape(features_test.shape[0], pixel_width, pixel_height, 1)

input_shape = (pixel_width, pixel_height, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

print(features_train[0])

features_train /= 255
features_test /= 255

labels_train = keras.utils.to_categorical(labels_train, num_of_classes)
labels_test = keras.utils.to_categorical(labels_test, num_of_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
print("POST CONV2D", model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print("POST MaxPooling2D", model.output_shape)
model.add(Dropout(0.25))
print("POST Dropout", model.output_shape)
