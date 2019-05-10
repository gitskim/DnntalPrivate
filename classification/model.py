#For building the model_1
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet169, MobileNet
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


path_train='/tmp/dentist_AI/dentist_AI/cropped_train'
path_validation='/tmp/dentist_AI/dentist_AI/cropped_validation'


conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

#Transfert learning model
TARGET_SHAPE = 128 
BATCH_SIZE = 64
EPOCHS = 50

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()