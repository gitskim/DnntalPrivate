#Model using a focal loss function

import os
import cv2
import glob

import imgaug as aug
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.python.keras.layers import Input, Flatten, SeparableConv2D, BatchNormalization
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.python.keras.utils import to_categorical

from keras import backend as K
import tensorflow as tf

##### Defining the path of images #####

# Define path to the data directory
data_dir = '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped'

# Path to train directory
train_dir = os.path.join(data_dir, 'train')

# Path to validation directory
validation_dir = os.path.join(data_dir, 'val')

# Get the path to the positive and negative sub-directories
train_positive_dir = os.path.join(train_dir, 'cropped_positive_xrays')
train_negative_dir = os.path.join(train_dir, 'cropped_negative_xrays')
validation_positive_dir = os.path.join(validation_dir, 'cropped_positive_xrays')
validation_negative_dir = os.path.join(validation_dir, 'cropped_negative_xrays')

# Get the list of all the images
positives_cases_train = glob.glob(os.path.join(train_positive_dir, '*.jpg'))
negatives_cases_train = glob.glob(os.path.join(train_negative_dir, '*.jpg'))
positives_cases_val = glob.glob(os.path.join(validation_positive_dir, '*.jpg'))
negatives_cases_val = glob.glob(os.path.join(validation_negative_dir, '*.jpg'))

# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []

# List that are going to contain validation images data and the corresponding labels
valid_data = []
valid_labels = []

# Go through all the normal cases. The label for these cases will be 0
for img in negatives_cases_train:
    train_data.append((str(img),0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in negatives_cases_train:
    train_data.append((str(img), 1))

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# Shuffle the data 
train_data = train_data.sample(frac=1.).reset_index(drop=True)

# Get the counts for each class
cases_count = train_data['label'].value_counts()
print(cases_count)

####Preprocess the validation dataset####

# We will convert into a image with 3 channels.
# We will normalize the pixel values and resizing all the images to 224x224

# Negatives cases
for img in negatives_cases_val:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

# Positives cases
for img in positives_cases_val:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

# Convert the list into numpy arrays
valid_data = np.array(valid_data)
valid_labels = np.array(valid_labels)




# Augmentation sequence 
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=30)]) # roatation


# some constants(not truly though!) 

# dimensions to consider for the images
img_rows, img_cols, img_channels = 224,224,3
# batch size for training  
batch_size=16



def data_generator(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    nb_batches = int(np.ceil(n/batch_size))

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)
    batch_labels = np.zeros((batch_size,), dtype=np.float32)
    
    while True:
        # shuffle indices for the training data
        np.random.shuffle(indices)
            
        for i in range(nb_batches):
            # get the next batch 
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            
            # process the next batch
            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(data.iloc[idx]["image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = seq.augment_image(img)
                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                label = data.iloc[idx]["label"]
                
                batch_data[j] = img
                batch_labels[j] = label
            
            batch_data = preprocess_input(batch_data)
            yield batch_data, batch_labels

# training data generator 
train_data_gen = data_generator(train_data, batch_size)

def read_images(images, label):
    data = []
    for img in images:
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
        data.append(img)
    
    labels = [label]*len(data)
    data = np.array(data).astype(np.float32)
    data = preprocess_input(data)
    return data, labels



# Get fine-tuning/transfer-learning model
def get_fine_tuning_model(base_model, top_model, inputs, learning_type):
    if learning_type=='transfer_learning':
        print("Doing transfer learning")
        K.set_learning_phase(0)
        base_model.trainable = False
        features = base_model(inputs)
        outputs = top_model(features)
    else:
        print("Doing fine-tuning")
        base_model.trainable = True
        features = base_model(inputs)
        outputs = top_model(features)
    return Model(inputs, outputs)

# Get the base model
base_model = ResNet50(input_shape=(img_rows, img_cols, img_channels), include_top=False, weights='imagenet', pooling='avg')

# Define a top model: extra layers that we are going to add on top of our base network
feature_inputs = Input(shape=base_model.output_shape, name='top_model_input')
x = Dense(50, activation='relu', name='fc1')(feature_inputs)
x = Dropout(0.5,name='drop')(x)
outputs = Dense(1, activation='sigmoid', name='fc2')(x)
top_model = Model(feature_inputs, outputs, name='top_model')


# get model for tranfser learning
inputs = Input(shape=(img_rows, img_cols, img_channels))
model = get_fine_tuning_model(base_model, top_model, inputs, "transfer_learning")
model.summary()

# focal loss 
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

#Optimizer
optimizer = RMSprop(0.0001)

# compile the model and check it 
model.compile(loss=focal_loss(), optimizer=optimizer, metrics=['accuracy'])

# always use earlystopping
# the restore_best_weights parameter load the weights of the best iteration once the training finishes
es = EarlyStopping(patience=5, restore_best_weights=True)

# checkpoint to save model
chkpt = ModelCheckpoint(filepath="model2", save_best_only=True)

# number of training and validation steps for training and validation
nb_train_steps = int(np.ceil(len(train_data)/batch_size))

# number of epochs 
nb_epochs=100

# train the model 
history1 = model.fit_generator(train_data_gen, 
                              epochs=nb_epochs, 
                              steps_per_epoch=nb_train_steps, 
                              validation_data=(valid_data, valid_labels),
                              callbacks=[es,chkpt],
                              class_weight={0:1.0, 1:2.0})