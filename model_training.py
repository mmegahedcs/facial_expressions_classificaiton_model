'''
Created on Dec 3, 2019

@author: Mohamed.Megahed
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
import itertools
from keras.preprocessing import image


batch_size = 128
num_epochs = 170
learning_rate=0.001

########################### Loading FER-2013 Dataset ########################
def load_fer2013(display_sample):
        data = pd.read_csv('D:\\Mine\\FER_Workshop\\cnn_model\\cnn_model\\fer2013.csv')
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),(48,48))
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        if display_sample == True:
            overview(25, data.sample(n=50)) 
        return faces, emotions

def extract_from_string(pixels):
        pixels = pixels.split(' ')
        pixels = np.array([int(i) for i in pixels])
        return np.reshape(pixels, (48, 48))

def overview(total_rows, df):
        fig = plt.figure(figsize=(10,10))
        idx = 0
        for i, row in df.iterrows():
            input_img = extract_from_string(row.pixels)
            ax = fig.add_subplot(10,10,idx+1)
            ax.imshow(input_img, cmap=plt.cm.get_cmap('gray'))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
            idx += 1
        plt.show()
        
#################### Data Normalization ##################
def preprocess_input(x):
        x = x.astype('float32')
        x = x / 255.0
        return x
    
#################### Splitting Training and Validation Data ##################
def split_data():
    faces, emotions = load_fer2013(True)
    faces = preprocess_input(faces)
    return train_test_split(faces, emotions,test_size=0.2)

#################### Data Augmentation ##################
def augment_training_data():
    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        rotation_range=2,
        zoom_range=0.1,
        horizontal_flip=True
        )
    return train_datagen

#################### Constructing CNN Model ##################
def construct_cnn():        
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
       
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer = Adam(lr = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

#################### Initializing Training Callbacks ##################
def init_callbacks():
    base_path = 'D:\\Mine\\FER_Workshop\\trained_weights\\'
    log_file_path = base_path + 'emotion1_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=50)
    reduce_lr =  ReduceLROnPlateau(monitor = "val_acc", factor = 0.1,mode='max', patience = 20, verbose = 1,cooldown = 0)
    trained_models_path = base_path + 'cnn1_model_weights'
    model_names = trained_models_path + '.{epoch:04d}--{val_loss:.4f}--{val_acc:.4f}.h5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,save_best_only=True)

    callbacks = [model_checkpoint, csv_logger, early_stop,reduce_lr]
    return callbacks

#################### Training CNN Model ##################
def train_model():
    xtrain,xtest,ytrain,ytest = split_data()
    model = construct_cnn()
    model.fit_generator(augment_training_data().flow(xtrain, ytrain,batch_size),
                            steps_per_epoch=len(xtrain) / batch_size,
                            epochs=num_epochs, verbose=1, callbacks= init_callbacks(),
                            validation_data=(xtest,ytest))

if __name__ == "__main__":
    cfg = tf.ConfigProto(device_count={'GPU': 0 , 'CPU': 56})
    sess = tf.Session(config=cfg)
    sess.graph.as_default()
    keras.backend.set_session(sess)
    train_model()
