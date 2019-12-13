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
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
import itertools
from keras.preprocessing import image

batch_size = 128
num_epochs = 170
learning_rate=0.001

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


def load_existing_model_weights():
    model = construct_cnn()
    import os
    from pathlib import Path
    user_home = str(Path.home())
    model.load_weights(os.path.join(user_home, "model_weights.h5"))
    return model
 
def startCapturing():
    expressions_category = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    model = load_existing_model_weights()
    graph = tf.get_default_graph()
    import os
    from pathlib import Path
    user_home = str(Path.home())
    ####### copy haarcascade_frontalface_default.xml file to your user home directory
    face_cascade = cv2.CascadeClassifier(os.path.join(user_home, "haarcascade_frontalface_default.xml"))
    webcam = cv2.VideoCapture(0)
    while True:
        rval, frame = webcam.read()
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_index = 0
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            detected_face = cv2.resize(face, (48, 48))
            x = image.img_to_array(detected_face)
            x = np.expand_dims(x, axis=0)
            x /= 255
            with graph.as_default():
                detected_expression = model.predict(x)[0]
                for index, emotion in enumerate(expressions_category):
                    cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), lineType=cv2.LINE_AA)
                    cv2.rectangle(frame, (70, index * 20 + 10), (70 + int(detected_expression[index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
        
        winname = ""
        cv2.imshow(winname, frame)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    cfg = tf.ConfigProto(device_count={'GPU': 0 , 'CPU': 56})
    sess = tf.Session(config=cfg)
    sess.graph.as_default()
    keras.backend.set_session(sess)
    startCapturing()
