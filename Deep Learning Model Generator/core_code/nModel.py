"""
    Core Code for generating the Neural Network Model Currently focusing mainly on 
    Image and Video related Applications, ignoring NLP and Speech Recognition.
    
    Author : Shashank Dwivedi
    Date : 09/100/2020
    
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dense, Flatten, MaxPool2D, MaxPool1D, MaxPooling3D
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD, RMSprop, Nadam, Ftrl

class nModel():
    def __init__(self, params):
        self.model = Sequential()
        self.params = params
        
    def model_generator(self):
        print(self.params.convs)