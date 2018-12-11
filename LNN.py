#import statements
import tensorflow as tf
from tensorflow import keras

import numpy as np
#import matplotlib.pyplot as plt

#def main():
	#training_data, test_data = tf.keras.datasets.mnist.load_data()

	#network = LNN(training_data[1]) # training data character labels
	#network.train(training_data[0], training_data[1])



#Letter Neural Net
class LNN(object):
    #CONSTRUCTOR
    #param cNames: the list of lables ex. index 0 => '0', index 10 => 'A'
    #param dimX (int): x dimemsion of input images
    #param dimY (int): y dimension of input images
    #param l1S (int): #of nodes in layer 1
    #param l2S (int): #of nodes in layer 2
    #param l1A (string): activation function of layer 1
    #param l2A (string): activation function of layer 2
    #param fA (string): acitivation function of the output layer
    #param opt (string): the optimization function
    #param lossF (string):the loss function
    def __init__(self,cNames, dimX=28, dimY=28, l1S=128, l2S=128, l1A=tf.nn.sigmoid ,l2A='sigmoid',fA=tf.nn.sigmoid, opt=tf.train.AdamOptimizer(), lossF='sparse_categorical_crossentropy'):
        
        self.labels = cNames
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(dimX, dimY)),
            keras.layers.Dense(l1S, activation=l1A),
            keras.layers.Dense(len(cNames), activation=fA)
        ])
        self.model.compile(optimizer=opt, 
              loss=lossF,
              metrics=['accuracy'])


    #Train the model
    #param trainImages: images to be classified
    #param trainLabels: what each image is
    #param epochs: the number of epochs to carry out
    #param batch_size: how to split up input data
    def train(self, trainImages, trainLabels, epochs=5,batch_size=32):
        print("Training Letter Neural Net")
        self.model.fit(trainImages,trainLabels,epochs=epochs)
        print("done")


    #classify input 
    #param letters: the letters to be classified
    #returns string with classification of inputs in the same order
    def classify(self,letters):
        pred = self.model.predict(letters)
        output = ""
        for p in pred:
            output+= (self.labels[np.argmax(p)])
        return output