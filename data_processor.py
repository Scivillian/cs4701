
#needs to be downloaded
#pip install mlxtend
from mlxtend.data import loadlocal_mnist

from LNN import LNN
import numpy as np
import tensorflow as tf
from PIL import Image

from scipy import misc
import scipy.ndimage
import matplotlib.pyplot as plt


"""Loads in the MNIST training and testing datasets and returns them"""
def load_datasets(mode):
    training_data = []
    test_data = []
    if(mode == 1):
        x1, y1 = loadlocal_mnist(
            images_path='data/emnist-byclass-train-images-idx3-ubyte',
            labels_path='data/emnist-byclass-train-labels-idx1-ubyte')
        x2, y2 = loadlocal_mnist(
            images_path='data/emnist-byclass-test-images-idx3-ubyte',
            labels_path='data/emnist-byclass-test-labels-idx1-ubyte')
        
        training_data = [x1, y1]#[:60000]]
        test_data = [x2, y2]#[:60000]]
    else:
        trd, ted = tf.keras.datasets.mnist.load_data()
        training_data = list(trd)
        test_data = list(ted)
    # un-flatten the image data into an array of 28x28 image pixel values
    #training_data[0] = np.reshape(training_data[0],(len(training_data[0]),28,28))
    #test_data[0] = np.reshape(test_data[0], (len(test_data[0]), 28, 28))
    x = training_data[0]#[:60000]
    y = test_data[0]#[:60000]

    training_data[0] = np.reshape(x,(len(x),28,28))
    test_data[0] = np.reshape(y, (len(y), 28, 28))

    return (training_data, test_data)


"""Loads in an image file, resizes it, and converts it to an array of
grayscale values """
def load_image(infilename) :
    img = Image.open(infilename)
    img = img.resize((500, 500), Image.ANTIALIAS)
    img.load()
    data = np.asarray( img, dtype="int32" )

    trimmed = data[:, :, [0, 1,2]]

    mult = np.array((.30,.59,.11))

    scaled = trimmed *mult[np.newaxis]

    result = np.sum(scaled, axis = 2)
    result = np.absolute(result - 255.0)
    return result

"""Takes as input x, an array of pixel values representing a grayscale image,
segments the image into characters, and returns a list of image arrays of
each character"""
def segment(x):
    plt.imshow(x, cmap='Greys')
    plt.show()
    for i,row in enumerate(x):
        for j,val in enumerate(row) :
            if val > 160.0:
                x[i][j] = 255
            else:
                x[i][j] = 0

    plt.imshow(x, cmap='Greys')
    plt.show()

    firstBlack = False
    images = []
    while(j < len(x[0]) ):
        j = 0
        firstBlack = False
        while(not firstBlack and (j < len(x[0])) ):
            if(255 in x[:,j]):
                firstBlack = True
            else: 
                j+=1

        left = j
        blackExists = True
        #j = black[1]
        while(blackExists and (j < len(x[0]))):
            if(255 in x[:,j]):
                j+=1
            else: 
                blackExists = False

        if((j < len(x[0])) ):
            y = x[:, left:j]
            x = x[:, j:]
        #print(i)
            y = chopRows(y)
            if(len(y) > 2 and len(y[0]) > 2):
                y = scipy.misc.imresize(y, 28/len(y))
            blurred_f = scipy.ndimage.gaussian_filter(y, 3)
            filter_blurred_f = scipy.ndimage.gaussian_filter(blurred_f, 1)
            alpha = 30
            sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
            images.append(y)
            
    return images

"""chops Rows and pads with white pixels so that dimensions are equal"""
def chopRows(y):
    lowestRow = len(y)
    highestRow = 0
    #print(lowestRow)
    for i,row in enumerate(y):
        for j,val in enumerate(row) :       
            if(255 == y[i][j]):
                #print(i)
                if(i < lowestRow ):
                    lowestRow = i
                if(i > highestRow):
                    #print('yes')
                    highestRow = i
    y = y[lowestRow-1:highestRow+2, :]

    height = len(y) 
    width = len(y[0])
    m = max(height, width)
    equalizer = abs(height - width)
    if( equalizer%2 == 0):
        split_equalizer1 = equalizer//2
        split_equalizer2 = equalizer//2
    else:
        split_equalizer1 = equalizer//2
        split_equalizer2 = equalizer//2 + 1
    
    padding = m//10



    if( m == width):
        y = np.pad(y, ((padding+split_equalizer1,padding+split_equalizer2),(padding,padding)), 'constant')
    else:
        y = np.pad(y, ((padding,padding),(padding+split_equalizer1,padding+split_equalizer2)), 'constant')
    return y

