import numpy as np
import _pickle as cPickle
import gzip
from PIL import Image


def load_data():
    #resource: https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
 
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_image(infilename) :

    img = Image.open(infilename)
    img = img.resize((280, 280), Image.ANTIALIAS)
    img.load()
    data = np.asarray( img, dtype="int32" )
    #data_vector = data.ravel()

    trimmed = data[:, :, [0, 1,2]]

    #print(trimmed)

    mult = np.array((.30,.59,.11))

    scaled = trimmed *mult[np.newaxis]

    #print(scaled)

    result = np.sum(scaled, axis = 2)

    result = np.absolute(result - 255.0)



    #print(result)

    #print(result)
    
    #result.ravel()

    return result

#load_image("pm3.jpeg")

# def load_image(infilename):
# 	image = Image.open('sample.png')
# 	width, height = image.size
# 	pixels = image.load()

# 	# Check if has alpha, to avoid "too many values to unpack" error
# 	has_alpha = len(pixels[0,0]) == 4

# 	# Create empty 2D list
# 	fill = 1
# 	array = [[fill for x in range(width)] for y in range(height)]

# 	for y in range(height):
# 	    for x in range(width):
# 	        if has_alpha:
# 	            r, g, b, a = pixels[x,y]
# 	        else:
# 	            r, g, b = pixels[x,y]
# 	        lum = 255-((r+g+b)/3) # Reversed luminosity
# 	        array[y][x] = lum/255 # Map values from range 0-255 to 0-1
