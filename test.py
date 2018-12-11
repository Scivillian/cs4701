
#needs to be downloaded
#pip install mlxtend
from mlxtend.data import loadlocal_mnist


from LNN import LNN
import numpy as np
import tensorflow as tf

#loads x -> the training images
#loads y -> the training labels
x, y = loadlocal_mnist(
	images_path='data/emnist-byclass-train-images-idx3-ubyte',
	labels_path='data/emnist-byclass-train-labels-idx1-ubyte')

#x = x[0:90000]
#y = y[0:90000]
#697932
x = np.reshape(x,(697932,28,28))


print(x.shape)
print(len(y))

lables = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
nn = LNN(lables)
nn.train(x,y)

