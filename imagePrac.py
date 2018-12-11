from scipy import misc
import numpy as np
import data_process as dp
#face = misc.face()
#misc.imsave('face.png', face) # First we need to create the PNG file

#face = misc.imread('2.jpg')
#type(face)      

#face.shape, face.dtype
import matplotlib.pyplot as plt
#plt.imshow(face)
#plt.show()
#print(type(face[0][0][0]))
#print(face)
#x = dp.load_image('pixelshot.jpeg')/255.0
x = dp.load_image('hand1.jpg')
plt.imshow(x, cmap='Greys')
plt.show()
print(x[0][0])
for val in x:
    row = val[0]
    col = val[1]
    rbg = val[2]
    #if (np.array_equal(rbg, [255, 255, 255])):
    #if(rbg.all() >= 0):
        #print('row: {}, col: {}'.format(row, col))

