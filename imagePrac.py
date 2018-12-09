from scipy import misc
import numpy as np
#face = misc.face()
#misc.imsave('face.png', face) # First we need to create the PNG file

face = misc.imread('2.jpg')
#type(face)      

face.shape, face.dtype
import matplotlib.pyplot as plt
plt.imshow(face)
plt.show()
#print(type(face[0][0][0]))
print(face)
""
for val in face:
    row = val[0]
    col = val[1]
    rbg = val[2]
    if (np.array_equal(rbg, [255, 255, 255])):
        print(rbg)

