from scipy import misc
import numpy as np
import data_process as dp
import scipy.ndimage
import matplotlib.pyplot as plt

#chops Rows and pads
def chopRows(y):
    lowestRow = len(y)
    highestRow = 0
    print(lowestRow)
    for i,row in enumerate(y):
        for j,val in enumerate(row) :       
            if(255 == y[i][j]):
                print(i)
                if(i < lowestRow ):
                    lowestRow = i
                if(i > highestRow):
                    #print('yes')
                    highestRow = i
    print('lowestRow : {}, highestRow: {}'.format(lowestRow, highestRow))
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


#face = misc.face()
#misc.imsave('face.png', face) # First we need to create the PNG file

#face = misc.imread('2.jpg')
#type(face)      

#face.shape, face.dtype
#plt.imshow(face)
#plt.show()
#print(type(face[0][0][0]))
#print(face)
#x = dp.load_image('pixelshot.jpeg')/255.0
def segment():
    x = dp.load_image('hand1.jpg')
    plt.imshow(x, cmap='Greys')
    plt.show()
    print(x)
# firstBlack = False
    for i,row in enumerate(x):
        for j,val in enumerate(row) :
            if val > 160.0:
                #print(val)
                #if not (firstBlack):
                #print('row: {}, col: {}'.format(i, j))
                #firstBlack = True
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

        print('found black : {}'.format(j))
        # for i,row in enumerate(x):
        #     for j,val in enumerate(row) :
        #         if not (firstBlack):
        #             if val == 255:
        #                 black = (i, j)
        #                 firstBlack = True
        #         # else:
        #         #     if val == 0:
        #         #         white = (i, j)
        #         #         x = [i:]
        
        #find white space starting from column index of first black, without touching the columns
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
            print('height : {}, width: {}'.format(len(y), len(y[0])))
            images.append(y)
            
        
    # for im in images:
    #     plt.imshow(im, cmap='Greys')
    #     plt.show()

    return images


    # for i,row in enumerate(x[:, x[black[1]:]):
    #     j = black[1]
    #     while(noWhite):
    #         if(x[ ] )


            #print('row: {}, col: {}'.format(row, col))

#our threshold is 160 
#plt.imshow(x, cmap='Greys')
#plt.show()
    # = val[0]
    # col = val[1]
    # rbg = val[2]
    #if (np.array_equal(rbg, [255, 255, 255])):
    #if(rbg.all() >= 0):
        #print('row: {}, col: {}'.format(row, col))

