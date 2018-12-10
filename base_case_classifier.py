"""
This program classifies letters and digits based on the average
darkness of the image. 

When given an input image, this classifier returns whatever letter 
or digit most resembles the input image based on whichever 
classification from the training data has the closest average
darkness.

This method is to serve as a base case, being a non-ideal way of
classifying letters and digits, to serve as a comparison for our
neural network implementation. 

The following is based on the code provided by mnielsen available at
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_average_darkness.py

"""

from collections import defaultdict

import tensorflow as tf



def main():
    # load data here, however we're loading data
    training_data, test_data = tf.keras.datasets.mnist.load_data()

    # use training data to compute averages
    avgs = avg_darkness(training_data)

    # classify test images
    num_correct = sum(int(guess_char(image, avgs) == char)\
                    for image, char in zip(test_data[0], test_data[1]))

    print ("Baseline classifier using average darkness of image.")
    print ("{0} of {1} values correct.".format(num_correct, len(test_data[1])))

""" Returns a dictionary whose keys are digits 0 through 61
representing digits 0 through 9 then uppercase and lowercase 
letters of the alphabet.
For each character in the training_data set we compute the average darkness of
training images containing that character.
The average darkness of an image is represented by the sum
of the darknesses for each pixel in the image.
"""
def avg_darkness(training_data):
    char_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, char in zip(training_data[0], training_data[1]):
        char_counts[char] += 1
        darknesses[char] += sum(image)
    avgs = defaultdict(float)
    for char, n in char_counts.items():
        avgs[char] = darknesses[char] / n
    return avgs

""" Returns the character whose average darkness in the training data
is closest to the darkness of image. avgs is assumed to be
a defaultdict whose keys are 0...61 and whose values are the
corresponding average darknesses across the training data.
"""
def guess_char(image, avgs):
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.items()}
    min_val = min(distances.values())
    result = [key for key, value in distances.items() if value == min_value]
    return result

if __name__ == "__main__":
    main()
