#import statements from all our combined parts
import sys
import data_processor
import base_case_classifier
import LNN
import numpy as np


def main():
	if(len(sys.argv) == 2):
		if(sys.argv[1] == '-h'):
			print("CS4701 final project")
			print("main.py [-h | -b | -l][-d | -n]")
			print("example: main.py -l -n")
			print("-h : help menu")
			print("-b : use base case")
			print("-l : use learned model")
			print("-d : diagnostic mode(train and run test data)")
			print("-n : normal mode(given a picture output answer)")
		else:
			print("main.py: invalid command try -h for help")
	elif(len(sys.argv) == 3):
		if(sys.argv[1] == '-b' and sys.argv[2] == '-d'):
			print("running in diagnostic mode with base case model")
			diag(1)
		elif(sys.argv[1] == '-b' and sys.argv[2] == '-n'):
			print("running in normal mode with base case model")
			norm(1)
		elif(sys.argv[1] == '-l' and sys.argv[2] == '-d'):
			print("running in diagnostic mode with learned model")
			diag(0)
		elif(sys.argv[1] == '-l' and sys.argv[2] == '-n'):
			print("running in normal mode with learned model")
			norm(0)
		else:
			print("main.py: invalid command try -h for help")
	else:
		print("main.py: invalid command try -h for help")


#running the model on a ton of test examples and graphing and such
def diag(model):
	training_data, test_data = data_processor.load_datasets(0)
	if (model == 0): 
		# run LNN
		labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
		classifier = LNN.LNN(labels)
		classifier.train(training_data[0], training_data[1])
		output = classifier.classify(test_data[0])
		#print(output[:10])
		#print(test_data[1][:10])
		num_correct = sum(int(output[i] == labels[test_data[1][i]]) for i in range(len(output)))
		print ("Neural network classifier")
		print ("{0} of {1} values correct.".format(num_correct, len(test_data[1])))
	else:
		# use base case classifier
		base_case_classifier.classify_dataset(training_data, test_data)



#input handwriting to be converted output text
def norm(model):
	#get training data from neel
	training_data, test_data = data_processor.load_datasets(0)

	#labels
	labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
	#create and train bills seperation model
	#create and train toms letter classification model
	if (model == 0):
		classifier = LNN.LNN(labels)
		classifier.train(training_data[0],training_data[1])
	#take input files and output
	while(1):
		file = input("hadwriten file to be converted: \n")
		image = data_processor.load_image(file)
		characters = np.array(data_processor.segment(image))

		#check which model to use
		#learned model
		if(model == 0):
			output = ""
			output = str(classifier.classify(characters))
			print (output)
		#basecase model
		else:	
			avgs = base_case_classifier.avg_darkness(training_data)
			output = ""
			numchars = 0
			for img in characters:
				output += labels[(base_case_classifier.guess_char(img, avgs))[0]]
				numchars += 1
				if (numchars == 10):
					numchars = 0
					output += "\n"
				else:
					output += " "
			print(output)
		

			


if __name__ == '__main__':
    main()