#import statements from all our combined parts
import sys
import data_process
import base_case_classifier
import LNN


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
	if (model == 0): 
		# run LNN
		print("LNN output")
	# use base case classifier
	else:
		base_case_classifier.main()



#input handwriting to be converted output text
def norm(model):
	#get training data from neel
	trainingPictures
	trainingLabels

	#labels
	labels = list("0123456789abcdefghijklmnopqrstuvwxyz")
	#create and train bills seperation model
	#create and train toms letter classification model
	classifier = LNN(labels)
	classifier.train(trainingPictures,trainingLabels)
	#take input files and output
	while(1):
		file = input("hadwriten file to be converted")
		#neel does something with this file to create input image
		#bill seperates and puts in 2d array
		characters = #bills stuff

		#check which model to use
		#learned model
		if(model == 0):
			output = ""
			numWords = 0
			for w in characters:
				output += classifier.classify(w)
				#count words to know when to go to new
				numWords += 1
				if(numWords == 10):
					numWords = 0
					output += "\n"
				else:
					output += " "
			print output
		#basecase model
		else:	
			#use KTs base case and output
			


if __name__ == '__main__':
    main()