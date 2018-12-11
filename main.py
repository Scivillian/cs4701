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
	training_data, test_data = data_process.load_data()
	if (model == 0): 
		# run LNN
		print("LNN output")
	# use base case classifier
	else:
		base_case_classifier.classify_dataset(training_data, test_data)



#input handwriting to be converted output text
def norm(model):
	#get training data from neel
	training_data, test_data = data_process.load_data()

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
		image = data_process.load_image(file)
		characters = data_process.segment(image)

		#check which model to use
		#learned model
		if(model == 0):
			output = ""
			numchars = 0
			for w in characters:
				output += classifier.classify(w)
				#count words to know when to go to new
				numchars += 1
				if(numchars == 10):
					numchars = 0
					output += "\n"
				else:
					output += " "
			print output
		#basecase model
		else:	
			avgs = base_case_classifier.avg_darkness(training_data)
			output = ""
			numchars = 0
			for img in characters:
				output += base_case_classifier.guess_char(img, avgs)
				numchars += 1
				if (numchars == 10):
					numchars = 0
					output += "\n"
				else:
					output += " "

		

			


if __name__ == '__main__':
    main()