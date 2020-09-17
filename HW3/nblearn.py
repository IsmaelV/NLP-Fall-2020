import argparse
import os
import string


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_input", metavar="train-data-path", help="Path to the file of train data", type=str)
	return parser.parse_args()


def get_data_directories(root_path):
	"""
	Grab the directories and give them back
	:return: All directories that we want to see
	"""
	negative_path = root_path
	positive_path = root_path
	for i in os.listdir(root_path):
		if "positive" in i:
			positive_path += i + '/'
		elif "negative" in i:
			negative_path += i + '/'

	neg_decept = negative_path
	neg_truth = negative_path
	pos_decept = positive_path
	pos_truth = positive_path

	for i in os.listdir(negative_path):
		if "decept" in i:
			neg_decept += i + '/'
		elif "truth" in i:
			neg_truth += i + '/'
	for i in os.listdir(positive_path):
		if "decept" in i:
			pos_decept += i + '/'
		elif "truth" in i:
			pos_truth += i + '/'

	return neg_decept, neg_truth, pos_decept, pos_truth


def split_train_dev(given_path):
	"""
	Split the data that we will use into training and development
	:param given_path: String of path to use to see and split
	:return: List of fold paths to train with and string of fold path to test with
	"""
	return [given_path + x + '/' for x in os.listdir(given_path) if "fold" in x][1:], \
		[given_path + x + '/' for x in os.listdir(given_path) if "fold" in x][0]


def preprocess_text(given_text):
	"""
	Get a string to process and make it all lowercase, remove punctuation (except apostrophe),
	and split it on spaces.
	:param given_text: Unprocessed string
	:return: List of processed words in the sentences
	"""
	text = given_text.lower()  																	# Lower all text
	text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  	# Remove punctuation
	words = []
	for word in text.split():
		words.append(word)
	return words


if __name__ == "__main__":
	arguments = check_arguments()
	neg_decept_path, neg_truth_path, pos_decept_path, pos_truth_path = get_data_directories(arguments.path_to_input)

	neg_decept_train, neg_decept_dev = split_train_dev(neg_decept_path)
	neg_truth_train, neg_truth_dev = split_train_dev(neg_truth_path)
	pos_decept_train, pos_decept_dev = split_train_dev(pos_decept_path)
	pos_truth_train, pos_truth_dev = split_train_dev(pos_truth_path)
