import argparse
import os
import string
import glob
import math
import random
import json


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_input", metavar="train-data-path", help="Path to the file of train data", type=str)
	return parser.parse_args()


class NbClassifier(object):

	def __init__(self):
		self.attribute_types = set()
		self.label_prior = {}
		self.word_given_label = {}
		self.stop_words = []

	def load_new(self, all_training_paths, neg_paths, pos_paths, dec_paths, truth_paths, stopword_file=None):
		if stopword_file:
			self.stop_words_extracted(stopword_file)

		self.collect_attribute_types(all_training_paths, 2)
		self.train(neg_paths, pos_paths, dec_paths, truth_paths)

	def stop_words_extracted(self, path_to_stopwords):
		"""
		Read stopwords and keep in list
		:param path_to_stopwords: String of path to stopwords
		:return: List of stopwords
		"""
		self.stop_words = []
		with open(path_to_stopwords, 'r') as f:
			for line in f:
				self.stop_words += preprocess_text(line)

		return self.stop_words

	def collect_attribute_types(self, all_paths, threshold):
		self.attribute_types = set()
		seen = {}
		for path in all_paths:
			all_files = glob.glob(path + "*.txt")
			for train_file in all_files:
				with open(train_file, 'r') as f:
					for line in f:
						processed_text = preprocess_text(line)
						for word in processed_text:
							if word in seen:
								seen[word] += 1
							else:
								seen[word] = 1
		for key in seen:
			if seen[key] >= threshold:
				self.attribute_types.add(key)
		return self.attribute_types

	def train(self, all_neg_paths, all_pos_paths, all_dec_paths, all_truth_paths):
		self.label_prior = {}
		self.word_given_label = {}
		positive_words = {}
		pos_num = 0
		negative_words = {}
		neg_num = 0
		deception_words = {}
		dec_num = 0
		truth_words = {}
		truth_num = 0
		c = 0.6

		for neg_path in all_neg_paths:
			neg_files = glob.glob(neg_path + "*.txt")
			for neg_file in neg_files:
				with open(neg_file, 'r') as f:
					for line in f:
						words = preprocess_text(line)
						neg_num += len(words)
						for word in words:
							if word in negative_words and word not in self.stop_words:
								negative_words[word] += 1
							elif word not in negative_words and word not in self.stop_words:
								negative_words[word] = 1
							if word not in positive_words and word not in self.stop_words:
								positive_words[word] = 0
		for pos_path in all_pos_paths:
			pos_files = glob.glob(pos_path + "*.txt")
			for pos_file in pos_files:
				with open(pos_file, 'r') as f:
					for line in f:
						words = preprocess_text(line)
						pos_num += len(words)
						for word in words:
							if word in positive_words and word not in self.stop_words:
								positive_words[word] += 1
							elif word not in positive_words and word not in self.stop_words:
								positive_words[word] = 1
							if word not in negative_words and word not in self.stop_words:
								negative_words[word] = 0
		for dec_path in all_dec_paths:
			dec_files = glob.glob(dec_path + "*.txt")
			for dec_file in dec_files:
				with open(dec_file, 'r') as f:
					for line in f:
						words = preprocess_text(line)
						dec_num += len(words)
						for word in words:
							if word in deception_words and word not in self.stop_words:
								deception_words[word] += 1
							elif word not in deception_words and word not in self.stop_words:
								deception_words[word] = 1
							if word not in truth_words and word not in self.stop_words:
								truth_words[word] = 0
		for truth_path in all_truth_paths:
			truth_files = glob.glob(truth_path + "*.txt")
			for truth_file in truth_files:
				with open(truth_file, 'r') as f:
					for line in f:
						words = preprocess_text(line)
						truth_num += len(words)
						for word in words:
							if word in truth_words and word not in self.stop_words:
								truth_words[word] += 1
							elif word not in truth_words and word not in self.stop_words:
								truth_words[word] = 1
							if word not in deception_words and word not in self.stop_words:
								deception_words[word] = 0

		total_pos_neg = pos_num + neg_num
		self.label_prior["positive"] = pos_num / total_pos_neg
		self.label_prior["negative"] = neg_num / total_pos_neg

		total_truth_dec = truth_num + dec_num
		self.label_prior["truth"] = truth_num / total_truth_dec
		self.label_prior["deception"] = dec_num / total_truth_dec

		vocab = len(self.attribute_types)
		for key in positive_words:
			self.word_given_label[(key, "positive")] = (positive_words[key] + c) / (pos_num + (c * vocab))
		for key in negative_words:
			self.word_given_label[(key, "negative")] = (negative_words[key] + c) / (neg_num + (c * vocab))
		for key in truth_words:
			self.word_given_label[(key, "truth")] = (truth_words[key] + c) / (truth_num + (c * vocab))
		for key in deception_words:
			self.word_given_label[(key, "deception")] = (deception_words[key] + c) / (dec_num + (c * vocab))

	def predict(self, text):
		words = preprocess_text(text)
		prediction = dict()

		prediction["positive"] = math.log(self.label_prior["positive"], 10)
		prediction["negative"] = math.log(self.label_prior["negative"], 10)
		prediction["truth"] = math.log(self.label_prior["truth"], 10)
		prediction["deception"] = math.log(self.label_prior["deception"], 10)

		# Positive calculations
		for word in words:
			if (word, "positive") not in self.word_given_label:
				continue
			prediction["positive"] += math.log(self.word_given_label[(word, "positive")], 10)
		# Negative calculations
		for word in words:
			if (word, "negative") not in self.word_given_label:
				continue
			prediction["negative"] += math.log(self.word_given_label[(word, "negative")], 10)
		# Truth calculations
		for word in words:
			if (word, "truth") not in self.word_given_label:
				continue
			prediction["truth"] += math.log(self.word_given_label[(word, "truth")], 10)
		# Deception calculations
		for word in words:
			if (word, "deception") not in self.word_given_label:
				continue
			prediction["deception"] += math.log(self.word_given_label[(word, "deception")], 10)

		return prediction

	def evaluate(self, all_neg_paths, all_pos_paths, all_dec_paths, all_truth_paths):
		true_positives = 0
		false_positives = 0
		true_negatives = 0
		false_negatives = 0

		all_files = []

		# ---------------------------------
		# Populate all_files for evaluation
		# ---------------------------------
		for neg_path in all_neg_paths:
			neg_files = glob.glob(neg_path + "*.txt")
			for neg_file in neg_files:
				all_files.append((neg_file, "negative"))
		for pos_path in all_pos_paths:
			pos_files = glob.glob(pos_path + "*.txt")
			for pos_file in pos_files:
				all_files.append((pos_file, "positive"))
		for dec_path in all_dec_paths:
			dec_files = glob.glob(dec_path + "*.txt")
			for dec_file in dec_files:
				all_files.append((dec_file, "deception"))
		for truth_path in all_truth_paths:
			truth_files = glob.glob(truth_path + "*.txt")
			for truth_file in truth_files:
				all_files.append((truth_file, "truth"))

		random.shuffle(all_files)

		# Perform evaluation
		for file_name, correct_tag in all_files:
			with open(file_name, 'r') as f:
				for line in f:
					prediction_result = self.predict(line)

					if correct_tag == "truth" or correct_tag == "deception":  	# Truth/Deception classifier
						if prediction_result["truth"] > prediction_result["deception"]:
							if correct_tag == "truth":
								true_positives += 1
							else:
								false_positives += 1
						else:
							if correct_tag == "truth":
								false_negatives += 1
							else:
								true_negatives += 1
					else:  														# Positive/Negative classifier
						if prediction_result["positive"] > prediction_result["negative"]:
							if correct_tag == "positive":
								true_positives += 1
							else:
								false_positives += 1
						else:
							if correct_tag == "positive":
								false_negatives += 1
							else:
								true_negatives += 1

		accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_negatives + false_positives)
		precision = true_positives / (true_positives + false_positives)
		recall = true_positives / (true_positives + false_negatives)
		fscore = (precision * recall * 2) / (precision + recall)

		return precision, recall, fscore, accuracy

	def save(self, save_file_name):
		save_file = open(save_file_name, 'w')
		new_words_given_label = dict()
		for key in self.word_given_label.keys():
			new_words_given_label[key[0] + "," + key[1]] = self.word_given_label[key]
		all_info = {"attribute_types": list(self.attribute_types), "label_prior": self.label_prior,
					"word_given_label": new_words_given_label, "stop_words": self.stop_words}
		save_file.write(json.dumps(all_info))
		save_file.close()
		return True

	def load_pretrained(self, model_file):
		self.attribute_types = set()
		self.label_prior = {}
		self.word_given_label = {}
		self.stop_words = []

		f = open(model_file, 'r')
		data = json.load(f)
		f.close()
		self.attribute_types = set(data["attribute_types"])
		self.stop_words = data["stop_words"]
		self.label_prior = data["label_prior"]
		for key in data["word_given_label"]:
			new_key = tuple(key.split(','))
			self.word_given_label[new_key] = data["word_given_label"][key]
		return True


def print_results(data):
	print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*data))
	return


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

	all_train_paths = neg_decept_train + neg_truth_train + pos_decept_train + pos_truth_train

	n_paths = neg_decept_train + neg_truth_train
	p_paths = pos_decept_train + pos_truth_train
	d_paths = neg_decept_train + pos_decept_train
	t_paths = neg_truth_train + pos_truth_train

	my_nb_classifier = NbClassifier()
	my_nb_classifier.load_new(all_train_paths, n_paths, p_paths, d_paths, t_paths, stopword_file="./data/stopwords-mini.txt")
	my_nb_classifier.save("./nbmodel.txt")

