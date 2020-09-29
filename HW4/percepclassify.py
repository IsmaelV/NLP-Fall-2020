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
	parser.add_argument("path_to_model", metavar="model-path", help="Path to the file of model", type=str)
	parser.add_argument("path_to_input", metavar="train-data-path", help="Path to the file of train data", type=str)
	return parser.parse_args()


class PercepClassifier(object):

	def __init__(self):
		self.label_prior = {}
		self.word_given_label = {}
		self.stop_words = []

	def load_new(self, neg_paths, pos_paths, dec_paths, truth_paths, stopword_file=None):
		if stopword_file:
			self.stop_words_extracted(stopword_file)

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
				self.stop_words += preprocess_text(line).split()

		return self.stop_words

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
						words = preprocess_text(line).split()
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
						words = preprocess_text(line).split()
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
						words = preprocess_text(line).split()
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
						words = preprocess_text(line).split()
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

		# vocab = len(self.attribute_types)
		# for key in positive_words:
		# 	self.word_given_label[(key, "positive")] = (positive_words[key] + c) / (pos_num + (c * vocab))
		# for key in negative_words:
		# 	self.word_given_label[(key, "negative")] = (negative_words[key] + c) / (neg_num + (c * vocab))
		# for key in truth_words:
		# 	self.word_given_label[(key, "truth")] = (truth_words[key] + c) / (truth_num + (c * vocab))
		# for key in deception_words:
		# 	self.word_given_label[(key, "deception")] = (deception_words[key] + c) / (dec_num + (c * vocab))

		return

	def predict(self, text):
		words = preprocess_text(text).split()
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

	def my_evaluation(self, all_neg_paths, all_pos_paths, all_dec_paths, all_truth_paths):
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

	def homework_testing(self, input_text):
		prediction_result = self.predict(input_text)
		result = ""
		result += "truthful " if prediction_result["truth"] > prediction_result["deception"] else "deceptive "
		result += "positive " if prediction_result["positive"] > prediction_result["negative"] else "negative "
		return result

	def save(self, save_file_name):
		save_file = open(save_file_name, 'w')
		new_words_given_label = dict()
		for key in self.word_given_label.keys():
			new_words_given_label[key[0] + "," + key[1]] = self.word_given_label[key]
		all_info = {"label_prior": self.label_prior, "word_given_label": new_words_given_label, "stop_words": self.stop_words}
		save_file.write(json.dumps(all_info))
		save_file.close()
		return True

	def load_pretrained(self, model_file):
		self.label_prior = {}
		self.word_given_label = {}
		self.stop_words = []

		f = open(model_file, 'r')
		data = json.load(f)
		f.close()
		self.stop_words = data["stop_words"]
		self.label_prior = data["label_prior"]
		for key in data["word_given_label"]:
			new_key = tuple(key.split(','))
			self.word_given_label[new_key] = data["word_given_label"][key]
		return True


def preprocess_text(given_text):
	"""
	Get a string to process and make it all lowercase, remove punctuation (except apostrophe),
	and split it on spaces.
	:param given_text: Unprocessed string
	:return: List of processed words in the sentences
	"""
	text = given_text.lower()  																	# Lower all text
	text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  	# Remove punctuation
	return text


def get_all_leaf_files(path_to_check):
	all_leaf_files = []

	for root, dirs, files in os.walk(path_to_check):
		for file in files:
			if file.endswith(".txt") and "README" not in file:
				all_leaf_files.append(os.path.join(root, file))

	return all_leaf_files


if __name__ == "__main__":
	arguments = check_arguments()
	my_percep_classifier = PercepClassifier()
	my_percep_classifier.load_pretrained(arguments.path_to_model)

	result_file = open("percepoutput.txt", 'w')

	init_files = get_all_leaf_files(arguments.path_to_input)
	final_files = [x for x in init_files if "fold1" in x]
	for i in range(len(final_files)):
		with open(final_files[i], "r") as f:
			for line in f:
				both_labels = my_percep_classifier.homework_testing(line)
				result_file.write(both_labels + final_files[i] + "\n")

	result_file.close()
