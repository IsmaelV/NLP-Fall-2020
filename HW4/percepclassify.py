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
		self.stop_words = []
		self.my_features = []
		self.all_training_data = dict()
		self.my_pn_weights_vanilla = dict()
		self.my_td_weights_vanilla = dict()
		self.pn_bias_vanilla = 0
		self.td_bias_vanilla = 0
		self.my_pn_weights_averaged = dict()
		self.my_td_weights_averaged = dict()
		self.pn_bias_averaged = 0
		self.td_bias_averaged = 0
		self.my_pn_weights_loaded = dict()
		self.my_td_weights_loaded = dict()
		self.pn_bias_loaded = 0
		self.td_bias_loaded = 0

	def collect_attribute_types(self, threshold):
		self.my_features = set()
		counts = dict()
		for key in self.all_training_data.keys():
			processed_text = self.all_training_data[key][0]
			for word in processed_text:
				if word in self.stop_words:
					continue
				if word in counts:
					counts[word] += 1
				else:
					counts[word] = 1
		counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
		for key in counts:
			if key[1] >= threshold:
				self.my_features.add(key[0])
			if len(self.my_features) >= 1000:
				break
		self.my_features = list(self.my_features)
		return self.my_features

	def load_new(self, all_training_data, threshold, epochs, stopword_file=None):
		if stopword_file:
			self.stop_words_extracted(stopword_file)
		self.all_training_data = all_training_data
		self.collect_attribute_types(threshold)
		self.train(epochs)

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

	def train(self, epochs):
		self.my_pn_weights_vanilla = {k: 0 for k in self.my_features}
		self.my_td_weights_vanilla = {k: 0 for k in self.my_features}
		self.pn_bias_vanilla = 0
		self.td_bias_vanilla = 0
		self.my_pn_weights_averaged = {k: 0 for k in self.my_features}
		self.my_td_weights_averaged = {k: 0 for k in self.my_features}
		self.pn_bias_averaged = 0
		self.td_bias_averaged = 0
		my_counter = 1
		my_shuffled_keys = list(self.all_training_data.keys())
		for e in range(epochs):
			print("\r", "Epoch {} of {}".format(e+1, epochs), end="")
			random.shuffle(my_shuffled_keys)
			for k in my_shuffled_keys:
				my_line, pn, td = self.all_training_data[k]
				my_observations = dict()
				for w in my_line:
					if my_observations.get(w):
						my_observations[w] += 1
					else:
						my_observations[w] = 1
				my_pn_activation = self.activation_function(self.my_pn_weights_vanilla, my_observations, self.pn_bias_vanilla)
				my_td_activation = self.activation_function(self.my_td_weights_vanilla, my_observations, self.td_bias_vanilla)

				if (my_pn_activation * pn) <= 0:
					for obs_keys in my_observations:
						if self.my_pn_weights_vanilla.get(obs_keys) is not None:
							self.my_pn_weights_vanilla[obs_keys] += (my_observations[obs_keys] * pn)
						if self.my_pn_weights_averaged.get(obs_keys) is not None:
							self.my_pn_weights_averaged[obs_keys] += (my_observations[obs_keys] * pn * my_counter)
					self.pn_bias_vanilla += pn
					self.pn_bias_averaged += (pn * my_counter)
				if (my_td_activation * td) <= 0:
					for obs_keys in my_observations:
						if self.my_td_weights_vanilla.get(obs_keys) is not None:
							self.my_td_weights_vanilla[obs_keys] += (my_observations[obs_keys] * td)
						if self.my_td_weights_averaged.get(obs_keys) is not None:
							self.my_td_weights_averaged[obs_keys] += (my_observations[obs_keys] * td * my_counter)
					self.td_bias_vanilla += td
					self.td_bias_averaged += (td * my_counter)

				my_counter += 1

		# Update the averaged weights to proper weights
		for k in self.my_td_weights_averaged.keys():
			self.my_td_weights_averaged[k] = self.my_td_weights_vanilla[k] - (self.my_td_weights_averaged[k] / my_counter)
			self.my_pn_weights_averaged[k] = self.my_pn_weights_vanilla[k] - (self.my_pn_weights_averaged[k] / my_counter)
		self.pn_bias_averaged = self.pn_bias_vanilla - (self.pn_bias_averaged / my_counter)
		self.td_bias_averaged = self.td_bias_vanilla - (self.td_bias_averaged / my_counter)
		print()
		return

	@staticmethod
	def activation_function(weights, observations, b):
		activation = 0
		for key in observations:
			if not weights.get(key) or weights[key] == 0:
				continue
			activation += (weights[key] * observations[key])
		activation += b
		return activation

	def predict(self, observations, toggle=0):
		if toggle == 1:
			return (self.activation_function(self.my_pn_weights_vanilla, observations, self.pn_bias_vanilla),
				self.activation_function(self.my_td_weights_vanilla, observations, self.td_bias_vanilla))
		elif toggle == -1:
			return (self.activation_function(self.my_pn_weights_averaged, observations, self.pn_bias_averaged),
				self.activation_function(self.my_td_weights_averaged, observations, self.td_bias_averaged))

		return (self.activation_function(self.my_pn_weights_loaded, observations, self.pn_bias_loaded),
				self.activation_function(self.my_td_weights_loaded, observations, self.td_bias_loaded))

	def my_evaluation(self, all_dev_data):
		true_positives_vanilla = 0
		false_positives_vanilla = 0
		true_negatives_vanilla = 0
		false_negatives_vanilla = 0
		true_positives_averaged = 0
		false_positives_averaged = 0
		true_negatives_averaged = 0
		false_negatives_averaged = 0

		all_dev_keys = list(all_dev_data.keys())
		random.shuffle(all_dev_keys)

		# Perform evaluation
		for k in all_dev_keys:
			input_text, correct_pn, correct_td = all_dev_data[k]
			my_observations = dict()
			for w in input_text:
				if my_observations.get(w):
					my_observations[w] += 1
				else:
					my_observations[w] = 1
			pred_pn_vanilla, pred_td_vanilla = self.predict(my_observations, toggle=1)
			pred_pn_averaged, pred_td_averaged = self.predict(my_observations, toggle=-1)

			if pred_pn_vanilla > 0 and correct_pn > 0:
				true_positives_vanilla += 1
			elif pred_pn_vanilla > 0 and correct_pn < 0:
				false_positives_vanilla += 1
			elif pred_pn_vanilla < 0 and correct_pn > 0:
				false_negatives_vanilla += 1
			elif pred_pn_vanilla < 0 and correct_pn < 0:
				true_negatives_vanilla += 1

			if pred_td_vanilla > 0 and correct_td > 0:
				true_positives_vanilla += 1
			elif pred_td_vanilla > 0 and correct_td < 0:
				false_positives_vanilla += 1
			elif pred_td_vanilla < 0 and correct_td > 0:
				false_negatives_vanilla += 1
			elif pred_td_vanilla < 0 and correct_td < 0:
				true_negatives_vanilla += 1

			if pred_pn_averaged > 0 and correct_pn > 0:
				true_positives_averaged += 1
			elif pred_pn_averaged > 0 and correct_pn < 0:
				false_positives_averaged += 1
			elif pred_pn_averaged < 0 and correct_pn > 0:
				false_negatives_averaged += 1
			elif pred_pn_averaged < 0 and correct_pn < 0:
				true_negatives_averaged += 1

			if pred_td_averaged > 0 and correct_td > 0:
				true_positives_averaged += 1
			elif pred_td_averaged > 0 and correct_td < 0:
				false_positives_averaged += 1
			elif pred_td_averaged < 0 and correct_td > 0:
				false_negatives_averaged += 1
			elif pred_td_averaged < 0 and correct_td < 0:
				true_negatives_averaged += 1

		accuracy_vanilla = (true_positives_vanilla + true_negatives_vanilla) / (
			true_positives_vanilla + true_negatives_vanilla + false_negatives_vanilla + false_positives_vanilla)
		precision_vanilla = true_positives_vanilla / (true_positives_vanilla + false_positives_vanilla)
		recall_vanilla = true_positives_vanilla / (true_positives_vanilla + false_negatives_vanilla)
		fscore_vanilla = (precision_vanilla * recall_vanilla * 2) / (precision_vanilla + recall_vanilla)

		vanilla_string_result = "Vanilla - Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(precision_vanilla,
																								recall_vanilla,
																								fscore_vanilla,
																								accuracy_vanilla)

		accuracy_averaged = (true_positives_averaged + true_negatives_averaged) / (
			true_positives_averaged + true_negatives_averaged + false_negatives_averaged + false_positives_averaged)
		precision_averaged = true_positives_averaged / (true_positives_averaged + false_positives_averaged)
		recall_averaged = true_positives_averaged / (true_positives_averaged + false_negatives_averaged)
		fscore_averaged = (precision_averaged * recall_averaged * 2) / (precision_averaged + recall_averaged)

		average_string_result = "Average - Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(precision_averaged,
																						recall_averaged,
																						fscore_averaged,
																						accuracy_averaged)

		return vanilla_string_result + "\n" + average_string_result

	def homework_testing(self, input_text):
		prediction_pn, prediction_td = self.predict(input_text)
		result = ""
		result += "truthful " if prediction_td > 0 else "deceptive "
		result += "positive " if prediction_pn > 0 else "negative "
		return result

	def save(self, save_vanilla_name, save_averaged_name):
		vanilla_save_file = open(save_vanilla_name, 'w')
		all_vanilla_info = {"stop_words": self.stop_words, "my_pn_weights": self.my_pn_weights_vanilla,
					"pn_bias": self.pn_bias_vanilla, "my_td_weights": self.my_td_weights_vanilla,
					"td_bias": self.td_bias_vanilla}
		vanilla_save_file.write(json.dumps(all_vanilla_info))
		vanilla_save_file.close()

		averaged_save_file = open(save_averaged_name, 'w')
		all_averaged_info = {"stop_words": self.stop_words, "my_pn_weights": self.my_pn_weights_averaged,
							"pn_bias": self.pn_bias_averaged, "my_td_weights": self.my_td_weights_averaged,
							"td_bias": self.td_bias_averaged}
		averaged_save_file.write(json.dumps(all_averaged_info))
		averaged_save_file.close()
		return True

	def load_pretrained(self, model_file):
		self.stop_words = []

		f = open(model_file, 'r')
		data = json.load(f)
		f.close()
		self.stop_words = data["stop_words"]
		self.my_pn_weights_loaded = data["my_pn_weights"]
		self.my_td_weights_loaded = data["my_td_weights"]
		self.pn_bias_loaded = data["pn_bias"]
		self.td_bias_loaded = data["td_bias"]
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
		with open(final_files[i], "r") as test_file:
			line = preprocess_text(test_file.readline()).split()
			obs = dict()
			for w in line:
				if obs.get(w):
					obs[w] += 1
				else:
					obs[w] = 1
			both_labels = my_percep_classifier.homework_testing(obs)
			result_file.write(both_labels + final_files[i] + "\n")

	result_file.close()
