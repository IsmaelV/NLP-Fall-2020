import re
import argparse


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_train_data", metavar="train-data-path", help="Path to the file of train data", type=str)
	parser.add_argument("path_to_test_data", metavar="test-data-path", help="Path to the file of test data", type=str)
	return parser.parse_args()


def write_results():
	result_text = open("lookup-output.txt", "w")
	for key in training_counts.keys():
		result_text.write(key + ": " + str(training_counts[key]) + "\n")
		print(key + ": " + str(training_counts[key]))
	result_text.close()


def populate_lemma_count(train_data_file):
	train_data = open(train_data_file, 'r', encoding="utf8")

	for line in train_data:
		# Tab character identifies lines containing tokens
		if re.search('\t', line):
			# Tokens represented as tab-separated fields
			field = line.strip().split('\t')

			# Word form in second field, lemma in third field
			form = field[1]
			lemma = field[2]

			if form not in lemma_count.keys():
				lemma_count[form] = {lemma: 1}
			else:
				lemma_count[form][lemma] = lemma_count[form].get(lemma, 0) + 1


def populate_lemma_max_and_training_counts():
	for form in lemma_count.keys():
		all_lemmas = lemma_count[form]

		# Populate lookup table (lemma_max)
		highest_lemma = max(all_lemmas, key=lambda key: all_lemmas[key])
		lemma_max[form] = highest_lemma

		# Populate training counts
		training_counts["Wordform types"] += 1
		all_lemma_count = 0
		for lemma in all_lemmas:
			all_lemma_count += all_lemmas[lemma]
			if lemma == form:
				training_counts["Identity tokens"] += all_lemmas[lemma]
		training_counts["Wordform tokens"] += all_lemma_count

		if len(all_lemmas) == 1:
			training_counts["Unambiguous types"] += 1
			training_counts["Unambiguous tokens"] += all_lemma_count
		else:
			training_counts["Ambiguous types"] += 1
			training_counts["Ambiguous tokens"] += all_lemma_count
			training_counts["Ambiguous most common tokens"] += all_lemmas[highest_lemma]


if __name__ == "__main__":
	arguments = check_arguments()

	# Paths for data are read from command line
	train_file = arguments.path_to_train_data
	test_file = arguments.path_to_test_data

	# Counters for lemmas in the training data: word form -> lemma -> count
	lemma_count = dict()

	# Lookup table learned from the training data: word form -> lemma
	lemma_max = dict()

	# Variables for reporting results
	# If wordform only has one lemma, then unambiguous
	training_stats = ['Wordform types', 'Wordform tokens', 'Unambiguous types', 'Unambiguous tokens', 'Ambiguous types',
					'Ambiguous tokens', 'Ambiguous most common tokens', 'Identity tokens']
	training_counts = dict.fromkeys(training_stats, 0)

	test_outcomes = ['Total test items', 'Found in lookup table', 'Lookup match', 'Lookup mismatch',
					'Not found in lookup table', 'Identity match', 'Identity mismatch']
	test_counts = dict.fromkeys(test_outcomes, 0)

	accuracies = dict()

	populate_lemma_count(train_file)
	populate_lemma_max_and_training_counts()
	write_results()
