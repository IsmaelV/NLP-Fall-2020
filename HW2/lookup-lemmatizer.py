"""
Simple Lemmatizer
a big lookup table, which maps every word form attested in the training data to the most common lemma associated with
that form. At test time, the program checks if a form is in the lookup table, and if so, it gives the associated lemma;
if the form is not in the lookup table, it gives the form itself as the lemma (identity mapping).

File is based off start file given by Prof. Artstein

:author: Ismael Villegas-Molina
"""
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
	result_text.write("Training statistics\n")
	for key in training_counts.keys():
		result_text.write(key + ": " + str(training_counts[key]) + "\n")
		print(key + ": " + str(training_counts[key]))
	print("Expected lookup:", str(accuracies["Expected lookup"]))
	result_text.write("Expected lookup accuracy: " + str(accuracies["Expected lookup"]) + "\n")
	print("Identities lookup:", str(accuracies["Identities lookup"]))
	result_text.write("Expected identity accuracy: " + str(accuracies["Identities lookup"]) + "\n")

	print()
	result_text.write("Test results\n")
	for key in test_counts.keys():
		result_text.write(key + ": " + str(test_counts[key]) + "\n")
		print(key + ": " + str(test_counts[key]))
	for key in accuracies.keys():
		if key == "Expected lookup" or key == "Identities lookup":
			continue
		result_text.write(key + ": " + str(accuracies[key]) + "\n")
		print(key + ": " + str(accuracies[key]))
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


def populate_expected_accuracies(train_data_file):
	train_data = open(train_data_file, "r", encoding="utf8")
	all_forms_total = 0
	correct_count = 0
	identity_count = 0
	for line in train_data:
		if re.search('\t', line):
			field = line.strip().split('\t')
			form = field[1]
			lemma = field[2]

			if lemma_max[form] == lemma:  					# Correct lemma in lookup table
				correct_count += 1
			if form in identity_set and form == lemma:  	# Correct lemma as identity
				identity_count += 1

			all_forms_total += 1

	accuracies['Expected lookup'] = correct_count / all_forms_total
	accuracies['Identities lookup'] = identity_count / all_forms_total


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
				identity_set.add(form)
		training_counts["Wordform tokens"] += all_lemma_count

		if len(all_lemmas) == 1:
			training_counts["Unambiguous types"] += 1
			training_counts["Unambiguous tokens"] += all_lemma_count
		else:
			training_counts["Ambiguous types"] += 1
			training_counts["Ambiguous tokens"] += all_lemma_count
			training_counts["Ambiguous most common tokens"] += all_lemmas[highest_lemma]


def run_test():
	test_data = open(test_file, "r", encoding="utf8")

	for line in test_data:
		# Tab character identifies lines containing tokens
		if re.search('\t', line):
			# Tokens represented as tab-separated fields
			field = line.strip().split('\t')

			# Word form in second field, lemma in third field
			form = field[1]
			lemma = field[2]

			test_counts["Total test items"] += 1
			if form in lemma_max.keys():
				test_counts["Found in lookup table"] += 1
				if lemma_max[form] == lemma:
					test_counts["Lookup match"] += 1
				else:
					test_counts["Lookup mismatch"] += 1
			else:
				test_counts["Not found in lookup table"] += 1
				if form == lemma:
					test_counts["Identity match"] += 1
				else:
					test_counts["Identity mismatch"] += 1

	accuracies["Lookup accuracy"] = test_counts["Lookup match"] / test_counts["Found in lookup table"]
	accuracies["Identity accuracy"] = test_counts["Identity match"] / test_counts["Not found in lookup table"]
	accuracies["Overall accuracy"] = (test_counts["Lookup match"] + test_counts["Identity match"]) / test_counts["Total test items"]


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

	# Dictionary holding all accuracies
	accuracies = dict()

	identity_set = set()

	# Training
	populate_lemma_count(train_file)
	populate_lemma_max_and_training_counts()
	populate_expected_accuracies(train_file)

	# Testing
	run_test()

	# Show results
	write_results()
