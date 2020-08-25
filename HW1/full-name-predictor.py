"""
Full Name Predictor
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import argparse


def read_train_key():
	"""
	Reads the train data and grabs the ground truth
	:return: List of target full name for former name
	"""
	key_names = []
	for line in open("./data/dev-key.csv"):
		csv_row = line.strip().split(',')
		key_names.append(csv_row[1])
	return key_names


def read_test(path_to_test):
	"""
	Reads the test data
	:return: List of former names, List of latter names
	"""
	first_names = []
	second_names = []
	for line in open(path_to_test):
		row = line.strip().split(" AND ")
		first_names.append(row[0])
		second_names.append(row[1])
	return first_names, second_names


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_test_data", metavar="data-path", help="Path to the csv file with test data", type=str)
	return parser.parse_args()


def surnames():
	"""
	Read surname data from 2010 Census
	:return: Frequency of each name
	"""
	frequency = dict()
	count = 0
	with open("./data/Surnames_2010Census.csv") as f:
		next(f)
		for line in f:
			csv_row = line.strip().split(',')
			count += int(csv_row[2])
			frequency[csv_row[0]] = int(csv_row[2])
	for key in frequency.keys():
		frequency[key] = frequency[key] / count
	return frequency


def female_male_names():
	"""
	Reads female and male names from data/readme-license.txt urls given
	:return: Frequencies of female names, frequencies of male names
	"""
	female_frequency = dict()
	with open("./data/female_names.txt") as f:
		next(f)
		for line in f:
			row = line.strip().split('\t')
			female_frequency[row[0]] = float(row[1])

	male_frequency = dict()
	with open("./data/male_names.txt") as f:
		next(f)
		for line in f:
			row = line.strip().split('\t')
			male_frequency[row[0]] = float(row[1])

	return female_frequency, male_frequency


def rule_1(first_tokens, second_tokens):
	if len(first_tokens) == 1 and len(second_tokens) == 2:
		return True, first_tokens[0] + " " + second_tokens[1]
	return False, ""


def rule_2(first_tokens, second_tokens):
	if len(first_tokens) == 1 and len(second_tokens) == 3:
		return True, first_tokens[0] + " " + second_tokens[2]
	return False, ""


def add_name(all_surnames, female, male, forename=False):
	"""
	Check to see if a name will be added or not
	:param all_surnames: List of frequencies of surname from Census data [-1, 1)
	:param female: List of frequencies of female names from data [-1, 1)
	:param male: List of frequencies of male names from data [-1, 1)
	:param forename: Boolean that determines whether we are looking to add forenames to prediction
	:return: List of booleans of name indices of which to add to the prediction
	"""
	first_over_surname = []
	for s in range(len(all_surnames)):
		if all_surnames[s] > 0 and all_surnames[s] > female[s] and all_surnames[s] > male[s]:
			if forename:
				first_over_surname.append(False)
			else:
				first_over_surname.append(True)
			continue
		if forename:
			first_over_surname.append(True)
		else:
			first_over_surname.append(False)
	return first_over_surname


def create_new_name(f_tokens, l_tokens, f_surnames, l_surnames, f_female, f_male, l_female, l_male):
	"""
	Creates the name of which we are predicting
	:param f_tokens: List of names before the " AND ". The former name. Need forenames from here
	:param l_tokens: List of names after the " AND ". The latter name. Need surnames from here
	:param f_surnames: List of surname frequency from Census data on former name tokens
	:param l_surnames: List of surname frequency from Census data on latter name tokens
	:param f_female: List of female name frequency from data on former name tokens
	:param f_male: List of male name frequency from data on former name tokens
	:param l_female: List of female name frequency from data on latter name tokens
	:param l_male: List of male name frequency from data on latter name tokens
	:return: String of predicted name
	"""
	# Always keep first token. Will always be a forename or a title (Professor, Dr., etc.)
	new_name = f_tokens[0]

	# Get forenames that we will keep from former
	add_forename = add_name(f_surnames, f_female, f_male, forename=True)
	for j in range(1, len(add_forename)):
		if add_forename[j]:
			new_name += " " + f_tokens[j]

	# Get surnames that we will keep
	add_surname = add_name(f_surnames, f_female, f_male)
	if True in add_surname:  # If a surname exists in the former name, then keep it and ignore the latter name
		for j in range(1, len(add_surname)):
			if add_surname[j]:
				new_name += " " + f_tokens[j]
	else:  # If surname does not exist in former name, check the surnames in the latter name
		add_surname = add_name(l_surnames, l_female, l_male)
		for j in range(1, len(add_surname)):
			if add_surname[j]:
				new_name += " " + l_tokens[j]

	return new_name


def evaluate_predictions():
	"""
	Read the predictions from predictions.txt made from create_predictions()
	:return: Precision of the predictions
	"""
	true_answers = read_train_key()
	predictions = open("predictions.txt", "r")
	ind = 0
	correct = 0
	for p in predictions:
		prediction = p.strip()
		if prediction == true_answers[ind]:
			correct += 1
		ind += 1
	predictions.close()
	print("Prediction precision:", correct/len(true_answers))
	return correct/len(true_answers)


def create_predictions(path_to_test, file_to_write):
	"""
	Create predictions and write them onto file for results
	:param path_to_test: String of path to where test data is found
	:param file_to_write: File that is being written to
	:return:
	"""
	former, latter = read_test(path_to_test)
	surname_freq = surnames()
	female_freq, male_freq = female_male_names()

	for i in range(len(former)):
		# --------------------------------
		# Create former and latter tokens
		# --------------------------------
		former_tokens = former[i].split()  	# Create tokens for former name
		latter_tokens = latter[i].split()  	# Create tokens for latter name

		# --------------------------------
		# Create and populate former masks
		# --------------------------------
		former_surname_mask = [-1]  		# Create surname mask for former name
		former_female_mask = [1]  			# Create female mask for former name
		former_male_mask = [1]  			# Create male mask for former name
		for fs in former_tokens[1:]:  		# Populate former masks
			former_surname_mask.append(surname_freq.get(fs, -1))
			former_female_mask.append(female_freq.get(fs, -1))
			former_male_mask.append(male_freq.get(fs, -1))

		# --------------------------------
		# Create and populate latter masks
		# --------------------------------
		latter_surname_mask = [-1]  		# Create surname mask for latter name
		latter_female_mask = [1]  			# Create female mask for latter name
		latter_male_mask = [1]  			# Create male mask for latter name
		for ls in latter_tokens[1:]:  		# Populate latter masks
			latter_surname_mask.append(surname_freq.get(ls, -1))
			latter_female_mask.append(female_freq.get(ls, -1))
			latter_male_mask.append(male_freq.get(ls, -1))

		# --------------------------------
		# Create the prediction name
		# --------------------------------
		my_prediction = create_new_name(former_tokens, latter_tokens, former_surname_mask, latter_surname_mask,
										former_female_mask, former_male_mask, latter_female_mask, latter_male_mask)

		# --------------------------------
		# Write results into file
		# --------------------------------
		file_to_write.write(my_prediction + "\n")

	return


if __name__ == "__main__":
	arguments = check_arguments()

	results = open("predictions.txt", "w")
	create_predictions(arguments.path_to_test_data, results)
	results.close()

	evaluate_predictions()
