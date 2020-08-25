"""
Full Name Predictor
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import argparse


def read_data():
	"""
	Reads the train and test data
	:return: first names, second names, and key of target full name for first name
	"""
	first_names = []
	second_names = []
	key_names = []
	for line in open("./data/dev-key.csv"):
		csv_row = line.strip().split(',')
		first_names.append(csv_row[0].split(" AND ")[0])
		second_names.append(csv_row[0].split(" AND ")[1])
		key_names.append(csv_row[1])
	return first_names, second_names, key_names


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


if __name__ == "__main__":
	arguments = check_arguments()

	former, latter, keys = read_data()
	surname_freq = surnames()
	female_freq, male_freq = female_male_names()

	correct = 0
	result = open("predictions.txt", "w")
	for i in range(len(keys)):
		former_tokens = former[i].split()
		latter_tokens = latter[i].split()
		former_surname_mask = [-1]
		former_female_mask = [1]
		former_male_mask = [1]
		for fs in former_tokens[1:]:
			former_surname_mask.append(surname_freq.get(fs, -1))
			former_female_mask.append(female_freq.get(fs, -1))
			former_male_mask.append(male_freq.get(fs, -1))
		latter_surname_mask = [-1]
		latter_female_mask = [1]
		latter_male_mask = [1]
		for ls in latter_tokens[1:]:
			latter_surname_mask.append(surname_freq.get(ls, -1))
			latter_female_mask.append(female_freq.get(ls, -1))
			latter_male_mask.append(male_freq.get(ls, -1))
		my_prediction = create_new_name(former_tokens, latter_tokens, former_surname_mask, latter_surname_mask, former_female_mask, former_male_mask, latter_female_mask, latter_male_mask)
		result.write(my_prediction + "\n")
		if my_prediction == keys[i]:
			correct += 1

	result.close()
	print("Percent correct:", correct/len(keys))
