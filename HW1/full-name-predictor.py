"""
Full Name Predictor
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import argparse


def read_train():
	"""
	Reads the train data and grabs the ground truth
	:return: List of target full name for former name
	"""
	key_names = []
	former = []
	latter = []
	for line in open("./data/dev-key.csv"):
		csv_row = line.strip().split(',')
		former_name, latter_name = csv_row[0].split(" AND ")
		former.append(former_name)
		latter.append(latter_name)
		key_names.append(csv_row[1])
	return key_names, former, latter


def read_test(path_to_test):
	"""
	Reads the test data
	:return: List of former names, List of latter names
	"""
	former = []
	latter = []
	for line in open(path_to_test):
		row = line.strip().split(" AND ")
		former.append(row[0])
		latter.append(row[1])
	return former, latter


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_test_data", metavar="data-path", help="Path to the csv file with test data", type=str)
	return parser.parse_args()


def get_surnames():
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


def get_female_male_names():
	"""
	Reads female and male names from data/readme-license.txt urls given
	:return: Frequencies of female names, frequencies of male names
	"""
	female_frequency = dict()
	with open("./data/female_names.txt") as f:
		next(f)
		for line in f:
			row = line.strip().split()
			female_frequency[row[0]] = float(row[1]) / 100

	male_frequency = dict()
	with open("./data/male_names.txt") as f:
		next(f)
		for line in f:
			row = line.strip().split()
			male_frequency[row[0]] = float(row[1]) / 100

	return female_frequency, male_frequency


def get_professional_titles():
	titles = set()
	with open("./data/professional_titles.txt") as f:
		for line in f:
			titles.add(line.strip())
	return titles


def single_name_predicted(init_prediction, l_tokens):
	"""
	Given that the prediction was a single name (not possible), we perform operations to add a surname. The initial way
	is to simply add the last token of the latter name to the prediction. The second way (commented) is to test the
	name frequencies of female and male names vs. surnames.
	:param init_prediction: String of initial prediction with no surname
	:param l_tokens: List of string tokens of latter name
	:return: Updated prediction with a surname
	"""
	# boolean_use = []
	# for i in range(1, len(l_tokens)):
	# 	if l_gendered[i] == -1:
	# 		boolean_use.append(True)
	# 	elif l_gendered[i] - l_surname[i] < 0.01:
	# 		boolean_use.append(True)
	# 	else:
	# 		boolean_use.append(False)
	#
	# for k in range(len(boolean_use)):
	# 	if boolean_use[k]:
	# 		init_prediction += " " + l_tokens[k+1]
	#
	# return init_prediction

	return init_prediction + " " + l_tokens[-1]


def determine_initial_name(tokens, professional_titles):
	if tokens[0] in professional_titles:
		return tokens[0] + " " + tokens[1], True
	return tokens[0], False


def add_name(surname_freq, gendered_freq, forename=False):
	"""
	Check to see if a name will be added or not
	:param surname_freq: List of frequencies of surname from Census data [-1, 1)
	:param gendered_freq: List of frequencies of either male or female names from data [-1, 1)
	:param forename: Boolean that determines whether we are looking to add forenames to prediction
	:return: List of booleans of name indices of which to add to the prediction
	"""
	first_over_surname = []
	for s in range(len(surname_freq)):
		if surname_freq[s] > gendered_freq[s] and abs(gendered_freq[s] - surname_freq[s]) * 100 > 0.05:
			if forename:
				first_over_surname.append(False)
			else:
				first_over_surname.append(True)
		elif forename:
			first_over_surname.append(True)
		else:
			first_over_surname.append(False)
	return first_over_surname


def create_new_name(f_tokens, l_tokens, f_surnames, l_surnames, f_gender, l_gender, professional_titles):
	"""
	Creates the name of which we are predicting
	:param f_tokens: List of names before the " AND ". The former name. Need forenames from here
	:param l_tokens: List of names after the " AND ". The latter name. Need surnames from here
	:param f_surnames: List of surname frequency from Census data on former name tokens
	:param l_surnames: List of surname frequency from Census data on latter name tokens
	:param f_gender: List of female or male name frequency from data on former name tokens
	:param l_gender: List of female or male name frequency from data on latter name tokens
	:param professional_titles: Set of professional titles
	:return: String of predicted name
	"""
	# Always keep first token. Will always be a forename or a title (Professor, Dr., etc.)
	new_name, title_found_former = determine_initial_name(f_tokens, professional_titles)

	# Get forenames that we will keep from former
	add_forename = add_name(f_surnames, f_gender, forename=True)
	init_index = 2 if title_found_former else 1
	for j in range(init_index, len(add_forename)):
		if add_forename[j]:
			new_name += " " + f_tokens[j]
		else:  # If not a forename, then skip all subsequent names
			break

	# Get surnames that we will keep
	add_surname = add_name(f_surnames, f_gender)

	# If a surname exists in the former name, then keep it and ignore the latter name
	if True in add_surname:

		# If surname found, make all subsequent surnames be True
		true_index = 100
		for i in range(len(add_surname)):
			if add_surname[i]:
				true_index = i
				break
		for j in range(true_index, len(add_surname)):
			add_surname[j] = True

		for j in range(init_index, len(add_surname)):
			if add_surname[j]:
				new_name += " " + f_tokens[j]

	# If surname does not exist in former name, check the surnames in the latter name
	else:
		added_surname = False
		add_surname = add_name(l_surnames, l_gender)

		# If surname found, make all subsequent surnames be True
		true_index = 100
		for i in range(len(add_surname)):
			if add_surname[i]:
				true_index = i
				break
		for j in range(true_index, len(add_surname)):
			add_surname[j] = True

		# Add all surnames
		_, title_found_latter = determine_initial_name(l_tokens, professional_titles)
		l_i = 2 if title_found_latter else 1
		for k in range(l_i, len(add_surname)):
			if add_surname[k]:
				new_name += " " + l_tokens[k]
				added_surname = True
		if not added_surname:  # If no surname was added, then simply append the last latter token to prediction
			new_name += " " + l_tokens[-1]

	return new_name


def evaluate_predictions():
	"""
	Read the predictions from full-name-output.csv made from create_predictions()
	:return: Accuracy of the predictions
	"""
	true_answers, former, latter = read_train()  # Former and Latter are extracted for debugging
	predictions = open("full-name-output.csv", "r")
	ind = 0
	correct = 0
	for p in predictions:
		prediction = p.strip().split(',')[1]
		if prediction == true_answers[ind]:
			correct += 1
		else:
			print("================================")
			print("Prediction:\t\t", prediction)
			print("Correct Ans:\t", true_answers[ind])
		ind += 1
	predictions.close()
	print("Prediction accuracy:", correct/len(true_answers))
	return correct/len(true_answers)


def create_predictions(path_to_test, file_to_write):
	"""
	Create predictions and write them onto file for results
	:param path_to_test: String of path to where test data is found
	:param file_to_write: File that is being written to
	:return:
	"""
	former, latter = read_test(path_to_test)
	surname_freq = get_surnames()
	female_freq, male_freq = get_female_male_names()
	professional_titles = get_professional_titles()

	for i in range(len(former)):
		# --------------------------------
		# Create former and latter tokens
		# --------------------------------
		former_tokens = former[i].split()  	# Create tokens for former name
		latter_tokens = latter[i].split()  	# Create tokens for latter name

		# --------------------------------
		# Create and populate former masks
		# --------------------------------
		_, title_found = determine_initial_name(former_tokens, professional_titles)  # Check if title is used
		former_gender_mask = [1, 1] if title_found else [1]  						# Create gendered mask for former name
		f_i = 1 if title_found else 0  												# Get forename index for former name
		use_female = female_freq.get(former_tokens[f_i], -1) >= male_freq.get(former_tokens[f_i], -1)  # Check gender
		former_surname_mask = [-1, -1] if title_found else [-1]  					# Create surname mask for former name
		f_i += 1  																	# Get following forename index
		for fs in former_tokens[f_i:]:  											# Populate former masks
			former_surname_mask.append(surname_freq.get(fs, -1))
			if use_female:
				former_gender_mask.append(female_freq.get(fs, -1))
			else:
				former_gender_mask.append(male_freq.get(fs, -1))

		# --------------------------------
		# Create and populate latter masks
		# --------------------------------
		_, title_found = determine_initial_name(latter_tokens, professional_titles)  	# Check if title is used
		latter_gender_mask = [1, 1] if title_found else [1]  							# Create gendered mask for latter name
		l_i = 1 if title_found else 0  													# Get forename index for latter name
		use_female = female_freq.get(latter_tokens[l_i], -1) >= male_freq.get(latter_tokens[l_i], -1)  # Check gender
		latter_surname_mask = [-1, -1] if title_found else [-1]  						# Create surname mask for latter name
		l_i += 1  																		# Get following forename index
		for ls in latter_tokens[l_i:]:  												# Populate latter masks
			latter_surname_mask.append(surname_freq.get(ls, -1))
			if use_female:
				latter_gender_mask.append(female_freq.get(ls, -1))
			else:
				latter_gender_mask.append(male_freq.get(ls, -1))

		# --------------------------------
		# Create the prediction name
		# --------------------------------
		my_prediction = create_new_name(former_tokens, latter_tokens, former_surname_mask, latter_surname_mask,
										former_gender_mask, latter_gender_mask, professional_titles)

		if len(my_prediction.strip().split()) == 1:
			my_prediction = single_name_predicted(my_prediction, latter_tokens)

		# --------------------------------
		# Write results into file
		# --------------------------------
		file_to_write.write(former[i] + " AND " + latter[i] + "," + my_prediction + "\n")

	return


if __name__ == "__main__":
	arguments = check_arguments()

	results = open("full-name-output.csv", "w")
	create_predictions(arguments.path_to_test_data, results)
	results.close()

	evaluate_predictions()
