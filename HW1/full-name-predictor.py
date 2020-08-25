"""
Full Name Predictor
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import argparse


def read_data():
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
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_test_data", metavar="data-path", help="Path to the csv file with test data", type=str)
	return parser.parse_args()


def surnames():
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
	new_name = f_tokens[0]
	add_forename = add_name(f_surnames, f_female, f_male, forename=True)
	for j in range(1, len(add_forename)):
		if add_forename[j]:
			new_name += " " + f_tokens[j]

	add_surname = add_name(f_surnames, f_female, f_male)
	if True in add_surname:
		for j in range(1, len(add_surname)):
			if add_surname[j]:
				new_name += " " + f_tokens[j]
	else:
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
