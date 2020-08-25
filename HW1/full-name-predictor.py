"""
Full Name Predictor
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import argparse
import csv


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
	rank = dict()
	frequency = dict()
	count = 0
	with open("./data/Surnames_2010Census.csv") as f:
		next(f)
		for line in f:
			csv_row = line.strip().split(',')
			count += int(csv_row[2])
			rank[csv_row[0]] = int(csv_row[1])
			frequency[csv_row[0]] = int(csv_row[2])
	for key in frequency.keys():
		frequency[key] = frequency[key] / count
	return rank, frequency


def rule_1(first_tokens, second_tokens):
	if len(first_tokens) == 1 and len(second_tokens) == 2:
		return True, first_tokens[0] + " " + second_tokens[1]
	return False, ""


def rule_2(first_tokens, second_tokens):
	if len(first_tokens) == 1 and len(second_tokens) == 3:
		return True, first_tokens[0] + " " + second_tokens[2]
	return False, ""


def rule_3(first_tokens, second_tokens):
	print()
	return False, ""


if __name__ == "__main__":
	arguments = check_arguments()

	former, latter, keys = read_data()
	surname_rank, surname_freq = surnames()
