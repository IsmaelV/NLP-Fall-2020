"""
Full Name Predictor
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import pandas as pd
import argparse


def read_data(path_to_test_data):
	return pd.read_csv("./data/dev-key.csv", header=None), pd.read_csv(path_to_test_data, header=None)


def check_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_test_data", metavar="data-path", help="Path to the csv file with test data", type=str)
	return parser.parse_args()


if __name__ == "__main__":
	arguments = check_arguments()

	train, test = read_data(arguments.path_to_test_data)

	print(train.head(5))
	print(test.head(5))
