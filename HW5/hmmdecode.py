import argparse
import os
import string
import random
import json


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_input", metavar="test-data-path", help="Path to the file of test data", type=str)
	return parser.parse_args()


if __name__ == "__main__":
	arguments = check_arguments()
