import argparse
import json


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_input", metavar="train-data-path", help="Path to the file of train data", type=str)
	parser.add_argument("homework_toggle", metavar="homework-toggle", help="0 for submission | 1 for local testing",
						nargs='*', default=0, type=int)
	return parser.parse_args()


class HMM(object):

	def __init__(self):
		self.all_training_data = None
		self.states = set()
		self.observations = set()

	def load_new(self, all_training_data):
		self.all_training_data = all_training_data


if __name__ == "__main__":
	arguments = check_arguments()
