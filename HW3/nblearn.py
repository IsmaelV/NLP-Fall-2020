import argparse


def check_arguments():
	"""
	Check to see that all the arguments are correct
	:return: Arguments that were parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_input", metavar="train-data-path", help="Path to the file of train data", type=str)
	return parser.parse_args()


if __name__ == "__main__":
	arguments = check_arguments()
	print("Hello world")
