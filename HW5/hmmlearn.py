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
		self.state_transition_prob = {"START": dict()}

	def load_new(self, all_training_data):
		self.read_data(all_training_data)

	def read_data(self, d):
		f = open(d, encoding="utf8")

		# Read each sentence and store counts
		for sent in f:
			prev_state = "START"
			for w in sent.split():
				word, tag = w.rsplit("/", 1)

				self.states.add(tag)
				self.observations.add(word)

				# TODO: Need to do emission probabilities, and see if it's words or tags
				# State transitions will be for tags, NOT WORDS
				if prev_state not in self.state_transition_prob:
					self.state_transition_prob[prev_state] = dict()
				tmp_dict = self.state_transition_prob[prev_state]
				tmp_val = tmp_dict.get(tag, 0) + 1
				tmp_dict[tag] = tmp_val
				# self.state_transition_prob[prev_state] = tmp_dict  # This might be unnecessary
				prev_state = tag

		# Convert counts into probabilities
		for k in self.state_transition_prob.keys():
			my_dict = self.state_transition_prob[k]
			tot_val = sum(my_dict.values())
			for j in my_dict.keys():
				my_dict[j] /= tot_val


if __name__ == "__main__":
	arguments = check_arguments()
	my_hmm = HMM()
	my_hmm.load_new(arguments.path_to_input)
