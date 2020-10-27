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
		self.emission_prob = dict()

	def load_new(self, all_training_data):
		"""
		Using the corpus data, load and calculate all data.
		:param all_training_data: Path to training data
		:return:
		"""
		self.read_data(all_training_data)
		return

	def load_pretrained(self, model_file):
		"""
		Using a text file of the model, load all the data.
		:param model_file: Path to model text file
		:return:
		"""
		f = open(model_file, 'r')
		data = json.load(f)
		f.close()

		self.states = set(data["states"])
		self.observations = set(data["observations"])
		self.state_transition_prob = data["state_transition_prob"]
		self.emission_prob = data["emission_prob"]
		return True

	def read_data(self, d):
		"""
		Take in data from corpus and read line-by-line. We create state transition and emission probabilities for POS
		tagging and Viterbi decoding. Perform add-one smoothing on state transition probabilities.
		:param d: Path to corpus data
		:return:
		"""
		f = open(d, encoding="utf8")

		# Read each sentence and store counts. Last line is adding END state
		for sent in f:
			prev_state = "START"
			for w in sent.split():
				word, tag = w.rsplit("/", 1)

				self.states.add(tag)
				self.observations.add(word)

				# State Transition Probabilities
				if prev_state not in self.state_transition_prob:
					self.state_transition_prob[prev_state] = dict()
				self.state_transition_prob[prev_state][tag] = self.state_transition_prob[prev_state].get(tag, 0) + 1
				prev_state = tag

				# Emission Probabilities
				if tag not in self.emission_prob:
					self.emission_prob[tag] = dict()
				self.emission_prob[tag][word] = self.emission_prob[tag].get(word, 0) + 1
			if prev_state not in self.state_transition_prob:
				self.state_transition_prob[prev_state] = dict()
			self.state_transition_prob[prev_state]["END"] = self.state_transition_prob[prev_state].get("END", 0) + 1

		# Add-One Smoothing for state transitions
		for s1 in self.states:
			for s2 in self.states:
				self.state_transition_prob[s1][s2] = self.state_transition_prob[s1].get(s2, 0) + 1
			self.state_transition_prob["START"][s1] = self.state_transition_prob["START"].get(s1, 0) + 1
			self.state_transition_prob[s1]["END"] = self.state_transition_prob[s1].get("END", 0) + 1
		# Convert counts into probabilities
		for k in self.state_transition_prob.keys():
			my_dict = self.state_transition_prob[k]
			tot_val = sum(my_dict.values())
			for j in my_dict.keys():
				my_dict[j] /= tot_val
		for k in self.emission_prob.keys():
			my_dict = self.emission_prob[k]
			tot_val = sum(my_dict.values())
			for j in my_dict.keys():
				my_dict[j] /= tot_val

		return

	def save(self, file_name):
		"""
		Save model into a text file.
		:param file_name: Name of file to save to
		:return: True
		"""
		my_save_file = open(file_name, 'w')
		all_info = {"states": list(self.states), "observations": list(self.observations),
					"state_transition_prob": self.state_transition_prob, "emission_prob": self.emission_prob}
		my_save_file.write(json.dumps(all_info))
		my_save_file.close()
		return True

	def viterbi(self, line, dev=False):
		key = []
		all_prev_states = ["START"]
		node_table = {(0, "START"): 1}
		lookback_table = {(0, "START"): (None, None)}
		step = 1
		for w in line.split():
			if dev:
				word, tag = w.rsplit("/", 1)
				key.append(tag)
			else:
				word = w

			# ----------------------------
			# Emission Probability
			# ----------------------------
			emission_nodes = dict()

			# Seen word
			if word in self.observations:
				for t in self.emission_prob.keys():
					tmp_emission = self.emission_prob[t].get(word, -1)
					if tmp_emission > 0:
						emission_nodes[t] = tmp_emission

			# Unseen word - Get top two transitions for each previous state and set emission probability equal
			else:
				top_possible_transitions = set()
				for p in all_prev_states:
					tmp_states = self.state_transition_prob[p]
					tmp_states = sorted(tmp_states.items(), key=lambda x: x[1], reverse=True)
					for i in range(2):
						top_possible_transitions.add(tmp_states[i][0])
				if "END" in top_possible_transitions:
					top_possible_transitions.remove("END")
				for t in top_possible_transitions:
					emission_nodes[t] = 0.000001

			# -------------------------------------------------------------
			# Go through each emission node (vertical nodes) and calculate
			# the max value at that node and its parent node
			# -------------------------------------------------------------
			for e in emission_nodes.keys():
				parent_node = ""
				curr_max = -1
				for s in all_prev_states:
					prev_state_node = node_table[(step - 1, s)]
					tmp_result = prev_state_node * self.state_transition_prob[s][e] * emission_nodes[e]
					if tmp_result > curr_max:
						parent_node = s
						curr_max = tmp_result
				node_table[(step, e)] = curr_max
				lookback_table[(step, e)] = (step - 1, parent_node)

			step += 1
			all_prev_states = list(emission_nodes.keys())

		# Perform calculations on END state. Same procedure as above in "for s in all_prev_state" loop, but for END state
		curr_max = -1
		parent_node = ""
		for s in all_prev_states:
			prev_state_node = node_table[(step - 1, s)]
			tmp_result = prev_state_node * self.state_transition_prob[s]["END"]
			if tmp_result > curr_max:
				parent_node = s
				curr_max = tmp_result
		node_table[(step, "END")] = curr_max
		lookback_table[(step, "END")] = (step - 1, parent_node)

		# Look back at parents and create POS prediction
		prediction = []
		prev_step, parent = lookback_table[(step, "END")]
		while parent:
			prediction.append(parent)
			prev_step, parent = lookback_table[(prev_step, parent)]
			if parent == "START":
				break
		prediction.reverse()
		return (prediction, key) if dev else prediction

	def dev(self, dev_file):
		f = open(dev_file, encoding="utf8")

		total = 0
		total_correct = 0
		# Read each sentence and perform Viterbi Algorithm
		for sent in f:
			prediction, key = self.viterbi(sent, dev=True)
			for j in range(len(prediction)):
				if prediction[j] == key[j]:
					total_correct += 1
				total += 1
		print("Total Correct:", total_correct/total)
		return


if __name__ == "__main__":
	arguments = check_arguments()
	my_hmm = HMM()
	my_hmm.load_new(arguments.path_to_input)
	my_hmm.save("hmmmodel.txt")
