"""
Full Name Predictor:
Given two or more names with first and surnames, identify the full name of the first person.
:author: Ismael Villegas-Molina
"""
import pandas as pd

if __name__ == "__main__":
	train = pd.read_csv("./data/dev-key.csv", header=None)
	print(train.head(5))
