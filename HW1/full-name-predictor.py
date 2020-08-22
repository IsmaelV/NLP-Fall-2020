import pandas as pd

if __name__ == "__main__":
	print("Hello world")
	train = pd.read_csv("./data/dev-key.csv", header=None)
	print(train.head(5))
