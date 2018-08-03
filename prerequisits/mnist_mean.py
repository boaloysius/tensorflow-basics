# This is a program prints the mean image of the label in the MNIST dataset

# Download MNIST dataset from https://www.kaggle.com/c/digit-recognizer/dats
# and store it as files/mnist/train.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import mnist data to dataframe
# labels are stored in labels column
df = pd.read_csv("files/mnist/train.csv")

label = 2

def print_mean(label):
	required_df = df[df['label'] == label]
	# convert df to matrix
	W = required_df.as_matrix()

	#remove label colum
	im = W[:,1:]

	# find mean of different matrix representations of the label
	im = im.mean(axis=0)

	#im is of shape (78, )reshape to (28, 28) to represent image pixels
	im = im.reshape(28,28)

	plt.imshow(im, cmap="gray")
	plt.show()

print_mean(int(raw_input("Enter a single digit number \n")))