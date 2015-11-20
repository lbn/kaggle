import pandas
import os

data_dir = os.path.dirname(__file__)
train = pandas.read_csv(os.path.join(data_dir, "train.csv"))
test = pandas.read_csv(os.path.join(data_dir, "test.csv"))
