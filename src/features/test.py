from src.datasets.utils import *

parts = [0]
ratio = 0.8
notes = read_interim_files(parts)
train, test = train_test_split(ratio, notes)
print(len(train))
print(len(test))