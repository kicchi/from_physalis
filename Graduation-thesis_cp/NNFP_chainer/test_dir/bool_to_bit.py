import numpy as np

data = [[True, False, True],
		[False, True, False],
		[True, True, False]]

def bool_to_bit(features):
	features = np.array(features)
	features = features * 1

bool_to_bit(data)

for i in data:
	print i
