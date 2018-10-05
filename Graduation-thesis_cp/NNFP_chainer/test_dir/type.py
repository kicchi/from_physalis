import numpy as np

a = np.array(range(2), dtype=np.string_)
b = np.array(range(2), dtype=np.float32)
c = np.array(1)


if type(a) == type(c):
	print "hgoe"
