import itertools

def tuplelist(x):
	LIST = []
	for v in itertools.product(*x):
		LIST.append(v)
	return LIST
