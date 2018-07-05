import argparse
import json
import itertools

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from io import BytesIO

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--v',
		help = 'Location of the column factors',
		required = True
		)
	parser.add_argument(
		'--r',
		help = 'Ratio of vectors to plot / total vectors. Default 0.1.',
		type = float,
		default = 0.01
		)
	parser.add_argument(
		'--x-axis',
		help = 'Which factor should be plotted along the x-axis. Default 0.',
		default = 0
		)
	parser.add_argument(
		'--y-axis',
		help = 'Which argument should be plotted along the y-axis. Default 1.',
		default = 1
		)
	parser.add_argument(
		'--item-map',
		help = 'Stores item ids or item names.',
		default = None,
		required = False
		)
	parser.add_argument(
		'--lookup-table',
		help = 'Stores item names indexed by item-id',
		default = None,
		required = False
		)
	parser.add_argument(
		'--u',
		help = 'Stores item names indexed by item-id',
		required = True
		)
	parser.add_argument(
		'--dataset',
		help = 'Stores the movies rated by the user',
		nargs = '+',
		required = False
		)

	return parser.parse_args()

def load_json(path):
	with open(path, 'r') as f:
		return json.load(f)

def visualise(U, V, item_map = None, lookup_table = None, idx = -1, M = None, r = 0.01, x_axis = -1, y_axis = -1, top_color = 'blue', bottom_color = 'red', watched_color = 'green', return_file = True):
	idx = np.random.randint(U.shape[0]) if idx < 0 or idx >= U.shape[0] else idx
	u = U[idx, :]
	pred = np.argsort(V.dot(u))
	top = pred[-1 : -int(r * V.shape[0]) - 1: -1]
	bottom = pred[0 : int(r * V.shape[0])]
	watched = M[idx, :].tocoo().col
	x_axis = np.random.randint(V.shape[1]) if x_axis < 0 else x_axis
	y_axis = x_axis if y_axis < 0 else y_axis
	while x_axis == y_axis and V.shape[1] > 1:
		y_axis = np.random.randint(V.shape[1])
	plt.scatter(V[top, x_axis], V[top, y_axis], color = top_color)
	plt.scatter(V[bottom, x_axis], V[bottom, y_axis], color = bottom_color)
	plt.scatter(V[watched, x_axis], V[watched, y_axis], color = watched_color)
	for i in itertools.chain(top, bottom, watched):
		if item_map:
			label = lookup_table.get(item_map[i], item_map[i]) if lookup_table else item_map[i]
		elif M and M[idx, i] != 0:
			label = M[idx, i]
		else:
			label = np.around(V[i, :].dot(u), decimals = 2)
		plt.annotate(label, V[i, [x_axis, y_axis]], ha = 'center')
	plt.title('User ' + str(idx))
	plt.xlabel('Factor ' + str(x_axis))
	plt.ylabel('Factor ' + str(y_axis))

	if return_file:
		im_file = BytesIO()
		plt.savefig(im_file)
		im_file.seek(0)
		plt.clf()
		return im_file

def main():
	args = get_args()

	V = np.load(args.v)
	U = np.load(args.u)
	item_map = np.array(load_json(args.item_map)) if args.item_map else None
	lookup_table = load_json(args.lookup_table) if args.lookup_table else None
	M = sum([load_npz(x).tocsr() for x in args.dataset])

	while True:
		visualise(U, V, item_map = item_map, lookup_table = lookup_table, M = M, r = args.r, x_axis = args.x_axis, y_axis = args.y_axis, return_file = False)
		plt.show()


if __name__ == '__main__':
	main() 
