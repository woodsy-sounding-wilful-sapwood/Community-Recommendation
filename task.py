import tensorflow as tf
import numpy as np
import datetime
import os
import sh
import math
import argparse

from scipy.sparse import load_npz
from tensorflow.contrib.factorization import WALSModel

def toSparseTensor(X):
	coo = X.tocoo()
	indices = np.column_stack((X.row, X.col))
	return tf.SparseTensor(indices, coo.data, coo.shape)

def getModel(data, args):
	rWeight, cWeight = None, None
	nRows, nCols = data.shape
	if args['weights']:
		rWeight = np.ones(nRows)
		cWeight = np.ones(nCols)

	with tf.Graph().as_default():
		tensor = tf.SparseTensor(np.column_stack((data.row, data.col)), (data.data).astype(np.float32), data.shape)
		model = WALSModel(nRows, nCols, args['n_components'], unobserved_weight = args['unobserved_weight'], regularization = args['regularization'], row_weights = rWeight, col_weights = cWeight)
	return tensor, model.row_factors[0], model.col_factors[0], model

def getSession(model, inputT, niter):
	session = tf.Session(graph = inputT.graph)

	with inputT.graph.as_default():
		uUpdate = model.update_row_factors(sp_input = inputT)[1]
		vUpdate = model.update_col_factors(sp_input = inputT)[1]

		session.run(model.initialize_op)
		session.run(model.worker_init)
		for _ in range(niter):
			tf.logging.info('Iteration {} started: {:%Y-%m-%d %H:%M:%S}'.format(_ + 1, datetime.datetime.now()))
			session.run(model.row_update_prep_gramian_op)
			session.run(model.initialize_row_update_op)
			session.run(uUpdate)
			session.run(model.col_update_prep_gramian_op)
			session.run(model.initialize_col_update_op)
			session.run(vUpdate)
			tf.logging.info('Iteration {} ended: {:%Y-%m-%d %H:%M:%S}'.format(_ + 1, datetime.datetime.now()))

	return session

def trainModel(args, data):
	tf.logging.info('Training Started: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
	inputT, rFactor, cFactor, model = getModel(data, args)
	session = getSession(model, inputT, args['niter'])
	tf.logging.info('Training Finished: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
	U = rFactor.eval(session = session)
	V = cFactor.eval(session = session)
	session.close()
	return U, V

def saveModel(args, rFactor, cFactor):
	modelDir = os.path.join(args['job_dir'], 'model')
	gsModelDir = None
	if modelDir.startswith('gs://'):
		gsModelDir = modelDir
		modelDir = '/tmp/{0}'.format(args['job_name'])
	if not os.path.exists(modelDir):
		os.makedirs(modelDir)
	np.save(os.path.join(modelDir, 'row'), rFactor)
	np.save(os.path.join(modelDir, 'col'), cFactor)

	if gsModelDir:
		sh.gsutil('cp', '-r', os.path.join(modelDir, '*'), gsModelDir)

def rmse(U, V, M):
	e = 0
	for i in range(M.nnz):
		u = U[M.row[i]]
		v = V[M.col[i]]
		e += math.pow(M.data[i] - np.dot(u, v), 2)
	return math.sqrt(e / M.nnz)

def getargs():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--train-data',
		help='GCS or local paths to training data',
		required=True
	)
	parser.add_argument(
		'--test-data',
		help='GCS or local paths to testing data',
		required=True
	)
	parser.add_argument(
		'--job-dir',
		help='GCS or local paths to output directory',
		required=True
	)
	parser.add_argument(
		'--n_components',
		type=int,
		help='Number of latent factors',
	)
	parser.add_argument(
		'--niter',
		type=int,
		help='Number of iterations',
		required=True
	)
	parser.add_argument(
		'--regularization',
		type=float,
		help='L2 regularization factor',
		required=True
	)
	parser.add_argument(
		'--unobserved_weight',
		type=float,
		help='Weight for unobserved values',
		required=True
	)
	parser.add_argument(
		'--hypertune',
		default=False,
		action='store_true',
		help='Switch to turn on or off hyperparam tuning'
	)
	args = parser.parse_args().__dict__
	args['weights'] = True
	args['wt_type'] = 0
	args['job_name'] = 'netflix'
	return args

if __name__ == '__main__':
	args = getargs()
	print(args)
	train = load_npz(args['train_data']).tocoo()
	test = load_npz(args['test_data']).tocoo()
	tf.logging.set_verbosity(tf.logging.INFO)
	U, V = trainModel(args, train)
	saveModel(args, U, V)
	trainError = rmse(U, V, train)
	testError = rmse(U, V, test)

	if args['hypertune']:
		raise NotImplementedError

	tf.logging.info('Train RMSE = {0}'.format(trainError))
	tf.logging.info('Test RMSE = {0}'.format(testError))