import tensorflow as tf
import numpy as np
import datetime
import os
import math
import argparse
import json

from scipy.sparse import load_npz
from tensorflow.contrib.factorization import WALSModel
from io import BytesIO
from tensorflow.python.lib.io import file_io
from tensorflow.core.framework.summary_pb2 import Summary

def getModel(data, args):
	nRows, nCols = data.shape
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
		import sh
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
		help = 'GCS or local path to training data',
		required=True
	)
	parser.add_argument(
		'--test-data',
		help = 'GCS or local path to testing data',
		required=True
	)
	parser.add_argument(
		'--job-dir',
		help = 'GCS or local path to output directory',
		default = '.'
	)
	parser.add_argument(
		'--n_components',
		type = int,
		help = 'Number of latent factors',
		default = 10
	)
	parser.add_argument(
		'--niter',
		type = int,
		help = 'Number of iterations',
		default = 20
	)
	parser.add_argument(
		'--regularization',
		type = float,
		help = 'L2 regularization factor',
		default = 0.05
	)
	parser.add_argument(
		'--unobserved-weight',
		type=float,
		help='Weight for unobserved values',
		default = 0
	)
	parser.add_argument(
		'--hypertune',
		default=False,
		action='store_true',
		help='Switch to turn on or off hyperparam tuning'
	)
	parser.add_argument(
		'--job-name',
		default='myjob',
		help='Only needed if you are training on Google Cloud'
	)
	args = parser.parse_args().__dict__

	if args['hypertune']:
		trial = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
		args['job_dir'] = os.path.join(args['job_dir'], trial)

	return args

def loadMatrix(loc):
	if loc.startswith('gs://'):
		return load_npz(BytesIO(file_io.read_file_to_string(loc, binary_mode=True))).tocoo()
	return load_npz(loc).tocoo()

def hyperLog(args, rmse):
	log = Summary(value = [Summary.Value(tag = 'training/hptuning/metric', simple_value = rmse)])
	logpath = os.path.join(args['job_dir'], 'eval')
	writer = tf.summary.FileWriter(logpath)
	writer.add_summary(log)
	writer.flush()

def checkpointSaver(res):
	dump(res, 'checkpoint')

if __name__ == '__main__':
	args = getargs()
	train = loadMatrix(args['train_data'])
	test = loadMatrix(args['test_data'])
	tf.logging.set_verbosity(tf.logging.INFO)
	U, V = trainModel(args, train)
	saveModel(args, U, V)
	trainError = rmse(U, V, train)
	testError = rmse(U, V, test)

	if args['hypertune']:
		hyperLog(args, testError)

	tf.logging.info('Train RMSE = {0}'.format(trainError))
	tf.logging.info('Test RMSE = {0}'.format(testError))