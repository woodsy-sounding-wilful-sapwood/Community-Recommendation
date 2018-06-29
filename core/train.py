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

def get_model(data, ncomponents = 10, unobserved_weight = 0, regularization = 0.05):
	nrows, ncols = data.shape
	r_weight = np.ones(nrows)
	c_weight = np.ones(ncols)

	with tf.Graph().as_default():
		tensor = tf.SparseTensor(np.column_stack((data.row, data.col)), (data.data).astype(np.float32), data.shape)
		model = WALSModel(nrows, ncols, ncomponents, unobserved_weight, regularization, row_weights = r_weight, col_weights = c_weight)
	return tensor, model.row_factors[0], model.col_factors[0], model

def get_session(model, input_tensor, niter = 20):
	session = tf.Session(graph = input_tensor.graph)

	with input_tensor.graph.as_default():
		u_update = model.update_row_factors(sp_input = input_tensor)[1]
		v_update = model.update_col_factors(sp_input = input_tensor)[1]

		session.run(model.initialize_op)
		session.run(model.worker_init)
		for _ in range(niter):
			tf.logging.info('Iteration {} started: {:%Y-%m-%d %H:%M:%S}'.format(_ + 1, datetime.datetime.now()))
			session.run(model.row_update_prep_gramian_op)
			session.run(model.initialize_row_update_op)
			session.run(u_update)
			session.run(model.col_update_prep_gramian_op)
			session.run(model.initialize_col_update_op)
			session.run(v_update)
			tf.logging.info('Iteration {} ended: {:%Y-%m-%d %H:%M:%S}'.format(_ + 1, datetime.datetime.now()))

	return session

def train_model(data, niter = 20, ncomponents = 10, unobserved_weight = 0, regularization = 0.05):
	tf.logging.info('Training Started: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
	input_tensor, r_factor, c_factor, model = get_model(data, ncomponents = ncomponents, unobserved_weight = unobserved_weight, regularization = regularization)
	session = get_session(model, input_tensor, niter = niter)
	tf.logging.info('Training Finished: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
	U = r_factor.eval(session = session)
	V = c_factor.eval(session = session)
	session.close()
	return U, V

def save_model(r_factor, c_factor, job_dir = '.', job_name = 'myjob'):
	model_dir = os.path.join(job_dir, 'model')
	gs_model_dir = None
	if model_dir.startswith('gs://'):
		gs_model_dir = model_dir
		model_dir = '/tmp/{0}'.format(job_name)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	np.save(os.path.join(model_dir, 'row'), r_factor)
	np.save(os.path.join(model_dir, 'col'), c_factor)

	if gs_model_dir:
		import sh
		sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)

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
		'--ncomponents',
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

def load_matrix(loc):
	if loc.startswith('gs://'):
		return load_npz(BytesIO(file_io.read_file_to_string(loc, binary_mode=True))).tocoo()
	return load_npz(loc).tocoo()

def hyper_log(rmse, job_dir):
	log = Summary(value = [Summary.Value(tag = 'training/hptuning/metric', simple_value = rmse)])
	logpath = os.path.join(job_dir, 'eval')
	writer = tf.summary.FileWriter(logpath)
	writer.add_summary(log)
	writer.flush()

if __name__ == '__main__':
	args = getargs()
	train = load_matrix(args['train_data'])
	test = load_matrix(args['test_data'])
	tf.logging.set_verbosity(tf.logging.INFO)
	U, V = train_model(train, niter = args['niter'], ncomponents = args['ncomponents'], unobserved_weight = args['unobserved_weight'], regularization = args['regularization'])
	save_model(U, V)
	train_error = rmse(U, V, train)
	test_error = rmse(U, V, test)

	if args['hypertune']:
		hyper_log(test_error, args['job_dir'])

	tf.logging.info('Train RMSE = {0}'.format(train_error))
	tf.logging.info('Test RMSE = {0}'.format(test_error))
