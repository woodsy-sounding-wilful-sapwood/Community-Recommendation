import datetime
import json
import matplotlib
matplotlib.use('Agg')

import numpy as np

from flask import Flask, request, send_file
from subprocess import check_output
from urllib.request import urlopen, Request
from redis import Redis
from io import BytesIO
from multiprocessing import Process
from os import getenv
from scipy.sparse import load_npz, save_npz

import core.predict
import core.preprocess
import core.train
import core.train_temporal
import core.visualise

app = Flask(__name__)
r = Redis(host = getenv('REDIS_HOSTNAME', 'redis'), port = int(getenv('REDIS_PORT', 6379)))
DEFAULT_ARGS = {'format': getenv('REC_FORMAT', 'json'),
				 'kwargs': {},
				 'col_order': [getenv('REC_COL_USER', 'userID'), getenv('REC_COL_ARTICLE', 'articleID'), getenv('REC_COL_RATING', 'ratings')], 
				 'k_cores': int(getenv('REC_K_CORES', 0)), 
				 'save_map': bool(getenv('REC_SAVE_MAP', True)), 
				 'train_size': float(getenv('REC_TRAIN_SIZE', 0.99)), 
				 'dtype': np.dtype(getenv('REC_DTYPE', np.float32)), 
				 'debug': bool(getenv('REC_DEBUG', False)), 
				 'niter': int(getenv('REC_NITER', 20)), 
				 'ncomponents': int(getenv('REC_NCOMPONENTS', 20)), 
				 'unobserved_weight': float(getenv('REC_UNOBSEREVED_WEIGHT', 0)), 
				 'regularization': float(getenv('REC_REGULARIZATION', 0.05)),
				 'beta': float(getenv('REC_BETA', 0.015)),
				 'nbins': int(getenv('REC_NBINS', 6)),
				 'reg_bias': float(getenv('REC_REG_BIAS', 0.01)),
				 'learn_rate': float(getenv('REC_LEARN_RATE', 0.01)),
				 'max_learn_rate': float(getenv('REC_MAX_LEARN_RATE', 1000.0)),
				 'reg_user': float(getenv('REC_REG_USER', 0.01)),
				 'reg_item': float(getenv('REC_REG_ITEM', 0.01)),
				 'bold': bool(getenv('REC_BOLD', False)),
				 'tol': float(getenv('REC_TOL', 1e-5))
				 }
PREPROCESS_ARGS = {'format', 'kwargs', 'col_order', 'k_cores', 'save_map', 'train_size', 'dtype', 'debug'}
WALS_TRAIN_ARGS = {'niter', 'ncomponents', 'unobserved_weight', 'regularization'}
TEMPORAL_TRAIN_ARGS = {'beta', 'nbins', 'reg_bias', 'niter', 'learn_rate', 'max_learn_rate', 'reg_user', 'reg_item', 'ncomponents', 'bold', 'tol'}
VIEW_WEIGHT = int(getenv('REC_VIEW_WEIGHT', 1))
DEFAULT_RECS = int(getenv('REC_NRECS', 5))
AUTH_TOKEN = getenv('LOG_AUTH_TOKEN', '')
DEFAULT_MODEL = getenv('DEFAULT_MODEL', 'wals')

#To train run this
#curl -i -X POST -H 'Content-Type: application/json' -d '{"article-view": "http://localhost:8000/logapi/event/article/view"}' http://locaost:3445/train

@app.route('/rec')
def get_recommendations():
	model = request.args.get('model', DEFAULT_MODEL)
	if model == 'wals':
		result = core.predict.predict(redis_get_helper('U'), redis_get_helper('V'), str(request.args.get('user')), request.args.get('nrecs', DEFAULT_RECS), user_map = json.loads(r.get('user_map').decode()), item_map = json.loads(r.get('item_map').decode()))
	elif model == 'timesvd':
		u = json.loads(r.get('t_user_map').decode()).get(str(request.args.get('user')), -1)
		min_stamp = float(r.get('t_min_stamp'))
		max_stamp = float(r.get('t_max_stamp'))
		user_mean_time = redis_get_helper('t_user_mean_time')
		beta = float(r.get('t_beta'))
		global_mean_time = float(r.get('t_global_mean_time'))
		item_biases = redis_get_helper('t_item_biases')
		b_it = redis_get_helper('t_b_it')
		c_u = redis_get_helper('t_c_u')
		c_ut = redis_get_helper('t_c_ut')
		user_biases = redis_get_helper('t_user_biases')
		alpha_u = redis_get_helper('t_alpha_u')
		y = redis_get_helper('t_y')
		U = redis_get_helper('t_U')
		V = redis_get_helper('t_V')
		alpha_uk = redis_get_helper('t_alpha_uk')
		train = redis_get_helper('t_train', True)
		b_ut = redis_get_helper('t_b_ut').ravel()[0]
		p_ukt = redis_get_helper('t_p_ukt').ravel()[0]

		pred = core.train_temporal.get_recommendations(u, request.args.get('nrecs', DEFAULT_RECS), datetime.datetime.utcnow().timestamp(), train, min_stamp, max_stamp, user_mean_time, beta, global_mean_time, item_biases, b_it, c_u, c_ut, b_ut, user_biases, alpha_u, y, U, V, p_ukt, alpha_uk)
		item_map = json.loads(r.get('t_item_map').decode())
		result = json.dumps({"map":{str(x) : item_map[x] for x in pred}, "pred":list(map(str, pred))})
	else:
		raise ValueError("Could not recognize model: {}.".format(model))
	return result

def redis_set_helper(key, data, pipe, npz = False):
	with BytesIO() as b:
		if npz:
			save_npz(b, data)
		else:
			np.save(b, data)
		pipe.set(key, b.getvalue())

def redis_get_helper(key, npz = False):
	if npz:
		return load_npz(BytesIO(r.get(key)))
	return np.load(BytesIO(r.get(key)))

def load_default_args(args):
	for k, v in DEFAULT_ARGS.items():
		if k not in args:
			args[k] = v
	return args

def fetch_logs(link, time = False):
	logs = []
	while link:
		q = Request(link, headers = {'Authorization': 'Token ' + AUTH_TOKEN})
		request = urlopen(q)
		result = json.loads(request.read().decode())
		logs.append(result['result'])
		link = result.get('next_link', '')

	articleIDs = [str(x['event']['article-id']) for log in logs for x in log]
	userIDs = [str(x['user-id'] or x['ip-address']) for log in logs for x in log]
	ratings = [VIEW_WEIGHT for log in logs for _ in range(len(log))]

	if time:
		timestamps = [(datetime.datetime.utcnow() - datetime.datetime.strptime(x['time-stamp'], '%Y-%m-%d %H:%M:%S')).days for log in logs for x in log]
		return articleIDs, userIDs, ratings, timestamps
	return articleIDs, userIDs, ratings

def train_wals(args):
	args = load_default_args(args)
	articleIDs, userIDs, ratings = fetch_logs(args['article-view'])
	df = json.dumps({args['col_order'][0] : userIDs, args['col_order'][1] : articleIDs, args['col_order'][2] : ratings})
	result = core.preprocess.preprocess(df, **{k : args[k] for k in PREPROCESS_ARGS})
	U, V = core.train.train_model(result['train'], **{k : args[k] for k in WALS_TRAIN_ARGS})
	pipe = r.pipeline()
	redis_set_helper('U', U, pipe)
	redis_set_helper('V', V, pipe)
	redis_set_helper('train', result['train'], pipe, True)
	redis_set_helper('test', result['test'], pipe, True)
	pipe.set('user_map', json.dumps(result['user_map'])).set('item_map', json.dumps(result['item_map']))
	pipe.execute()
	r.set('train_error', core.train.rmse(U, V, result['train']))
	r.set('test_error', core.train.rmse(U, V, result['test']))

def train_timesvd(args):
	args = load_default_args(args)
	articleIDs, userIDs, ratings, timestamps = fetch_logs(args['article-view'], time = True)
	if len(args['col_order']) < 4:
		args['col_order'].append('timestamp')
	df = json.dumps({args['col_order'][0] : userIDs, args['col_order'][1] : articleIDs, args['col_order'][2] : ratings, args['col_order'][3]: timestamps})
	pre_result = core.preprocess.preprocess(df, timestamp = True, **{k : args[k] for k in PREPROCESS_ARGS})
	train_result = core.train_temporal.train_model(pre_result['train'], pre_result['timestamp'], **{k : args[k] for k in TEMPORAL_TRAIN_ARGS})
	pipe = r.pipeline()
	redis_set_helper('t_train', pre_result['train'], pipe, True)
	redis_set_helper('t_test', pre_result['train'], pipe, True)
	redis_set_helper('t_user_mean_time', train_result['user_mean_time'], pipe)
	redis_set_helper('t_item_biases', train_result['item_biases'], pipe)
	redis_set_helper('t_b_it', train_result['b_it'], pipe)
	redis_set_helper('t_c_u', train_result['c_u'], pipe)
	redis_set_helper('t_c_ut', train_result['c_ut'], pipe)
	redis_set_helper('t_user_biases', train_result['user_biases'], pipe)
	redis_set_helper('t_alpha_u', train_result['alpha_u'], pipe)
	redis_set_helper('t_y', train_result['y'], pipe)
	redis_set_helper('t_U', train_result['U'], pipe)
	redis_set_helper('t_V', train_result['V'], pipe)
	redis_set_helper('t_alpha_uk', train_result['alpha_uk'], pipe)
	redis_set_helper('t_b_ut', dict(train_result['b_ut']), pipe)
	redis_set_helper('t_p_ukt', {k: dict(v) for k, v in train_result['p_ukt'].items()}, pipe)
	pipe.set('t_min_stamp', train_result['min_stamp']).set('t_max_stamp', train_result['max_stamp']).set('t_beta', train_result['beta']).set('t_global_mean_time', train_result['global_mean_time']).set('t_user_map', json.dumps(pre_result['user_map'])).set('t_item_map', json.dumps(pre_result['item_map']))
	pipe.execute()
	r.set('t_train_error', core.train_temporal.rmse(pre_result['train'], pre_result['timestamp'], pre_result['train'], **train_result))
	r.set('t_test_error', core.train_temporal.rmse(pre_result['test'], pre_result['timestamp'], pre_result['train'], **train_result))


@app.route('/train', methods = ['POST'])
def train():
	model = request.json.get('model', DEFAULT_MODEL)
	if model == 'wals':
		Process(target = train_wals, args = (request.json,)).start()
	elif model == 'timesvd':
		Process(target = train_timesvd, args = (request.json,)).start()
	else:
		raise ValueError("Could not recognize model: {}.".format(model))
	return "OK"

@app.route('/visual')
def visualise():
	train = redis_get_helper('train', True).tocsr()
	test = redis_get_helper('test', True).tocsr()
	img = core.visualise.visualise(redis_get_helper('U'), redis_get_helper('V'), item_map = json.loads(r.get('item_map').decode()), M = train + test, r = request.args.get('r', 1), idx = request.args.get('user', -1))
	return send_file(img, mimetype = 'image/png')
