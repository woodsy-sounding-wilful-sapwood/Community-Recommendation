import datetime
import json

import numpy as np

from flask import Flask, request
from subprocess import check_output
from urllib.request import urlopen, Request
from redis import Redis
from io import BytesIO
from multiprocessing import Process
from os import getenv

import core.predict
import core.preprocess
import core.train

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
				 'regularization': float(getenv('REC_REGULARIZATION', 0.05))
				 }
PREPROCESS_ARGS = {'format', 'kwargs', 'col_order', 'k_cores', 'save_map', 'train_size', 'dtype', 'debug'}
WALS_TRAIN_ARGS = {'niter', 'ncomponents', 'unobserved_weight', 'regularization'}
TEMPORAL_TRAIN_ARGS = {'beta', 'nbins', 'reg_bias', 'niter', 'learn_rate', 'max_learn_rate', 'reg_user', 'reg_item', 'ncomponents', 'bold', 'tol'}
VIEW_WEIGHT = int(getenv('REC_VIEW_WEIGHT', 1))
DEFAULT_RECS = int(getenv('REC_NRECS', 5))
AUTH_TOKEN = getenv('LOG_AUTH_TOKEN', '')
MODEL_TYPE = getenv('DEFAULT_MODEL', 'wals')

#To train run this
#curl -i -X POST -H 'Content-Type: application/json' -d '{"article-view": "http://localhost:8000/logapi/event/article/view"}' http://locaost:3445/train

@app.route('/rec')
def get_recommendations():
	result = core.predict.predict(redis_get_helper('U'), redis_get_helper('V'), request.args.get('user'), request.args.get('nrecs', DEFAULT_RECS), user_map = json.loads(r.get('user_map').decode()), item_map = json.loads(r.get('item_map').decode()))
	return result

def redis_set_helper(key, data, pipe):
	with BytesIO() as b:
		np.save(b, data)
		pipe.set(key, b.getvalue())

def redis_get_helper(key):
	return np.load(BytesIO(r.get(key)))

def train_wals(args):
	for k, v in DEFAULT_ARGS.items():
		if k not in args:
			args[k] = v
	logs = []
	link = args['article-view']
	while link:
		q = Request(link)
		q.add_header('Auth', 'Token ' + AUTH_TOKEN)
		request = urlopen(q)
		logs += json.loads(request.read().decode())['result']
		link = log[-1].get('next_link', '')

	articleIDs = [str(x['event']['article-id']) for log in logs for x in log]
	userIDs = [str(x['user-id'] or x['ip-address']) for log in logs for x in log]
	ratings = [VIEW_WEIGHT for log in logs for _ in range(len(log))]
	df = json.dumps({args['col_order'][0] : userIDs, args['col_order'][1] : articleIDs, args['col_order'][2] : ratings})
	result = core.preprocess.preprocess(df, **{k : args[k] for k in PREPROCESS_ARGS})
	U, V = core.train.train_model(result['train'], **{k : args[k] for k in WALS_TRAIN_ARGS})
	pipe = r.pipeline()
	redis_set_helper('U', U, pipe)
	redis_set_helper('V', V, pipe)
	pipe.set('user_map', json.dumps(result['user_map'])).set('item_map', json.dumps(result['item_map']))
	pipe.execute()
	r.set('train_error', core.train.rmse(U, V, result['train']))
	r.set('test_error', core.train.rmse(U, V, result['test']))

@app.route('/train', methods = ['POST'])
def train():
	Process(target = train_wals, args = (request.json,)).start()
	return "OK"
