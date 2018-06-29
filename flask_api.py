import datetime
import json

import numpy as np

from flask import Flask, request
from subprocess import check_output
from urllib.request import urlopen
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
				 'n_components': int(getenv('REC_N_COMPONENTS', 20)), 
				 'unobserved_weight': float(getenv('REC_UNOBSEREVED_WEIGHT', 0)), 
				 'regularization': float(getenv('REC_REGULARIZATION', 0.05))
				 }
PREPROCESS_ARGS = {'format', 'kwargs', 'col_order', 'k_cores', 'save_map', 'train_size', 'dtype', 'debug'}
TRAIN_ARGS = {'niter', 'n_components', 'unobserved_weight', 'regularization'}
VIEW_WEIGHT = int(getenv('REC_VIEW_WEIGHT', 1))
DEFAULT_RECS = int(getenv('REC_NRECS', 5))

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

def train_helper(args):
	for k, v in DEFAULT_ARGS.items():
		if k not in args:
			args[k] = v

	request = urlopen(args['article-view'])
	logs = json.loads(request.read().decode())['result']

	articleIDs = [str(x['event']['article-id']) for x in logs]
	userIDs = [str(x['user-id'] or x['ip-address']) for x in logs]
	ratings = [VIEW_WEIGHT for _ in range(len(logs))]
	df = json.dumps({args['col_order'][0] : userIDs, args['col_order'][1] : articleIDs, args['col_order'][2] : ratings})
	result = core.preprocess.preprocess(df, **{k : args[k] for k in PREPROCESS_ARGS})
	U, V = core.train.train_model(result['train'], **{k : args[k] for k in TRAIN_ARGS})
	pipe = r.pipeline()
	redis_set_helper('U', U, pipe)
	redis_set_helper('V', V, pipe)
	pipe.set('user_map', json.dumps(result['user_map'])).set('item_map', json.dumps(result['item_map']))
	pipe.execute()
	r.set('train_error', core.train.rmse(U, V, result['train']))
	r.set('test_error', core.train.rmse(U, V, result['test']))

@app.route('/train', methods = ['POST'])
def train():
	Process(target = train_helper, args = (request.json,)).start()
	return "OK"
