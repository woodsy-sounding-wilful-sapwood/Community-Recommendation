import numpy as np

from datetime import timedelta
from collections import defaultdict
		

def count_days(t1, t2):
	return abs((timedelta(milliseconds = t1) - timedelta(milliseconds = t2)).days) + 1

def rand_array(size, low = 0.0, high = 1.0):
	return np.uniform.random(low = low, high = high, size = size)

def mean_time(data, time):
	if data.nnz == 0:
		return 0
	return sum(time[i, j] for i, j in zip(data.row, data.col))/data.nnz

def mean_time_helper(coord, data, time, global_mean = 0):
	return mean_time(data[coord[0], :].tocoo(), time) or global_mean

def dev(uid, day, user_mean_time, beta):
	delta = user_mean_time[uid] - day
	return np.sign(delta) * pow(abs(delta), beta)

def update_learn_rate(itr, learn_rate, max_learn_rate, loss, prev_loss, bold, beta):
	if learn_rate < 0:
		return learn_rate
	if bold and itr > 1:
		learn_rate *= (0.5 + (abs(prev_loss) > abs(loss)))
	elif decay > 0 and decay < 1:
		learn_rate *= decay
	if max_learn_rate >0 and learn_rate > max_learn_rate:
		learn_rate = max_learn_rate
	return learn_rate


def init_model(train, time_mat, beta, nbins):
	train = train.tocsr()
	time_mat = time_mat.todok()
	nusers, nitems = time_mat.shape()
	max_stamp = time_mat.max()
	min_stamp = time_mat.data.min()
	ndays = count_days(time_mat.max(), time_mat.data.min())
	user_biases = rand_array((nusers,))
	item_biases = rand_array((nitems,))
	alpha_u = rand_array((nusers,))
	b_it = rand_array((nitems, nbins))
	y = rand_array((nitems, ncomponents))
	alpha_uk = rand_array((nusers, ncomponents))
	b_ut = defaultdict(lambda: np.random.uniform())
	p_ukt = defaultdict(lambda: defaultdict(lambda: np.random.uniform()))
	c_u = rand_array((nusers,))
	c_ut = rand_array((nusers, ndays))
	U = rand_array((nusers, ncomponents))
	V = rand_array((nitems, ncomponents))
	global_mean_time = mean_time(train, time_mat) #sparse_type_coo
	user_mean_time = np.fromiter((mean_time(train.getrow(i), time_mat) or global_mean_time for i in range(nusers)), type = np.float32, count = nusers) #sparse_type_csr
	return max_stamp, min_stamp, user_biases, item_biases, alpha_u, b_it, y, alpha_uk, b_ut, p_ukt, c_u, c_ut, U, V, global_mean_time, user_mean_time

def train_model_helper(train, time_mat, user_mean_time, item_biases, b_it, user_biases, c_u, c_ut, b_ut, alpha_u, p_ukt, alpha_uk, y, U, V, reg_item, reg_user, reg_bias, learn_rate, bold, niter, tol):
	time_mat = time_mat.todok()
	train = train.tocoo()
	earliest_time = time_mat.data.min()
	prev_loss = 0
	ncomponents = U.shape[1]
	for itr in range(niter):
		for data, u, i in zip(train.data, train.row, train.col):
			timestamp = time_mat[u, i]
			t = count_days(timestamp, earliest_time)
			bin_t = int(t * b_it.shape[1] / c_ut.shape[1])
			dev_ut = dev(u, t, user_mean_time)

			_b_i = item_biases[i]
			_b_it = b_it[i, bin_t]
			_b_u = user_biases[u]
			_c_u = c_u[u]
			_c_ut = c_ut[u, t]
			_b_ut = b_ut[(u, t)]
			_alpha_u = alpha_u[u]

			R_u = train.getrow(u).tocoo().col
			_y = sum(y[j, :] * V[i, :].transpose() for j in R_u)
			_w_i = (1 / sqrt(R_u.size)) if R_u.size > 0 else 0
			_p_ui = global_mean + (_b_i + _b_it) * (_c_u + _c_ut) + _b_u +_alpha_u * dev_ut + _b_ut + _y * _w_i
			
			for k in range(ncomponents):
				_p_uk = U[u, k] + alpha_uk[u, k] * dev_ut + p_ukt[u][(k, t)]
				_p_ui += _p_uk * V[i, k]

			_e_ui = _p_ui - data
			loss += pow(_e_ui, 2)

			scale = _e_ui * (_c_u * _c_ut) + reg_bias * _b_i
			item_biases[i] += -learn_rate * scale
			loss += reg_bias * pow(_b_i, 2)

			scale =  _e_ui * (_c_u * _c_ut) + reg_bias * _b_it
			b_it[i, bin_t] += -learn_rate * scale
			loss += reg_bias * pow(_b_it, 2)

			scale = _e_ui * (_b_i + _b_it) + reg_bias * _c_u
			c_u[u] += -learn_rate * scale
			loss += reg_bias * pow(_c_u, 2)

			scale = _e_ui * (_b_i + _b_it) + reg_bias * _c_ut
			c_ut[u, t] += -learn_rate * scale
			loss += reg_bias * pow(_c_ut, 2)

			scale = _e_ui + reg_bias * _b_u
			user_biases[u] += -learn_rate * scale
			loss += reg_bias * pow(_b_u, 2)

			scale = _e_ui * dev_ut + reg_bias * _alpha_u
			alpha_u[u] += -learn_rate * scale
			loss += reg_bias * pow(_alpha_u, 2)

			scale = _e_ui * dev_ut + reg_bias * _b_ut
			b_ut[(u, t)] = _b_ut - learn_rate * scale
			loss += reg_bias * pow(_b_ut, 2)

			for k in range(ncomponents):
				_v_ik = V[i, k]
				_p_uk = U[u, k]
				_alpha_uk = alpha_uk[u, k]
				_p_kt = p_ukt[u][(k, t)]

				_p_ukt = _p_uk + _alpha_uk * dev_ut + _p_kt
				_y_k = sum(y[j, k] for j in R_u)

				scale = _e_ui * (_p_ukt + _w_i * _y_k) + reg_item * _v_ik
				V[i, k] += -learn_rate * scale
				loss += reg_item * pow(_v_ik, 2)

				scale = _e_ui * _v_ik + reg_user * _p_uk
				U[u, k] += -learn_rate * scale
				loss += reg_user * pow(_p_uk, 2)

				scale = _e_ui * _v_ik * dev_ut + reg_user * _alpha_uk
				alpha_uk[u, k] += -learn_rate * scale
				loss += reg_user * pow(_alpha_uk, 2)

				scale = _e_ui * _v_ik + reg_user * _p_kt
				p_ukt[u][(k, t)] = _p_kt - learn_rate * scale
				loss += reg_user * pow(_p_kt, 2)

				for j in R_u:
					_y_jk = Y[j, k]
					scale = _e_ui * _w_i * _v_ik + reg_item * _y_jk
					Y[j, k] += -learn_rate * scale
					loss += reg_item * pow(_y_jk, 2)

		loss = loss / 2
		delta_loss = abs(loss - prev_loss)
		if delta_loss <= tol:
			break
		elif np.isnan(delta_loss) or np.isinf(delta_loss):
			raise FloatingPointError("Loss is not a finite number. Try training with different hyperparameters.")
		learn_rate = update_learn_rate(itr, learn_rate, max_learn_rate, loss, prev_loss, bold, beta)
		prev_loss = loss
		return b_it, c_u, c_ut, item_biases, b_it, c_u, c_ut, b_ut, user_biases, alpha_u, y, U, V, p_ukt

def train_model(train, time_mat, beta = 0.015, nbins = 6, reg_bias = 0.01, niter = 100, learn_rate = 0.01, max_learn_rate = 1000.0, reg_user = 0.01, reg_item = 0.01, ncomponents = 10, bold = False, tol = 1e-5):
	max_stamp, min_stamp, user_biases, item_biases, alpha_u, b_it, y, alpha_uk, b_ut, p_ukt, c_u, c_ut, U, V, global_mean_time, user_mean_time = init_model(train, time_mat, beta, nbins)
	b_it, c_u, c_ut, item_biases, b_it, c_u, c_ut, b_ut, user_biases, alpha_u, y, U, V, p_ukt = train_model_helper(train, time_mat, user_mean_time, item_biases, b_it, user_biases, c_u, c_ut, b_ut, alpha_u, p_ukt, alpha_uk, y, U, V, reg_item, reg_user, reg_bias, learn_rate, bold, niter, tol)
	return_set = {'min_stamp', 'max_stamp', 'user_mean_time', 'beta', 'global_mean_time', 'item_biases', 'b_it', 'c_u', 'c_ut', 'b_ut', 'user_biases', 'alpha_u', 'y', 'U', 'V', 'p_ukt', 'alpha_uk'}
	return {x : locals()[x] for x in return_set}


def predict_item(u, i, timestamp, min_stamp, user_mean_time, beta, global_mean_time, item_biases, b_it, c_u, c_ut, b_ut, user_biases, alpha_u, y, train, U, V, p_ukt, alpha_uk):
	ndays = c_ut.shape[1]
	t = days(timestamp, min_stamp)
	bin_t = int(t * b_it.shape[1]  / ndays) if t < ndays else ndays
	dev_ut = dev(u, t, user_mean_time, beta)
	_c_ut = c_ut[t] if t < c_ut.size else 0

	R_u = train.getrow(u).tocoo().col
	_y = sum(y[j, :] * V[i, :].transpose() for j in R_u)
	_w_i = (1 / sqrt(R_u.size)) if R_u.size > 0 else 0

	pred = global_mean_time + (item_biases[i] + b_it[i, bin_t]) * (c_u[u] + _c_ut) + user_biases[u] + alpha_u[u] * dev_ut + b_ut.get((u, t), 0) + _y * _w_i

	for k in range(U.shape[1]):
		pred += (U[u, k] + alpha_uk[u, k] * dev_ut + p_ukt[u].get((k, t), 0)) * V[i, k]

	return pred

def get_recommendations(u, nrecs, timestamp, train, min_stamp, max_stamp, user_mean_time, beta, global_mean_time, item_biases, b_it, c_u, c_ut, b_ut, user_biases, alpha_u, y, U, V, p_ukt, alpha_uk):
	already_seen = set(train.getrow(u).tocoo().col)
	timestamp = min(timestamp, max_stamp)
	pred = np.fromiter(predict_item(u, i, timestamp, min_stamp, user_mean_time, beta, global_mean_time, item_biases, b_it, c_u, c_ut, b_ut, user_biases, alpha_u, y, train, U, V, p_ukt, alpha_uk) for i in range(train.shape[1]))
	pred = np.argsort(pred)[-1 : -len(already_seen) - nrecs - 1 : -1]
	pred = [x for x in pred if x not in already_seen][:nrecs]
	return pred
