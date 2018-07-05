# Community-Recommendation
recommendations of content tied to a user

## Requirements 
* [NumPy (1.14.3+)](http://www.numpy.org/)
* [SciPy (1.1.0+)](https://www.scipy.org/)
* [sh (1.12.14+)](https://amoffat.github.io/sh/)
* [TensorFlow (1.8.0+)](https://www.tensorflow.org/)
* [Pandas (0.22.0+)](https://pandas.pydata.org/)
* [Scikit-Learn (0.19.1+)](http://scikit-learn.org/stable/index.html)
* [Matplotlib (2.2.2+)](https://matplotlib.org)

Additionally, to run the api you need:

* [Flask (1.0.2+)](http://flask.pocoo.org/)
* [Redis-Py (2.10.5+)](https://redislabs.com/lp/python-redis/)

## Usage

### Overview

There are essentially three steps involved:

1. Preprocessing: `preprocess.py`  Accepts either a JSON or a CSV file as input and outputs two Scipy sparse matrices: one for training another for testing.
2. Training: `task.py` Accepts the Scipy sparse matrices and trains the model on them
3. Prediction: `predict.py`  Will output the top `n` ratings for a specified user id.

You can run any of these scripts with the `-h` or `--help` option for more information on supported options.

### Example

First of all, you need a dataset. You can use any that catches your fancy. We will be working the [5-core amazon music dataset](http://jmcauley.ucsd.edu/data/amazon/)

Now, we need to take this JSON and transform it into a trainging matrix and a testing matrix. 

```console
$ python preprocess.py --data Digital_Music_5.json --format json --col-order reviewerID asin overall --lines True
```

There should now be two files in your directory: `train.npz` and `test.npz`. We now train the model.

```console
$ python task.py --train-data train.npz --test-data test.npz
```

Now, there should be a new directory called `model` with two files: `row.npy` and `col.npy`. To get predictions, run

```console
$ python predict.py --u model\row.npy --v model\col.npy --user-id 12
[ 990 1973 1255 2268  644]
```
`12` here is the row idex of the user in our matrix and `[ 990 1973 1255 2268  644]` are the column indices of the recommended music in our matrix. To recover the actual user id and music id, you need to set `--save-map` in `preprocess.py` and supply the two maps to `predict.py`. See help for more details. (Accesible by running `python preprocess.py --help` and `python task.py --help`)


### API Overview
At present, the API is tightly coupled with the [Collaborative Communities project](https://github.com/fresearchgroup/Collaboration-System). It is only useful for making recommendations based on a CC user's viewing history. To use the API, you will need to have the [event logging module](https://github.com/fresearchgroup/Collaboration-System/tree/eventlogs) installed.

To get recommendations, make a `GET` request to the server with the user id and (optionally) the number of recommendations needed.

eg: `http://localhost:3445/rec?user=12&nrecs=3`

**Before the API is able to generate recommendations, it must be trained.** To train the API make a POST request to the server specifying the URI of the logs and optionally specify the parameters for preprocessing and training.

eg: `curl -i -X POST -H 'Content-Type: application/json' -d '{"article-view": "http://localhost:8000/logapi/event/article/view/?after=1970-01-01T00:00:00"}' http://localhost:3445/train`

To visualise recommendations, make a `GET` request to the server. Optionally, you may specify the user id and the percentage of items to display. 

eg: `http://localhost:3445/visual?user=1&r=3`

### Installation

#### Installation in a Virtual environment


* Install redis: `sudo apt−get install redis−server`)
* Create a virtual environment: `virtualenv --system-site-packages -p python3 rec_api`
* Activate the virtual environment: `source ~/rec_api/bin/activate`
* Clone this repo: `git clone https://github.com/fresearchgroup/Community-Recommendation.git`
* Change into the directory: `cd Community-Recommendation`
* Install dependencies: `pip3 install -r requirements.txt`
* Set up Flask: `export FLASK_APP=flask_api.py` and, optionally, `export FLASK_ENV=development`
* Set the token for event logs: `export LOG_AUTH_TOKEN=Your_Token_Here`

* Run the server (eg: `flask run --host 0.0.0.0 --port 3445`)

#### Installation using Docker

* Install [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [Docker-Compose](https://docs.docker.com/compose/install/#install-compose)
* Clone this repo: `git clone https://github.com/woodsy-sounding-wilful-sapwood/Community-Recommendation.git`
* Change into the directory: `cd Community-Recommendation`
* Add the event logs token: ` echo ”Your Token Here” >> .env`
* Build the system: `sudo docker-compose build`
* Run the system: `sudo docker-compose up`

