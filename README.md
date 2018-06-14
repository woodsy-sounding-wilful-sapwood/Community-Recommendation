# Community-Recommendation
recommendations of content tied to a user

## Requirements 
* [NumPy (1.14.3+)](http://www.numpy.org/)
* [SciPy (1.1.0+)](https://www.scipy.org/)
* [sh (1.12.14+)](https://amoffat.github.io/sh/)
* [TensorFlow (1.8.0+)](https://www.tensorflow.org/)
* [Pandas (0.22.0+)](https://pandas.pydata.org/)
* [Scikit-Learn (0.19.1+)](http://scikit-learn.org/stable/index.html)

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

