import argparse
import json

import numpy as np

from scipy.sparse import load_npz

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--u',
        help = 'Location of the row factors',
        required = True
        )
    parser.add_argument(
        '--v',
        help = 'Location of the column factors',
        required = True
        )
    parser.add_argument(
        '--user-id',
        help = 'ID of the user for whom predictions are needed',
        required = True
        )
    parser.add_argument(
        '--user-map',
        help = 'Location of the user ids :-> [0...nusers] map. If it is not provided, user id = user index.',
        required = False
        )
    parser.add_argument(
        '--item-map',
        help = 'Location of the item ids :-> [0...nitems] map. If it is not provided, user id = user index.',
        required = False
        )
    parser.add_argument(
        '--n-recs',
        help = 'Number of recommendations needed. Default: 5',
        default = 5
        )
    parser.add_argument(
        '--fallback',
        help = 'If the algorithm has not been trained on the user, we use a simple recommendation system. If a fallback is specified, then that file is assumed to contain a list of movies that we can recommend to all users. Otherwise, we take the average over U dot V.',
        default = None,
        required = False
        )
    parser.add_argument(
        '--dataset',
        help = 'If provided, this is used to make sure content already rated by the user is not recommended again.',
        default = None,
        required = False
        )

    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    args = get_args()

    user_map = load_json(args.user_map) if args.user_map else None
    user_idx = user_map.get(args.user_id, -1) if user_map else int(args.user_id)

    if user_idx != -1 or not args.fallback:
        U = np.load(args.u)
        V = np.load(args.v)
        item_map = load_json(args.item_map) if args.item_map else None

        if args.dataset and user_idx != -1:
            M = load_npz(args.dataset).tocsr()
            already_seen = M[user_idx].tooo().cols
        else:
            already_seen = []

        if user_idx == -1:
            u = U.mean(axis = 0)
        else:
            u = U[user_idx, :]

        pred = np.argsort(V.dot(u))[-1 : -len(already_seen) - args.n_recs - 1 : -1]
    else:
        pred = np.array(load_json(args.fallback))[:args.n_recs]

    if item_map:
        print(np.array(item_map)[pred])
    print(pred)

if __name__ == '__main__':
    main() 
