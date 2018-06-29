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
        type = int,
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
    parser.add_argument(
        '--lookup-table',
        help = 'Stores item names indexed by item-id',
        default = None,
        required = False
        )

    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def int_or_neg(s):
    try:
        return int(s)
    except ValueError:
        return -1

def predict(U, V, user_idx, n_recs = 5, user_map = None, item_map = None, lookup_table = None, M = None, fallback = None):
    user_idx = user_map.get(user_idx, -1) if user_map else int_or_neg(user_idx)
    try:
        n_recs = int(n_recs)
    except ValueError:
        n_recs = 5
    if user_idx >= 0 or (fallback is None):
        already_seen = M[user_idx].tocoo().cols if (M is not None) and user_idx >= 0 else []
        u = U[user_idx, :] if user_idx >= 0 else U.mean(axis = 0)
        pred = np.argsort(V.dot(u))[-1 : -len(already_seen) - n_recs - 1 : -1]
        pred = [x for x in pred if x not in already_seen][:n_recs]
    else:
        pred = fallback[:n_recs]

    if item_map is not None:
        if lookup_table is not None:
            pred_map = {str(x) : lookup_table.get(item_map[x], item_map[x]) for x in pred}
        else:
            pred_map = {str(x) : item_map[x] for x in pred}
        return json.dumps({'predictions' : list(map(str, pred)), 'map' : pred_map})
    else:
        return json.dumps({'predictions' : list(map(str, pred))})

def main():
    args = get_args()

    user_map = load_json(args.user_map) if args.user_map else None
    user_idx = user_map.get(args.user_id, -1) if args.user_map else int_or_neg(args.user_id)
    item_map = np.array(load_json(args.item_map)) if args.item_map else None
    lookup_table = load_json(args.lookup_table) if args.lookup_table else None

    if user_idx != -1 or not args.fallback:
        U = np.load(args.u)
        V = np.load(args.v)
        M = load_npz(args.dataset).tocsr() if args.dataset and user_idx > 0 else None
        fallback = None
    else:
        U, V, M = None
        fallback = np.array(load_json(args.fallback))

    print(predict(U, V, args.user_id, n_recs = args.n_recs, user_map = user_map, item_map = item_map,lookup_table = lookup_table, M = M, fallback = fallback))

if __name__ == '__main__':
    main() 
