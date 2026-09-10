
import itertools
import os

import math
from typing import List

import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

import json
from datetime import datetime

DATA_PATH_18 = 'datasets/Twitter_2018'
KEYS = ["tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
              "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
              "sampled_words"]
WINDOW_SIZE = 3

def convert_np_to_json(np_data):
    print("Converting data to JSON format...")

    assert all(len(row) == len(KEYS) for row in np_data), "Each row must have 16 elements!"

    data_list = [dict(zip(KEYS, row)) for row in np_data]

    return data_list

def split_train_test_validation(data: List):
    block=[]
    for i in range(len(data)):
        if i == 0:
            data_size = len(data[i])
            valid_size = math.ceil(data_size *0.2)
            train, valid = train_test_split(data[i], test_size=valid_size, random_state=42, shuffle=True)
            block.append({"train": train, "test": [], "valid": valid})

        elif i % WINDOW_SIZE == 0:

            sub_data = []
            for j in range(WINDOW_SIZE):
                sub_data += data[i-j]
            sub_data_size = len(sub_data)
            sub_valid_size = math.ceil( sub_data_size * 0.2)
            train, valid = train_test_split(sub_data, test_size=sub_valid_size, random_state=42, shuffle=True)
            block.append({"train": train, "test": data[i], "valid": valid})
        else:
            block.append({"train": [], "test": data[i], "valid": []})

    return block


def split_into_blocks(data):
    data = sorted(data, key=lambda x: x['created_at'])
    groups = itertools.groupby(data, key=lambda x: x['created_at'].timetuple().tm_yday)
    groups = {k: list(g) for k, g in groups}
    days = sorted(groups.keys())
    blk0 = [groups[d] for d in days[:7]]
    blk0 = [it for b in blk0 for it in b]

    day_blk = [groups[d] for d in days[7:]]

    blocks = [blk0] + day_blk
    datacount = [len(sublist) for sublist in blocks]

    return split_train_test_validation(blocks)


def pre_process(data):
    print("split data into blocks... ")
    data_json = convert_np_to_json(data)
    blocks = split_into_blocks(data_json)
    print("\tDone")

    return blocks

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    np_data_file = f'{DATA_PATH_18}/All_French.npy'

    print(f"load data from {DATA_PATH_18} ... ", end='')
    np_data = np.load(np_data_file, allow_pickle=True)
    print("\tDone")
    if not os.path.exists('datasets/cache'):
        os.makedirs('datasets/cache')

    blk_data = pre_process(np_data)


    output_path = os.path.join('datasets/cache', 'twitter18.json')
    print(f"save data to 'cache/twitter18.json' ... ", end='')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blk_data, f, ensure_ascii=False, indent=4, default=convert_to_serializable)

    print(f"Data successfully saved to {output_path}")

    exit(0)

