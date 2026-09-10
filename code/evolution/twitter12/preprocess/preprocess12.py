import json
import os
from datetime import datetime
import pandas as pd

def load_data_blocks(dataset):
    print(f"load data from '{dataset}'... ", end='')
    path_to_data = os.path.join("datasets/cache/", f"{dataset}.json")
    with open(path_to_data, 'r', encoding='utf-8') as file:
        data_blocks = json.load(file)
    print("\tDone")
    return data_blocks

data_blocks = load_data_blocks("twitter12")

json_list = []
days_list = []

for i in range(1,22):
    for m in data_blocks[i]['test']:
        json_list.append({'text': m['text']})
        days = pd.to_datetime(m['created_at'])
        days_list.append(days.date())

unique_days = sorted(set(days_list))
day_to_id = {day: idx for idx, day in enumerate(unique_days)}

time_slices = [day_to_id[day] for day in days_list]


import json

with open("evolution/twitter12/text.jsonlist", "w", encoding="utf-8") as f:
    for item in json_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("evolution/twitter12/times.txt", "w", encoding="utf-8") as f:
    for num in time_slices:
        f.write(f"{num}\n")


from preprocess import Preprocess
import scipy.sparse

processor = Preprocess(vocab_size=10000)

rst = processor.preprocess_jsonlist("evolution/twitter12/")

scipy.sparse.save_npz('evolution/twitter12/bow.npz', scipy.sparse.csr_matrix(rst['train_bow']))

with open("evolution/twitter12/vocab.txt", "w", encoding="utf-8") as f:
    for word in rst['vocab']:
        f.write(word.strip() + "\n")

with open("evolution/twitter12/texts.txt", "w", encoding="utf-8") as f:
    for word in rst['train_texts']:
        f.write(word.strip() + "\n")


scipy.sparse.save_npz("evolution/twitter12/word_embeddings.npz", rst['word_embeddings'])

print(rst)