import re
import json
from os.path import exists
from utils import preprocess_sentence, preprocess_french_sentence, SBERT_embed
import pickle
from evolution.SEminimization import hier_2D_SE_mini, get_global_edges, search_stable_points
from utils import evaluate, decode
from datetime import datetime
import math
import numpy as np
import argparse

def get_chunk_embeddings(args, text_list, b):
    save_path = f'evolution/{args.dataset}/M{b}_chunk/'
    SBERT_embedding_path = f'{save_path}SBERT_embeddings.pkl'
    text_list = [preprocess_sentence(s) for s in text_list]
    print('chunk contents preprocessed.')

    if args.dataset == 'twitter12':
        embeddings = SBERT_embed(text_list, language = 'English')
    if args.dataset == 'twitter18':
        embeddings = SBERT_embed(text_list, language = 'French')

    with open(SBERT_embedding_path, 'wb') as fp:
        pickle.dump(embeddings, fp)
    print('SBERT embeddings stored.')
    return

def get_stable_point(path):
    stable_point_path = path + 'stable_point.pkl'
    if not exists(stable_point_path):
        embeddings_path = path + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        first_stable_point, global_stable_point = search_stable_points(embeddings)
        stable_points = {'first': first_stable_point, 'global': global_stable_point}
        with open(stable_point_path, 'wb') as fp:
            pickle.dump(stable_points, fp)
        print('Stable points stored.')

    with open(stable_point_path, 'rb') as f:
        stable_points = pickle.load(f)
    print('Stable points loaded.')
    return stable_points

def run_hier_2D_SE_mini(args, b, block_n, all_node_features, n = 800, e_a = True, e_s = True):
    save_path = f'evolution/{args.dataset}/M{b}_chunk/'
    
    embeddings_path = save_path + 'SBERT_embeddings.pkl'
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    all_node_features = all_node_features

    global_edges = get_global_edges(all_node_features, embeddings, 1, e_a = e_a, e_s = e_s)

    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0]
    
    division = hier_2D_SE_mini(block_n, weighted_global_edges, len(embeddings), n = n)

    prediction = decode(division)

    return prediction, division 

def chunk_SE(args, chunk, last_block_node, b):
    merged_data = {}

    for item in chunk:
        if not item.get("important_keywords") or not item["important_keywords"][0]:
            continue
        important_keywords = item["important_keywords"][0]
        content = item["content"]
        if "KEYWORDs:" not in content:
            continue
        keywords = content.split("KEYWORDs:")[1].strip()

        cleaned_keywords = re.sub(r"[^\w\s,À-ÿ_]", "", keywords).lower()

        keyword_list = list(set(cleaned_keywords.split(", ")))
        
        if important_keywords in merged_data:
            merged_data[important_keywords].extend(keyword_list)
            merged_data[important_keywords] = list(set(merged_data[important_keywords]))
        else:
            merged_data[important_keywords] = keyword_list

    block_n = len(merged_data)

    if last_block_node is not None:
         merged_data.update(last_block_node)

    text_list = [f"{key}, {', '.join(value)}" for key, value in merged_data.items()]
    get_chunk_embeddings(args, text_list, b)
    all_chunk_kw = list(merged_data.values())
    chunk_clus, division = run_hier_2D_SE_mini(args, b, block_n, all_chunk_kw, n = 500, e_a = True, e_s = False)

    chunk_name = list(merged_data.keys())
    return merged_data, chunk_name, chunk_clus, division, block_n

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataset", default="twitter12", type=str)
    args = parser.parse_args()
    if args.dataset == "twitter12":
        M_n = 21
    if args.dataset == "twitter18":
        M_n = 16

    result = {}
    
    for b in range(1, M_n+1):
        with open(f'evolution/{args.dataset}/M{b}_chunk/M{b}.json', 'r') as file:
                chunk = json.load(file)
        if b == 1:
            merged_data, chunk_name, chunk_clus, division, block_n = chunk_SE(args, chunk, None, b)
            last_block_node = {}
            merged_data_list = list(merged_data)
            for g, group in enumerate(division):
                group_content = [(merged_data_list[idx-1], merged_data[merged_data_list[idx-1]]) for idx in group]
                child_event_list = [{b: group_content}]
                result["Event "+str(g)] = child_event_list

                group_kw = []
                group_kw = [element for idx in group for element in merged_data[merged_data_list[idx-1]]]
                group_kw = list(set(group_kw))
                last_block_node["Event "+str(g)] = group_kw

            with open(f'evolution/{args.dataset}/M{b}_chunk/M{b}_node.json', 'w') as file:
                json.dump(last_block_node, file)
            with open(f'evolution/{args.dataset}/M{b}_chunk/M{b}_result.json', 'w') as file:
                json.dump(result, file)

        else:
            with open(f'evolution/{args.dataset}/M{b-1}_chunk/M{b-1}_node.json', 'r') as file:
                last_block_node = json.load(file)
            merged_data, chunk_name, chunk_clus, division, block_n = chunk_SE(args, chunk, last_block_node, b)

            last_block_node = {}
            merged_data_list = list(merged_data)

            filtered_division = [sublist for sublist in division if len(sublist) != 1 or sublist[0] <= block_n]
            for g, group in enumerate(filtered_division):
                group_content = [(merged_data_list[idx-1], merged_data[merged_data_list[idx-1]]) for idx in group if idx <= block_n]

                group_kw = []
                group_kw = [element for idx in group if idx <= block_n for element in merged_data[merged_data_list[idx-1]]]
                group_kw = list(set(group_kw))

                last_event = None
                for idx in group:
                    if idx > block_n:
                        last_event = merged_data_list[idx-1]

                if last_event is None:
                    child_event_list = [{b: group_content}]
                    last_block_node["Event "+str(len(result))] = group_kw
                    result["Event "+str(len(result))] = child_event_list
                else:
                    result[last_event].append({b: group_content})
                    last_block_node[last_event] = group_kw
            
            with open(f'evolution/{args.dataset}/M{b}_chunk/M{b}_node.json', 'w') as file:
                json.dump(last_block_node, file)
            with open(f'evolution/{args.dataset}/M{b}_chunk/M{b}_result.json', 'w') as file:
                json.dump(result, file)





