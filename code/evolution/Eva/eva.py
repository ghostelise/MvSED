import argparse
import json
from collections import Counter
import numpy as np
import scipy.io
from topmost.eva.topic_coherence import dynamic_coherence
from topic_diversity import dynamic_diversity

import file_utils

from gensim.topic_coherence import direct_confirmation_measure
from nan import custom_log_ratio_measure

direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure

def get_event(args):
    vocab_path = f'{args.evolution_path}{args.dataset}/vocab.txt'

    vocab = list()
    with open(vocab_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            vocab.append(line.strip())

    if args.dataset == "twitter12":
        M_n = 21
    if args.dataset == "twitter18":
        M_n = 16

    for b in range(1, M_n+1):
        with open(f'{args.evolution_path}{args.dataset}/M{b}_chunk/M{b}_node.json', 'r') as file:
            event = json.load(file)

        filtered_event_dict = {
            event: [word for word in keywords if word in vocab]
            for event, keywords in event.items()
        }


        with open(f'{args.evolution_path}{args.dataset}/texts.txt', "r", encoding="utf-8") as f:

            tokens = f.read().split()

        token_counts = Counter(tokens)

        vocab_counts = {word: token_counts.get(word, 0) for word in vocab}

        filtered_event_dict_trimmed = {}

        for event, keywords in filtered_event_dict.items():
            if len(keywords)<=5:
                continue
            if len(keywords) <= 20:
                filtered_event_dict_trimmed[event] = keywords
            else:
                sorted_keywords = sorted(
                    keywords, 
                    key=lambda x: vocab_counts.get(x, 0), 
                    reverse=True
                )

                filtered_event_dict_trimmed[event] = sorted_keywords[:20]


        output_file = f'{args.evolution_path}{args.dataset}/Event_over_times'

        with open(output_file, "a") as f:
            for idx, (event, keywords) in enumerate(filtered_event_dict_trimmed.items()):
                line = f"Time-{b-1}_K-{idx} " + " ".join(keywords) + "\n"
                f.write(line)

        print(f"SEE results are written to {output_file}")

    return


def evaTCandTD(args):
    train_texts = file_utils.read_text(f'{args.evolution_path}{args.dataset}/texts.txt')
    train_bow = scipy.sparse.load_npz(f'{args.evolution_path}{args.dataset}/bow.npz').toarray().astype('float32')
    train_times = np.loadtxt(f'{args.evolution_path}{args.dataset}/times.txt').astype('int32')
    vocab = file_utils.read_text(f'{args.evolution_path}{args.dataset}/vocab.txt')

    topic_path = f'{args.evolution_path}{args.dataset}/Event_over_times'
    time_topic_dict = file_utils.read_topic_words(topic_path)

    TC = dynamic_coherence(train_texts, train_times, vocab, list(time_topic_dict.values()))
    print(f"===>dynamic_TC: {TC:.5f}")

    TD = dynamic_diversity(list(time_topic_dict.values()), train_bow, train_times, vocab)
    print(f"===>dynamic_TD: {TD:.5f}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--evolution_path", default="evolution/", type=str)
    parser.add_argument("--dataset", default="twitter12", type=str)
    args = parser.parse_args()

    get_event(args)
    evaTCandTD(args)

