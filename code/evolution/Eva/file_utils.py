from collections import defaultdict

def read_text(path):
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def read_topic_words(path):
    topic_str_list = read_text(path)
    time_topic_dict = convert_topicStr_to_dict(topic_str_list)

    return time_topic_dict


def convert_topicStr_to_dict(topic_str_list):
    time_topic_dict = defaultdict(list)
    # topic_str:  Time-0_K-0 w1 w2 w3 ...
    for topic_str in topic_str_list:
        item_info = topic_str.split()[0]
        time, k = (int(item.split('-')[1]) for item in item_info.split('_'))

        time_topic_dict[time].append(' '.join(topic_str.split()[1:]))

    return time_topic_dict