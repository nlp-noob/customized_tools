import numpy as np
import Levenshtein

def levenshtein_len_dist_rate(str1, str2):
    dist = abs(len(str1) - len(str2))
    return 1 - dist / (len(str1) + len(str2))

def levenshtein_dist_rate(str1, str2, up_dist_thresh=0.):
    #return difflib.SequenceMatcher(None, str1, str2).ratio()
    if up_dist_thresh > 0. and levenshtein_len_dist_rate(str1, str2) < up_dist_thresh:
        return 0.
    return Levenshtein.ratio(str1.lower(), str2.lower())

def dist_group(corpus, text_key, dist_thresh=0.9):
    leader = None
    group_dict = {}
    dist_dict = {}
    for idx, msg in enumerate(corpus):
        msg_text = text_key(msg)
        if leader is None:
            # new leader
            leader = msg_text
            group_dict[leader] = [msg]
            dist_dict[leader] = [1.0]
            continue

        dist = levenshtein_dist_rate(leader, msg_text)
        if dist < dist_thresh:
            # check new group
            if len(msg_text.split(' ')) < 1: # do not recall
                cur_leaders = list(group_dict.keys())
                leaders_dist = [levenshtein_dist_rate(msg_text, i, dist_thresh) for i in cur_leaders]
                max_i = np.argmax(leaders_dist)
                max_leader_dist = leaders_dist[max_i]
            max_leader_dist = dist
            if max_leader_dist < dist_thresh:
                # new group
                leader = msg_text
                group_dict[leader] = [msg]
                dist_dict[leader] = [1.0]
                continue
            else:
                # old group
                leader = cur_leaders[max_i]
                dist = leaders_dist[max_i]

        # append to group
        group_dict[leader].append(msg)
        dist_dict[leader].append(dist)

    return group_dict, dist_dict

def deduplicate_texts(corpus, dist_thresh=0.85, debug=False, text_key=lambda x: x, return_group_dict=False):
    corpus.sort(key=lambda x: text_key(x).lower())

    filtered_corpus = []
    group_dict, dist_dict = dist_group(corpus, text_key, dist_thresh)
    group_keys = group_dict.keys()
    for group_key in group_keys:
        msgs = group_dict[group_key]
        dists = dist_dict[group_key]

        if debug:
            print('==' * 20)
            print(group_key)
            for msg, dist in zip(msgs, dists):
                print('--' * 20)
                print(text_key(msg), dist)

        # select one
        msg_size = [len(text_key(i)) for i in msgs]    
        max_i = np.argmax(msg_size)
        filtered_corpus.append([msgs[max_i], len(msgs)])
    if return_group_dict:
        return filtered_corpus, group_dict
    else:
        return filtered_corpus

def cluster_texts(corpus, dist_thresh=0.85, debug=False, text_key=lambda x: x):
    corpus.sort(key=lambda x: text_key(x).lower())

    groups = []
    group_dict, dist_dict = dist_group(corpus, text_key, dist_thresh)
    group_keys = group_dict.keys()
    for group_key in group_keys:
        msgs = group_dict[group_key]
        dists = dist_dict[group_key]

        if debug:
            print('==' * 20)
            print(group_key)
            for msg, dist in zip(msgs, dists):
                print('--' * 20)
                print(text_key(msg), dist)

        # sort group
        msgs.sort(key=lambda x: text_key(x).lower())
        groups.append(msgs)

    return groups

def test():
    import sys
    import json
    order_file = sys.argv[1]
    with open(order_file, 'r') as fin:
        orders = json.load(fin)

    texts = []
    for _, order_list in orders.items():
        for order in order_list:
            msc = order['msc']
            for _, split_texts, _ in msc:
                texts.extend([i for i in split_texts if len(i.strip()) > 1])

    print(len(texts))
    dedup_texts = deduplicate_texts(texts, dist_thresh=0.85, debug=True)
    print(len(dedup_texts))
    for msg, count in dedup_texts:
        print(count, msg)

if __name__ == '__main__':
    test()
