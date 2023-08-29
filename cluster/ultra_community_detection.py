import sys
import os
import random
import argparse
import json
import numpy as np
import time
import re
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

from deduplicate_text import cluster_texts


class Cluster():
    def __init__(self, model_name_or_path, batch_size=16):
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = batch_size

    def cluster(self, corpus, threshold=0.75, min_community_size=6, init_max_size=5000):
        corpus_embeddings =  self.model.encode(corpus, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True)
        return self.community_detection(corpus_embeddings, threshold=threshold, min_community_size=min_community_size, batch_size=init_max_size)

    # modify code from sentence_transformers
    def community_detection(self, embeddings, threshold=0.75, min_community_size=10, batch_size=1024):
        """
        Function for Fast Community Detection
        Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
        Returns only communities that are larger than min_community_size. The communities are returned
        in decreasing order. The first element in each list is the central point in the community.
        """
    
        threshold = torch.tensor(threshold, device=embeddings.device)
    
        extracted_communities = []
    
        # Maximum size for community
        min_community_size = min(min_community_size, len(embeddings))
        sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))
    
        for start_idx in range(0, len(embeddings), batch_size):
            # Compute cosine similarity scores
            cos_scores = util.cos_sim(embeddings[start_idx:start_idx + batch_size], embeddings)
    
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)
    
            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    new_cluster = []
    
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)
    
                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold:
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)
    
                    for idx, val in zip(top_idx_large.tolist(), top_val_large):
                        if val < threshold:
                            break
    
                        new_cluster.append(idx)
    
                    extracted_communities.append(new_cluster)
    
            del cos_scores
    
        # Largest cluster first
        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)
    
        # Step 2) Remove overlapping communities
        unique_communities = []
        extracted_ids = set()
    
        # origin version
        # for cluster_id, community in enumerate(extracted_communities):
        #     community = sorted(community)
        #     non_overlapped_community = []
        #     for idx in community:
        #         if idx not in extracted_ids:
        #             non_overlapped_community.append(idx)
    
        #     if len(non_overlapped_community) >= min_community_size:
        #         unique_communities.append(non_overlapped_community)
        #         extracted_ids.update(non_overlapped_community)
    
        # unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)
    
        # customized version
        print("extracting result...")
        for cluster_id, community in enumerate(tqdm(extracted_communities)):
            community = sorted(community)
            unique_overlap_rate = []
            for unique_idx, unique_community in enumerate(unique_communities):
                unique_community_set = set(unique_community)
                non_overlapped_community = []
                for idx in community:
                    if idx not in unique_community_set:
                        non_overlapped_community.append(idx)
                unique_overlap_rate.append([(len(community) - len(non_overlapped_community)) / len(community), unique_idx, non_overlapped_community])
            sorted_unique_overlap_rate = sorted(unique_overlap_rate, key=lambda x: x[0], reverse=True) 
            if len(sorted_unique_overlap_rate) > 0 and sorted_unique_overlap_rate[0][0] > 0.5:
                unique_idx = sorted_unique_overlap_rate[0][1]
                non_overlapped_community = sorted_unique_overlap_rate[0][2]
                unique_communities[unique_idx].extend(non_overlapped_community)
                extracted_ids.update(non_overlapped_community)
            else:
                non_overlapped_community = list(set(community) - extracted_ids)
                unique_communities.append(non_overlapped_community)
                extracted_ids.update(non_overlapped_community)
        return unique_communities



def cluster_text_list(engine, text_list, deduplicate_thresh=0.85, sim_thresh=0.6):
    hamming_clusters = cluster_texts(text_list, deduplicate_thresh)
    dedup_txts, dedup_idxs = [], []
    dedup_group_idxs = list(range(len(hamming_clusters)))

    resp_to_hamming_cluster_idx = {}
    for hamming_cluster_idx, hamming_cluster in enumerate(hamming_clusters):
        for resp in hamming_cluster:
            resp_to_hamming_cluster_idx[resp] = hamming_cluster_idx

    for cluster in hamming_clusters:
        # choose median length txt
        txt_size = [len(i) for i in cluster]
        si = np.argsort(txt_size)
        median_idx = si[len(cluster)//2]
        txt = cluster[median_idx]
        dedup_txts.append(txt)
        dedup_idxs.append(median_idx)
        
    # shuffle dedup lists
    tmp = list(zip(dedup_txts, dedup_idxs, dedup_group_idxs))
    random.shuffle(tmp)
    dedup_txts, dedup_idxs, dedup_group_idxs = zip(*tmp)
    print("origin resp size: {}".format(len(text_list)))
    print("size after hamming: {}".format(len(dedup_txts)))
    sim_clusters = engine.cluster(dedup_txts, threshold=sim_thresh)

    # from rich.console import Console
    # console = Console()
    # for idxs in sim_clusters:
    #     cluster_text_list = []
    #     for idx in idxs:
    #         cluster_text_list.append(dedup_txts[idx])
    #     console.print(cluster_text_list)
    #     input()

    if False:
        print('**' * 19 + 'sim cluster' + '**' * 20)
        sim_groups = []
        for cluster in sim_clusters:
            cur_group = []
            for i in cluster:
                gidx = dedup_group_idxs[i]
                midx = dedup_idxs[i]
                msg = hamming_clusters[gidx][midx]
                cur_group.append(msg)
            sim_groups.append(cur_group)
        print_cluster(sim_groups)

    # merge deduplicate cluster and sim clusters
    final_groups = []
    gidx_set = set()
    for cluster in sim_clusters:
        cur_group = []
        for i in cluster:
            gidx = dedup_group_idxs[i]
            gidx_set.add(gidx)
            cur_group.extend(hamming_clusters[gidx])
        final_groups.append(cur_group)
    for gidx, hamming_group in enumerate(hamming_clusters):
        if gidx in gidx_set:
            continue
        final_groups.append(hamming_group)

    return final_groups


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="", type=str, default="./data/tag_data/splits/flatten/")
    parser.add_argument('-m', '--model', help='the sentence-transformer model path', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--sim_thresh', help='cluster siminarity threshold', type=float, default=0.75)
    parser.add_argument('--hamming_thresh', help='hamming siminarity threshold', type=float, default=0.85)
    parser.add_argument('--gpus', default='1', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


    text_list = ["hello", "hello", "hello", "You are welcome"]

    engine = Cluster(args.model)
    clusters = cluster_text_list(engine, text_list, args.hamming_thresh, args.sim_thresh)
    print("clusters cnt: {}".format(len(clusters)))

    resp_to_cluster_idx = {}
    for cluster_idx, cluster in enumerate(clusters):
        for resp in cluster:
            resp_to_cluster_idx[resp] = cluster_idx
    print(resp_to_cluster_idx)


if __name__ == "__main__":
    test()
