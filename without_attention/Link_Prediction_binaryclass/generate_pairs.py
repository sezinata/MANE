import node2vec
import time
import random
import numpy as np


def save_walks(walks, out_file, elapsed):
    """
    Save node2vec walks.
    """
    with open(out_file, "w") as f_out:
        for walk in walks:
            f_out.write(" ".join(map(str, walk)) + "\n")
        print("Elapsed time during walks: ", elapsed, " seconds.\n")

    return


def save_pairs(pairs, out_file, elapsed):
    """
    Save pairs of word2vec.
    """
    with open(out_file, "w") as f_out:
        for pair in pairs:
            f_out.write(" ".join(map(str, pair)) + "\n")
    return


def save_train_neigh(pair_node, out_file):
    with open(out_file, 'w') as f:
        f.write(" ".join(map(str, pair_node)))

    return


def construct_word2vec_pairs(G, view_id, common_nodes, pvalue, qvalue, window_size, n_walk, walk_length, output_pairs,
                             node2idx):
    """
    Generate and save Word2Vec pairs.
    """
    path = output_pairs
    list_neigh = []
    G_ = node2vec.Graph(G, False, pvalue, qvalue)
    G_.preprocess_transition_probs()
    start_time = time.time()
    walks = G_.simulate_walks(n_walk,
                              walk_length)
    end = time.time()
    walk_file = path + "/Walks_" + str(view_id) + ".txt"
    elapsed = end - start_time
    save_walks(walks, walk_file, elapsed)
    start_time = time.time()
    for walk in walks:
        for pos, word in enumerate(walk):
            reduced_window = random.randint(1, window_size)
            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - window_size + reduced_window)
            for pos2, word2 in enumerate(walk[start:(pos + window_size + 1 - reduced_window)], start):
                # don't train on the `word` itself
                if word != word2:
                    list_neigh.append((node2idx[word], node2idx[word2]))
    pair_file = path + "/Pairs_" + str(view_id) + ".txt"
    list_neigh.sort(key=lambda x: x[0])  # sorted based on keys
    list_neigh = np.array(list_neigh)
    save_pairs(list_neigh, pair_file, elapsed)

    nodes_idx, neigh_idx = zip(*[(tupl[0], tupl[1]) for tupl in list_neigh])  # gives tuple
    nodesidx_file = path + "/nodesidxPairs_" + str(view_id) + ".txt"
    save_train_neigh(np.array(list(nodes_idx)), nodesidx_file)

    neigh_idx_file = path + "/neighidxPairs_" + str(view_id) + ".txt"
    save_train_neigh(np.array(list(neigh_idx)), neigh_idx_file)
    end = time.time()

    elapsed = end - start_time
    print("Elapsed time during pairs for network " + str(view_id) + ": ", elapsed, " seconds.\n")

    return np.array(list(nodes_idx)), np.array(list(neigh_idx))
