import numpy as np
from sympy.utilities.iterables import multiset_permutations


def map_states(n_states, sequence1, sequence2):
    # Maps the best permutation of sequence 2 to sequence 1
    all_states = range(n_states)

    min_T = np.min([len(sequence1), len(sequence2)])
    sequence1 = sequence1[:min_T]
    sequence2 = sequence2[:min_T]

    best_score = -np.inf
    best_mapping = sequence2
    for permutation in multiset_permutations(all_states):
        permutation = np.array(permutation)
        remapped_sequence2 = permutation[sequence2]
        score = np.sum(remapped_sequence2 == sequence1)
        if score > best_score:
            best_score = score
            best_mapping = remapped_sequence2
    return best_mapping
