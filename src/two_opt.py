import numpy as np

from src.utils import compute_length


def swap2opt(tsp_sequence, i, j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j + 1] = np.flip(tsp_sequence[i:j + 1], axis=0)
    return new_tsp_sequence


def gain(i, j, tsp_sequence, matrix_dist):
    old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
        tsp_sequence[j], tsp_sequence[j + 1]])
    changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[
        tsp_sequence[i], tsp_sequence[j + 1]])
    return - old_link_len + changed_links_len


def step2opt(solution, matrix_dist, distance):
    seq_length = len(solution) - 1
    tsp_sequence = np.array(solution)
    uncrosses = 0
    for i in range(1, seq_length - 1):
        for j in range(i + 1, seq_length):
            new_tsp_sequence = swap2opt(tsp_sequence, i, j)
            new_distance = distance + gain(i, j, tsp_sequence, matrix_dist)
            if new_distance < distance:
                uncrosses += 1
                tsp_sequence = np.copy(new_tsp_sequence)
                distance = new_distance
    return tsp_sequence, distance, uncrosses


def loop2opt(solution, instance, max_num_of_uncrosses=10000):
    matrix_dist = instance.dist_matrix
    new_len = compute_length(solution, matrix_dist)
    new_tsp_sequence = np.copy(np.array(solution))
    uncross = 0
    while uncross < max_num_of_uncrosses:
        new_tsp_sequence, new_reward, uncr_ = step2opt(new_tsp_sequence, matrix_dist, new_len)
        uncross += uncr_
        if new_reward < new_len:
            new_len = new_reward
        else:
            return new_tsp_sequence.tolist()

    # return new_tsp_sequence.tolist(), new_len, uncross
    return new_tsp_sequence.tolist()
