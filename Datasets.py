# This file exists as a workplace for me to display stats about
# the different datasets.
#
# If you are looking through this project's code to try to
# understand how it works, you're probably not going to find
# anything useful here.

from Lib import *
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_seq_lengths(pos, neg, name):
    def count_lengths(arr):
        counts = {}
        for i in arr:
            if len(i) not in counts:
                counts[len(i)] = 0
            counts[len(i)] += 1
        return counts

    counts_pos = count_lengths(pos)
    counts_neg = count_lengths(neg)
    max_count = max(max(counts_pos.keys()), max(counts_neg.keys()))
    for i in range(0, max_count + 1):
        if i not in counts_neg:
            counts_neg[i] = 0
        if i not in counts_pos:
            counts_pos[i] = 0
    counts_pos = {key:val for (key, val) in sorted(counts_pos.items())}
    counts_neg = {key:val for (key, val) in sorted(counts_neg.items())}

    bar_width = 0.35
    index = np.arange(0, max_count + 1)
    plt.bar(index, counts_pos.values(), bar_width, color='r', label='Positives', align='edge')
    plt.bar(index + bar_width, counts_neg.values(), bar_width, color='b', label='Negatives', align='edge')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.title('Lengths of Sequences for {}'.format(name))
    plt.xticks(index + bar_width / 2, range(0, max_count + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

datasets = [folder for folder in os.listdir('.') if folder.startswith("dataset")]
for ds in datasets:
    positives = read_file_in_dataset(ds, "positive")
    negatives = read_file_in_dataset(ds, "negative")
    print("+{}: {}".format(ds, len(positives)))
    print("-{}: {}".format(ds, len(negatives)))
