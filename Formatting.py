# This file is for formatting the results files into TSV files.
#
# If you are looking through this project to try to understand
# how it works, you're probably in the wrong place.

import os

def add_old():
    sets = ["tyagi", "lee", "hc"]

    for s in sets:
        for i in range(10):
            with open("results_seed_{}/t{}.txt".format(s, i)) as file:
                mcc = file.readline().strip().split(": ")[1]
                acc = file.readline().strip().split(": ")[1]
                recall = file.readline().strip().split(": ")[1]
                spec = file.readline().strip().split(": ")[1]
                prec = file.readline().strip().split(": ")[1]
                f1 = file.readline().strip().split(": ")[1]

                with open("results.tab", "a") as res_file:
                    res_file.write("dataset_{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(s, i, mcc, acc, recall, spec, prec, f1))

def sort_res():
    lines = []
    with open("results.tab", "r") as file:
        next(file)
        for line in file:
            lines.append(line.strip().split("\t"))

    lines.sort(key = lambda x: int(x[1]))
    lines.sort(key = lambda x: x[0])
    with open("results.tab", "w") as file:
        file.write("Dataset\tSeed\tMCC\tAccuracy\tRecall\tSpecificity\tPrecision\tF1\n")
        for line in lines:
            file.write("\t".join(line) + "\n")

sort_res()
