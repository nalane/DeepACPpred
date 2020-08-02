
# This file contains functions used by all the other files in this project.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import time
import re
import sys
import random
import math
import tensorflow as tf

# This section contains physical characteristics of the amino acids. Currently
# unused, though they may prove to be useful in the future.
NONE, POLAR, NONPOLAR, AROMATIC = 0, 1, 2, 3
NONALIPHATIC, ALIPHATIC = 0, 1
NEUTRAL, POSITIVE, NEGATIVE = 0, 1, 2
HYDRONEUTRAL, HYDROPHILIC, HYDROPHOBIC = 0, 1, 2

VOCAB = {
    #     CODE CHARGED ALIPHATIC AROMATIC POLAR H.PHOBIC POS.CHARGE NEG.CHARGE TINY SMALL LARGE MASS 
    ' ': [0,  -1,     -1,       -1,      -1,   -1,      -1,        -1,        -1,  -1,   -1,      0.0000],
    'A': [1,   0,      0,        0,       0,    0,       0,         0,         1,   0,    0,     71.0779],
    'R': [2,   1,      0,        0,       1,    0,       1,         0,         0,   0,    1,    156.1857],
    'N': [3,   0,      0,        0,       1,    0,       0,         0,         0,   1,    0,    114.1026],
    'D': [4,   1,      0,        0,       1,    0,       0,         1,         1,   0,    0,    115.0874],
    'C': [5,   0,      0,        0,       0,    1,       0,         0,         1,   0,    0,    103.1429],
    'E': [6,   1,      0,        0,       1,    0,       0,         1,         0,   1,    0,    129.1140],
    'Q': [7,   0,      0,        0,       1,    0,       0,         0,         0,   1,    0,    128.1292],
    'G': [8,   0,      0,        0,       0,    0,       0,         0,         1,   0,    0,     57.0513],
    'H': [9,   1,      0,        1,       0,    0,       1,         0,         0,   1,    0,    137.1393],
    'I': [10,  0,      1,        0,       0,    1,       0,         0,         0,   1,    0,    113.1576],
    'L': [11,  0,      1,        0,       0,    1,       0,         0,         0,   1,    0,    113.1576],
    'K': [12,  1,      0,        0,       1,    0,       1,         0,         0,   1,    0,    128.1723],
    'M': [13,  0,      0,        0,       0,    1,       0,         0,         0,   1,    0,    131.1961],
    'F': [14,  0,      0,        1,       0,    1,       0,         0,         0,   0,    1,    147.1739],
    'P': [15,  0,      0,        0,       0,    0,       0,         0,         0,   1,    0,     97.1152],
    'S': [16,  0,      0,        0,       0,    0,       0,         0,         1,   0,    0,     87.0773],
    'T': [17,  0,      0,        0,       0,    0,       0,         0,         1,   0,    0,    101.1039],
    'W': [18,  0,      0,        1,       0,    1,       0,         0,         0,   0,    1,    186.2099],
    'Y': [19,  0,      0,        1,       0,    0,       0,         0,         0,   0,    1,    163.1733],
    'V': [20,  0,      1,        0,       0,    1,       0,         0,         0,   1,    0,     99.1311],
    ';': [21,  -1,     -1,       -1,      -1,   -1,      -1,        -1,        -1,  -1,   -1,      0.0000]
}

# These functions contain code for reading the different kinds of input files.
def read_cppd_file(filename):
    peptide_set = set()
    with open(filename, "r") as file:
        pattern = re.compile(r'^(\d{4})\s+(\w+)$')
        for line in file:
            match = pattern.match(line)
            if match:
                peptide = match.group(2) + ' '
                peptide_set.add(peptide)

    return list(peptide_set)

def read_fasta_file(filename):
    peptide_set = set()
    with open(filename, "r") as file:
        peptide = ""
        for line in file:
            if line[0] == '>':
                if peptide != "":
                    peptide_set.add(peptide + ';')
                    peptide = ""
            else:
                peptide += line.strip()
    return list(peptide_set)

def read_file_in_dataset(dataset, file):
    # Read in files
    input_file = [f for f in os.listdir(dataset) if f.startswith(file)][0]
    input_file = os.path.join(dataset, input_file)
    if input_file.endswith("cppd"):
        return read_cppd_file(input_file)
    elif input_file.endswith("fasta"):
        return read_fasta_file(input_file)
    else:
        print("Unknown file format for {}".format(input_file))
        return []
