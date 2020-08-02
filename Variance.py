
# This file is for processing various statistics of the results
# of all th diffrent tests we performed.
#
# If you are looking through this project's code to try to
# understand how it works, you're probably not going to find
# anything useful here.

from math import sqrt
from statsmodels.stats.weightstats import ztest

def mean(list):
    total = 0.0
    for val in list:
        total += val

    return total / len(list)

def std_dev(list):
    total = 0.0
    mu = mean(list)
    for val in list:
        diff = val - mu
        total += (diff ** 2)
    
    return sqrt(total / len(list))

stats = {}
with open("results.tab", "r") as file:
    next(file)
    for line in file:
        [ds, _, mcc, acc, recall, spec, prec, f1] = line.split("\t")
        if ds not in stats:
            stats[ds] = {
                "mcc" : [],
                "acc" : [],
                "recall" : [],
                "spec" : [],
                "prec" : [],
                "f1" : []
            }
        
        stats[ds]["mcc"].append(float(mcc))
        stats[ds]["acc"].append(float(acc))
        stats[ds]["recall"].append(float(recall))
        stats[ds]["spec"].append(float(spec))
        stats[ds]["prec"].append(float(prec))
        stats[ds]["f1"].append(float(f1))

for ds, measures in stats.items():
    print("{}:".format(ds))
    for metric, vals in measures.items():
        print("\t{} Mean: {}".format(metric, mean(vals)))
        print("\t{} Std Dev: {}\n".format(metric, std_dev(vals)))

for ds in stats:
    if ds.startswith("acpdl"):
        [_, ds] = ds.split("_")
        acpdl = stats["acpdl_{}".format(ds)]["mcc"]
        deepacppred = stats["deepacppred_{}".format(ds)]["mcc"]
        tstat, pvalue = ztest(acpdl, deepacppred, alternative='larger')
        print("{} - {} - {}".format(ds, tstat, pvalue))
