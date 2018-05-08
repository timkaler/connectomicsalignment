import sys
sys.path = ["/efs/home/wheatman/.local/lib/python2.7/site-packages"] + sys.path

import numpy
from matplotlib import pyplot

with open("for_hist.csv", "r") as f:
    pos = [float(item) for item in f.readline().split(",")[:-1]]
    neg = [float(item) for item in f.readline().split(",")[:-1]]

bins = numpy.linspace(min(min(pos), min(neg)), max(max(pos), max(neg)), 50)

pyplot.hist(pos, bins, alpha=0.5, label='pos')
pyplot.hist(neg, bins, alpha=0.5, label='neg')
pyplot.yscale('log')
pyplot.legend(loc='upper right')
pyplot.savefig("hist.png")


total = len(pos)+len(neg)
for cutoff in bins:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for item in pos:
        if item > cutoff:
        	tp += 1
        else:
        	fn += 1
    for item in neg:
    	if item > cutoff:
    		fp += 1
    	else:
    		tn += 1
    print "for cutoff", cutoff, "accuracy is", float(tp + tn) /total, "tp =", tp, "fp =", fp, "tn =", tn, "fn =", fn
            

