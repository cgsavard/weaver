#!/usr/bin/python

import uproot3
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, roc_auc_score
import sys, getopt

plt.style.use(hep.style.CMS)

def main(argv):
    
    # -----LOAD IN FILES-----
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('Input file is', inputfile)
    print('Output file is', outputfile)
    
    try:
        in_ = (uproot3.open(inputfile)["Jets"].arrays("*", namedecode="utf-8"))
        out_ = (uproot3.open(outputfile)["Jets"].arrays("*", namedecode="utf-8"))
    except:
        print('Input/output files are incorrect')
        return
   
    # -----EVALUATE MODEL-----

    # Create roc curve with AUC (area under curve) value
    fpr, tpr, dt = roc_curve(out_['is_EMJ'],out_['score_is_EMJ'])
    auc = roc_auc_score(out_['is_EMJ'],out_['score_is_EMJ'])
    
    plt.plot(fpr,tpr)
    plt.xlabel('FPR',fontsize=20)
    plt.ylabel('TPR',fontsize=20)
    plt.title('AUC = '+str(auc),fontsize=20)
    plt.savefig('ROC_curve.png')
    plt.close()
    
    # Plot TPR and FPR vs decision threshold (dt)
    plt.plot(dt[1:],fpr[1:],label='FPR')
    plt.plot(dt[1:],tpr[1:],label='TPR')
    plt.xlabel('decision thresh.',fontsize=20)
    plt.ylabel('metric',fontsize=20)
    plt.legend(loc='best')
    plt.savefig('TPR_FPR_vs_dt.png')
    plt.close()
    
    return
    
if __name__ == "__main__":
    main(sys.argv[1:])
