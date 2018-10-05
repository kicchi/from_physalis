# coding: utf-8


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from sklearn.utils import shuffle
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole

from rdkit.Chem import rdDepictor
from rdkit.six import StringIO

import sys 
 
argvs = sys.argv  
argc = len(argvs)

if (argc != 3):  
    print 'Usage: # python %s inputFilename makeFilename' % argvs[0]
    quit()


sdf = PandasTools.LoadSDF(argvs[1])
active_list = sdf.activity
dict={}


graphsize = 0
f = open(argvs[2], 'w')
for mol in sdf['ROMol']:
    com = AllChem.RemoveHs(mol)
    #print "t #",graphsize
    f.write("t # "+str(graphsize)+" ")#+active_list[graphsize]+"\n")
    if active_list[graphsize] == 'Active':
        f.write('1\n')
    else:
        f.write('-1\n')
    graphsize +=1
    vn = 0
    virtexs = [a.GetSymbol() for a in com.GetAtoms()]
    gl = AllChem.GetConnectivityInvariants(com)
    #print virtexs
    for i in range(len(virtexs)):
        labeli = -1
        if gl[i] not in dict:
            labeli = len(dict)+1
            dict[gl[i]] = labeli
        else:
            labeli = dict[gl[i]]
        #print "v",i,labeli
        f.write("v "+str(i)+" "+str(labeli)+"\n")
    bonds = [(a.GetBeginAtom().GetIdx(),a.GetEndAtom().GetIdx()) for a in com.GetBonds()]
    #print bonds
    for fr,to in bonds:
        #print "e",fr,to,"1"
        f.write("e "+str(fr)+" "+str(to)+" 1\n")
    #print
    f.write("\n")
print "Graph num : ",graphsize
print "Label num : ",len(dict)
print dict

f.close()

