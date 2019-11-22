# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:38:07 2019

@author: trishaj
"""

import numpy as np
import random
import math
import time
import pdb
import sys
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


#%%
ccs = {}

with open('ICD_to_CCS_lookup.csv') as f:
    next(f)
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        k = elements[0].strip('\'')
        k = k.strip()
        v = elements[1].strip('\'')
        icd_flag = int(elements[2].strip('\''))
        ccs[(k,icd_flag)] = int(v)

num_classes = len(set(ccs.values()))
#%%

df = pd.DataFrame(list(ccs.items()), columns=['ICD_w_flag', 'CCS_diag']) 

df[['Diag', 'Icd_Flag']] = pd.DataFrame(df['ICD_w_flag'].tolist(), index=df.index)  

df= df.drop('ICD_w_flag', axis=1)





df.to_csv('ICD_diag_to_CCS.csv', index=False)



#df2 = pd.read_csv("ICD_to_CCS.csv", dtype=str)
