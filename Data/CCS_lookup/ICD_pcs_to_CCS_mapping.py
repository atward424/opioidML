# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:21:09 2019

@author: trishaj
"""

import numpy as np
import pandas as pd
#%% Create the dict for mapping ICD 10 codes to the CCS classes
icd10_dict = {}

with open('ICD10_pcs_to_CCS_lookup.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        #icd10_dict[elements[0].strip('\'"')] = elements[1].strip('\'"').strip()
        k = elements[0].strip('\'')
        k = k.strip()
        v = elements[1].strip('\'')
        icd10_dict[k] = int(v)

f.close()
num_classes_icd10 = len(set(icd10_dict.values()))



icd_table = pd.DataFrame(list(icd10_dict.items()))
icd_table = icd_table.rename(columns={0: "ICD_pcs", 1: "CCS"})
icd_table.to_csv('S:/users/trishaj/Documents/Opioids/ICD10_pcs_to_CCS.csv', index=False)




#%% Create the dict for mapping ICD 9 codes to the CCS classes
icd9_dict = {}

with open('ICD9_pcs_to_CCS_lookup.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        icd9_dict[elements[0].strip('\'"')] = elements[1].strip('\'"').strip()
f.close()
num_classes_icd9 = len(set(icd9_dict.values()))

#elements[0].strip('\'"') 




icd_table = pd.DataFrame(list(icd9_dict.items()))
icd_table = icd_table.rename(columns={0: "ICD_pcs", 1: "CCS"})
icd_table.to_csv('S:/users/trishaj/Documents/Opioids/ICD9_pcs_to_CCS.csv', index=False)

#%%
icd10_values = set(icd10_dict.values())
icd9_values = set(icd9_dict.values())
num_icd10_values = len(icd10_values)
num_icd9_values = len(icd9_values)


#%% None of the keys are shared, so we can actually just use a single dictionary
icd10_keys = set(icd10_dict.keys())
icd9_keys = set(icd9_dict.keys())
icd9_10_keys_intersect = icd10_keys.intersection(icd9_keys)


#%% Use a single dictionary

icd_dict = {}

with open('ICD9_pcs_to_CCS_lookup.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        k = elements[0].strip('\'')
        k = k.strip()
        v = elements[1].strip('\'')
        icd_dict[k] = int(v)
        
f.close()

with open('ICD10_pcs_to_CCS_lookup.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        k = elements[0].strip('\'')
        k = k.strip()
        v = elements[1].strip('\'')
        icd_dict[k] = int(v)

f.close()

num_classes = len(set(icd_dict.values()))

#%%

icd_table = pd.DataFrame(list(icd_dict.items()))
icd_table = icd_table.rename(columns={0: "Proc", 1: "CCS_proc"})
icd_table.to_csv('S:/users/trishaj/Documents/Opioids/ICD_pcs_to_CCS.csv', index=False)



