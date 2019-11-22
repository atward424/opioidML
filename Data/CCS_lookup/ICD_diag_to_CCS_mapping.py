# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:36:52 2019

@author: trishaj
"""
import numpy as np
import pandas as pd
#%% Create the dict for mapping ICD 10 codes to the CCS classes
icd10_dict = {}

with open('ICD10_diag_to_CCS_lookup.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        icd10_dict[elements[0].strip('\'"')] = elements[1].strip('\'"').strip()

f.close()
num_classes_icd10 = len(set(icd10_dict.values()))

#%% Create the dict for mapping ICD 9 codes to the CCS classes
icd9_dict = {}

with open('ICD9_diag_to_CCS_lookup.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        icd9_dict[elements[0].strip('\'"')] = elements[1].strip('\'"').strip()
f.close()
num_classes_icd9 = len(set(icd9_dict.values()))

#elements[0].strip('\'"') 


#%% Some keys do intersect.....

icd10_keys = set(icd10_dict.keys())
icd9_keys = set(icd9_dict.keys())
icd9_10_keys_intersect = icd10_keys.intersection(icd9_keys)

#Since there is overlap in the keys between icd 9 and 10,
#have to use two separate dictionaries. ICD9 also uses codes that 
#start with  E and V


#%% Turn into data frames and save 

icd9_table = pd.DataFrame(list(icd9_dict.items()))

icd9_table = icd9_table.rename(columns={0: "ICD9", 1: "CCS"})


icd10_table = pd.DataFrame(list(icd10_dict.items()))

icd10_table = icd10_table.rename(columns={0: "ICD10", 1: "CCS"})

icd9_table.to_csv('S:/users/trishaj/Documents/Opioids/ICD9_diag_to_CCS.csv', index=False)
icd10_table.to_csv('S:/users/trishaj/Documents/Opioids/ICD10_diag_to_CCS.csv', index=False)
