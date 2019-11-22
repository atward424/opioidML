# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 08:46:53 2019

@author: trishaj
"""


import numpy as np
import pandas as pd
#%% Create the dict for mapping ICD 10 codes to the CCS classes
cpt_dict = {}

with open('CPT_to_CCS.csv') as f:
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        code_range = elements[0].strip('\'"')
        range_elements = code_range.split("-")
        range_start = range_elements[0]
        range_end = range_elements[1]
        
        #We only want to consider CPT codes. 
        #The HCPCS codes all start with a letter, so we can just ignore them
        if range_start[0].isalpha() == False: #CPT codes start with number
            suffix = range_start[-1]
            code_length = 5 #each code is of length 5
            if suffix.isalpha() == False: #check if last element is a number
                #cast to int
                range_start = int(range_start)
                range_end = int(range_end)
                #map all the numbers in the range to the corresponding CCS class
                for cpt_code in range(range_start, range_end+1):
                    cpt_dict[str(cpt_code).zfill(code_length)] = elements[1].strip('\'"').strip() 
            else: #last element is a letter
                range_start = int(range_start[:-1])
                range_end = int(range_end[:-1])
                for cpt_code in range(range_start, range_end+1):
                    cpt_dict[(str(cpt_code) + suffix).zfill(code_length)] = elements[1].strip('\'"').strip()
                

f.close()
num_classes = len(set(cpt_dict.values()))

#%% Convert and save

cpt_table = pd.DataFrame(list(cpt_dict.items()))

cpt_table = cpt_table.rename(columns={0: "CPT", 1: "CCS"})

cpt_table.to_csv('S:/users/trishaj/Documents/Opioids/CPT_to_CCS.csv', index=False)

