# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:22:34 2019

@author: trishaj
"""
import pandas as pd

def convert_diag_to_CCS(X):

    #diag_to_ccs = pd.read_csv("ICD_diag_to_CCS.csv")
    ccs_diag_names = pd.read_csv("../CCS_lookup/CCS_diag_lookup.csv")
    
    diag_to_ccs = pd.read_csv("../CCS_lookup/ICD_diag_to_CCS.csv", dtype={'Diag': str})


        
    patient_diag_and_ccs = pd.merge(X, diag_to_ccs, on=["Diag", "Icd_Flag"], how="left")
    #Get rid of the diagnosis codes that do not have a corresponding CCS code. 
    #these dx codes are invalid (not specific enough), and therefore do not have a CCS
    patient_diag_and_ccs.dropna(inplace= True)
    patient_diag_and_ccs.drop(["Diag", "Icd_Flag"], axis = 1, inplace = True)
    patient_diag_and_ccs.set_index(['Patid', 'surgery_date'], inplace=True)
    
    patient_diag_and_ccs['CCS_diag'] = patient_diag_and_ccs['CCS_diag'].astype(int)
    patient_diag_and_ccs['CCS_diag'] = patient_diag_and_ccs['CCS_diag'].astype(str)
    
    
    patient_diag_ccs_1hot = pd.get_dummies(patient_diag_and_ccs["CCS_diag"])#, prefix="CCS_diag")
    patient_diag_ccs_1hot = patient_diag_ccs_1hot.groupby(['Patid', 'surgery_date']).max()
    
    #Rename the columns with the label names as well for ease of analysis
    ccs_diag_names['CCS_diag_and_name'] = ccs_diag_names['CCS_diag'] + "_" + ccs_diag_names['Label']+ "_ccs_diag" 
    ccs_diag_mapping = pd.Series(ccs_diag_names.CCS_diag_and_name.values,index=ccs_diag_names.CCS_diag).to_dict()
    patient_diag_ccs_1hot.rename(columns = ccs_diag_mapping, inplace = True)
    
    
    return patient_diag_ccs_1hot