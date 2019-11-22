# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:44:50 2019

@author: trishaj
"""
import pandas as pd

#function that converts any codes (either icd diagnosis or procedure or cpt procedure) to the CCS encoding
#class type is either ICD or CPT
# X is the unencoded input data of the history of either ICD/CPT procedure codes

def convert_procedure_to_CCS(class_type, X):

    if class_type == "ICD":
        ccs_lookup = pd.read_csv("../CCS_lookup/ICD_pcs_to_CCS.csv", dtype={'Proc': str}, skipinitialspace=True)
    elif  class_type == "CPT": #CPT
        ccs_lookup = pd.read_csv("../CCS_lookup/CPT_to_CCS.csv",  dtype={'Proc': str}, skipinitialspace=True)
    else:
        print("Code type doesnt exist")

    ccs_proc_names = pd.read_csv("../CCS_lookup/CCS_proc_lookup.csv")

        
    patient_code_and_ccs = pd.merge(X, ccs_lookup, on=["Proc"], how="left")
    
    #nulls = patient_code_and_ccs[patient_code_and_ccs.isna().any(axis=1)]
    #set(nulls.Proc)
    
    patient_code_and_ccs.dropna(inplace= True)
    patient_code_and_ccs.drop("Proc", axis = 1, inplace = True)
    patient_code_and_ccs.set_index(['Patid', 'surgery_date'], inplace=True)
    
    patient_code_and_ccs['CCS_proc'] = patient_code_and_ccs['CCS_proc'].astype(int)
    patient_code_and_ccs['CCS_proc'] = patient_code_and_ccs['CCS_proc'].astype(str)
    
    patient_code_and_ccs_1hot = pd.get_dummies(patient_code_and_ccs["CCS_proc"])#, prefix="CCS_proc")
    patient_code_and_ccs_1hot = patient_code_and_ccs_1hot.groupby(['Patid', 'surgery_date']).max()
    
    #Rename the columns with the label names as well for ease of analysis
    ccs_proc_names['CCS_proc_and_name'] = ccs_proc_names['CCS_proc'] + "_" + ccs_proc_names['Label']
    ccs_proc_mapping = pd.Series(ccs_proc_names.CCS_proc_and_name.values,index=ccs_proc_names.CCS_proc).to_dict()
    patient_code_and_ccs_1hot.rename(columns = ccs_proc_mapping, inplace = True)
    
    return patient_code_and_ccs_1hot