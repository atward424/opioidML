# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:59:25 2019

@author: atward
"""
#%%
import pandas as pd
from pprint import pprint
import numpy as np
import json
import os
print("Done!")
cpts = pd.read_csv('../../CPT_ICD_lookups/all_CPTs_with_descriptions.csv')
cpt_ccs = pd.read_csv('../../CPT_ICD_lookups/CPT_to_CCS.csv')
ccs_surg_names = pd.read_csv('../../CPT_ICD_lookups/CCS_names_surgery.csv')

#%%
dtypes = {'Proc_Cd':'str'}
surgs_all1 = pd.read_excel('../Metadata/surg_anes_cpts/CPTs_marked_by_Tony.xlsx', 'all_surgs', dtype = dtypes)
surgs_all2 = pd.read_excel('../Metadata/surg_anes_cpts/CPTs_marked_by_Tony.xlsx', 'nonsurgs_removed2', dtype = dtypes)
surgs_age = pd.read_excel('../Metadata/surg_anes_cpts/CPTs_marked_by_Tony.xlsx', 'allsurgs_byage', dtype = str)

#%%
surg_anes = pd.read_csv('../Metadata/surg_anes_cpts/12_21_ct_output.csv')
surg_anes2 = surg_anes.merge(cpt_ccs, left_on = 'Proc_Cd', right_on = 'CPT', how = 'left')
surg_anes3 = surg_anes2.merge(ccs_surg_names, on = 'CCS', how = 'left')
surg_anes4 = surg_anes3.merge(cpts, on = 'Proc_Cd', how = 'left')
#surg_anes5 = surg_anes4[['Proc_Cd','CCS', 'tony_surg', 
#                         'num_surgs_no_anes','num_surgs_with_anes',
#                         'procedure_name','CPT_DESCR']].sort_values(by = ['CCS', 'tony_surg', 'num_surgs_with_anes'], 
#                                                                   ascending = [True, False, False])
#%%
#surg_cpts = surg_anes.Proc_Cd[surg_anes.Proc_Cd.isin(surgs_all2.Proc_Cd)]
#
#aa = surg_anes4[surg_anes4.CCS.isna()]
#
#bb = surg_anes4[~surg_anes4.CCS.isna() &
#               surg_anes4.CCS_surgery.isna()]
tony_removed_CPTs = surgs_all1.loc[~surgs_all1.Proc_Cd.isin(surgs_all2.Proc_Cd), 'Proc_Cd']
colnms = ['Proc_Cd' + str(i) for i in [0,1,2,3]]
tony_nonsurg_cpts = []
for cn in colnms:
    
    trcpts = surgs_age.loc[~surgs_age[cn].isin(surgs_age[cn + 'm']), cn]
#    print(trcpts)
    tony_nonsurg_cpts += trcpts.tolist()
    
for cn in colnms:
    cn = cn + 'm'
    print(surgs_age.loc[surgs_age[cn].isin(trcpts), cn])

tony_nonsurg_cpts.remove('92567')
tony_nonsurg_cpts += tony_removed_CPTs.tolist()
#%%
writer = pd.ExcelWriter('../Metadata/surg_anes_cpts/surgical_cpts_to_mark_20191004.xlsx', engine='xlsxwriter')
cohorts = []
coh_nms = []
d1 = surg_anes4[surg_anes4.Proc_Cd.isin(surgs_all2.Proc_Cd)]
cohorts.append(d1)
coh_nms.append('surgical_CPTs')

d1 =surg_anes4[~surg_anes4.Proc_Cd.isin(surgs_all2.Proc_Cd) & 
               ~surg_anes4.Proc_Cd.isin(tony_nonsurg_cpts) &
               ~surg_anes4.CCS_surgery.isna()]
cohorts.append(d1)
coh_nms.append('surgical_CCS_not_surgery')

d1 =surg_anes4[surg_anes4.Proc_Cd.isin(tony_nonsurg_cpts)]
cohorts.append(d1)
coh_nms.append('marked_nonsurgical_CPTs')

d1 =surg_anes4[~surg_anes4.Proc_Cd.isin(surgs_all2.Proc_Cd) & 
               ~surg_anes4.Proc_Cd.isin(tony_nonsurg_cpts) &
               ~surg_anes4.CCS.isna() &
               surg_anes4.CCS_surgery.isna()]
cohorts.append(d1)
coh_nms.append('marked_nonsurgical_CCS')

d1 =surg_anes4[~surg_anes4.Proc_Cd.isin(surgs_all2.Proc_Cd) & 
               ~surg_anes4.Proc_Cd.isin(tony_nonsurg_cpts) &
               surg_anes4.CCS.isna()]
cohorts.append(d1)
coh_nms.append('missing_CCS')

for i, data in enumerate(cohorts):
    zz1 = data[['Proc_Cd', 
                             'num_surgs_no_anes','num_surgs_with_anes',
                             'Procedure_Name','Procedure_Description', 'CCS', 'Description',]].sort_values(by = 
              ['num_surgs_with_anes'], 
                                                                       ascending = [False])
    
    zz1.loc[(zz1.num_surgs_no_anes > 0) & (zz1.num_surgs_no_anes < 10), 
            'num_surgs_no_anes'] = 10
    zz1.loc[(zz1.num_surgs_with_anes > 0) & (zz1.num_surgs_with_anes < 10), 
            'num_surgs_with_anes'] = 10
    zz1.to_excel(writer, sheet_name = coh_nms[i], index  = False)
writer.save() 

#%%

aa =surg_anes4[~surg_anes4.Proc_Cd.isin(surgs_all2.Proc_Cd) & 
               ~surg_anes4.Proc_Cd.isin(tony_nonsurg_cpts) &
               ~surg_anes4.CCS.isna() &
               surg_anes4.CCS_surgery.isna()]
aa =surg_anes4[~surg_anes4.Proc_Cd.isin(surgs_all2.Proc_Cd) & 
               ~surg_anes4.Proc_Cd.isin(tony_nonsurg_cpts) &
               surg_anes4.CCS.isna()]