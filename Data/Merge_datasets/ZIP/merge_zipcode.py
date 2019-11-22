# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:08:02 2019

@author: trishaj
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

save_data = True

#%%
#Read in the data files
zip_to_zcta = pd.read_csv("zip_to_zcta.csv", dtype={'ZIP_CODE': str, 'ZCTA': str})
census = pd.read_csv("zip5_census_feature_subset.csv", dtype={'ZCTA': str})


zip_to_zcta.set_index('ZCTA', inplace = True)
census.set_index('ZCTA', inplace = True)


census = census.join(zip_to_zcta)

census.isna().sum()

#hmm, maybe drop the median income column
#census.drop("Median income (dollars)", axis=1, inplace = True)


#%% Aggregate the education features
census['Pct_atleast_grad'] = census['Graduate or professional degree']
census['Pct_atleast_bach'] = census['Pct_atleast_grad'] + census['Bachelor\'s degree'] 
census['Pct_atleast_assoc'] = census['Pct_atleast_bach'] + census['Some college or associate\'s degree'] 
census['Pct_atleast_hs'] = census['Pct_atleast_assoc'] + census['High school graduate (includes equivalency)'] 


census.drop(['Graduate or professional degree',
             'Bachelor\'s degree',
             'Some college or associate\'s degree',
             'High school graduate (includes equivalency)',
             'Less than high school graduate'], axis=1, inplace = True)


#%% Aggregate the income features
census['Pct_atleast_75k'] = census['Population 15 years and over - $75,000 or more']
census['Pct_atleast_65k'] = census['Pct_atleast_75k'] + census['Population 15 years and over - $65,000 to $74,999']
census['Pct_atleast_50k'] = census['Pct_atleast_65k'] + census['Population 15 years and over - $50,000 to $64,999']
census['Pct_atleast_35k'] = census['Pct_atleast_50k'] + census['Population 15 years and over - $35,000 to $49,999']
census['Pct_atleast_25k'] = census['Pct_atleast_35k'] + census['Population 15 years and over - $25,000 to $34,999']
census['Pct_atleast_15k'] = census['Pct_atleast_25k'] + census['Population 15 years and over - $15,000 to $24,999']
census['Pct_atleast_10k'] = census['Pct_atleast_15k'] + census['Population 15 years and over - $10,000 to $14,999']

census.drop(['Population 15 years and over - $1 to $9,999 or loss',
             'Population 15 years and over - $10,000 to $14,999',
             'Population 15 years and over - $15,000 to $24,999',
             'Population 15 years and over - $25,000 to $34,999',
             'Population 15 years and over - $35,000 to $49,999',
             'Population 15 years and over - $50,000 to $64,999',
             'Population 15 years and over - $65,000 to $74,999',
             'Population 15 years and over - $75,000 or more'], axis=1, inplace = True)



#%%
#census.dropna(inplace=True)

census.reset_index(level=0, inplace=True)

census.dropna(inplace= True)

if save_data:
    census.to_csv('Zip5_census_lookup.csv', index=False)

#census.set_index("ZIP_CODE", inplace=True)

#%%
patient_zipcode_info = {}
with open('Get_zipcode_subset_output.csv') as f:
    next(f)
    for line in f:
        elements = [x.strip() for x in line.split(",")]
        patid = elements[0].strip('\'')
        zipcodes = elements[1].strip('\'')
        zipcode = zipcodes.split("_")[0]
        patient_zipcode_info[patid] = zipcode
 
patient_zipcodes = pd.DataFrame(list(patient_zipcode_info.items()), columns=['Patid', 'Zip']) 

       
#%% 

zip_to_income = pd.read_csv("Zip_to_income.csv", thousands = ",", dtype={'Zip': str})

zip_to_income.replace(".", np.nan, inplace= True)

zip_to_income.Mean = zip_to_income.Mean.str.replace(',','').astype(float)


#zip_to_income[zip_to_income['Mean'].isnull()] which rows have null mean

zip_to_income.Median.hist(bins=100)
zip_to_income.Mean.hist(bins=100)

#%% Join tables
patient_zipcodes = patient_zipcodes.set_index('Zip')
zip_to_income = zip_to_income.set_index('Zip')

patient_zip_income = patient_zipcodes.join(census)

#how many zipcodes do not have income info?
patient_zip_income.isna().sum()
