# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:43 2019

@author: trishaj
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from convert_proc_to_CCS import convert_procedure_to_CCS
from convert_diag_to_CCS import convert_diag_to_CCS
from split_data import StratifiedGroupShuffleSplit

save_data = True

#%%
#Read in the data files
dataset_type = 'ZIP/'

# Drug history
all_drug_counts = pd.read_csv(dataset_type + "Days_supply_of_AHFS_classes_output.csv")
ahfs_names = pd.read_csv("AHFS_lookup.csv")


# 1 year procedure and diag history 
diag_history = pd.read_csv(dataset_type + "Diag_history_output.csv")
proc_history = pd.read_csv(dataset_type + "Proc_history_output.csv")
cpt_proc_history = pd.read_csv(dataset_type + "CPT_proc_history_output.csv") 

chronic_pain= pd.read_csv(dataset_type + "get_pain_m_codes_output_coms_per_pt.csv")


# Opioid usage history
mme_periop = pd.read_csv(dataset_type + "MME_extra_variables.csv")
mme_0_30 = pd.read_csv(dataset_type + "MME_extra_variables_0_30d_before.csv")
mme_0_180 = pd.read_csv(dataset_type + "MME_extra_variables_0_180d_before.csv")
mme_0_365 = pd.read_csv(dataset_type + "MME_extra_variables_0_365d_before.csv")

if (dataset_type == 'ZIP/'):
    patient_features = pd.read_csv(dataset_type + "Patient_cohort_with_full_demographic_info_output.csv")


#%% TODO: deal with the consecutive surgeries with the same anesthesia code

#create a new dataframe with patid, surgery date, and age in days
patid_surg_name_age =  patient_features.filter(['Patid','age_in_days', 'surgery_date'], axis=1)
patid_surg_name_age.sort_values(by = ['Patid', 'surgery_date'], inplace=True)
patid_surg_name_age.drop_duplicates(['Patid', 'age_in_days'], keep='last', inplace=True)
patid_surg_name_age = patid_surg_name_age.set_index(['Patid','surgery_date'])

#create a new dataframe with patid, surgery date, and procedure code
patid_surg_name_date =  patient_features.filter(['Patid','Proc_Cd','surgery_date'], axis=1)
patid_surg_name_date.rename(columns={"Proc_Cd":"Proc"}, inplace = True) 

#Create a dataframe that is patid, surgery date, and a one hot encoding of the ccs class of surgery cpt code
surgery_proc_ccs = convert_procedure_to_CCS("CPT", patid_surg_name_date)
keep_same = {"Patid", "surgery_date"}
surgery_proc_ccs.columns = ['{}{}'.format(c, '' if c in keep_same else '_surgerycode') for c in surgery_proc_ccs.columns]

#Add age in years to the dataframe
surgery_proc_ccs = surgery_proc_ccs.join(patid_surg_name_age)


#Reindex, and sort the data by patid, surgery date
surgery_proc_ccs.reset_index( inplace=True)
surgery_proc_ccs.sort_values(by = ['Patid', 'surgery_date'], inplace=True)

#create a copy for referencing 
surgery_proc_ccs_copy= surgery_proc_ccs.copy()

#Initialize the difference between surgery dates
surgery_proc_ccs['diff_next_surg'] = 0
surgery_proc_ccs['patid_next'] = 0


#Find the difference between surgery dates
surgery_proc_ccs['diff_next_surg'].iloc[:-1] = np.array(surgery_proc_ccs.age_in_days.iloc[1:]) - surgery_proc_ccs.age_in_days.iloc[:-1]
surgery_proc_ccs['patid_next'].iloc[:-1] = np.array(surgery_proc_ccs.Patid[1:])
print(surgery_proc_ccs.shape)

#Find the duplicates, ie, rows where same patient within 2 days
dups = np.where((surgery_proc_ccs.diff_next_surg > 0) &
                    (surgery_proc_ccs.diff_next_surg <= 2) &
                    (surgery_proc_ccs.patid_next == surgery_proc_ccs.Patid))
old_surg_dates = np.array(surgery_proc_ccs.surgery_date)
surgery_proc_ccs.surgery_date.iloc[dups[0] + 1] = np.array(surgery_proc_ccs.surgery_date.iloc[dups[0]]) #should it be +1??

#keep flagging the duplicates
while np.all(old_surg_dates != np.array(surgery_proc_ccs.surgery_date)):
    old_surg_dates = np.array(surgery_proc_ccs.surgery_date)
    print(surgery_proc_ccs.shape)
    dups = np.where((surgery_proc_ccs.diff_next_surg > 0) & (surgery_proc_ccs.diff_next_surg <= 2) & (surgery_proc_ccs.patid_next == surgery_proc_ccs.Patid))
    dup_surg_dates = np.array(surgery_proc_ccs.surgery_date.iloc[dups])
    surgery_proc_ccs.surgery_date.iloc[dups + 1] = np.array(surgery_proc_ccs.surgery_date.iloc[dups])


surgery_proc_ccs.sort_values(by = ['Patid', 'surgery_date'], inplace=True)
surgery_proc_ccs['diff_next_surg'] = 0
surgery_proc_ccs['patid_next'] = 0
surgery_proc_ccs['diff_next_surg'].iloc[:-1] = np.array(surgery_proc_ccs.age_in_days.iloc[1:]) - surgery_proc_ccs.age_in_days.iloc[:-1]
surgery_proc_ccs['patid_next'].iloc[:-1] = np.array(surgery_proc_ccs.Patid[1:])


surgery_proc_ccs2= surgery_proc_ccs.drop(['diff_next_surg', 'patid_next', 'age_in_days'], axis=1)

surgery_proc_ccs3 = pd.DataFrame(surgery_proc_ccs2.groupby(['Patid', 'surgery_date']).aggregate(np.any))

surgery_proc_ccs3.reset_index( inplace=True)


#%% Turn region into 1 hot encoding
if (dataset_type == 'ZIP/'):
    #states = pd.get_dummies(patient_features["state"])
    regions = pd.get_dummies(patient_features["Division"])
    #patient_features = patient_features.join(states)
    patient_features = patient_features.join(regions)
    patient_features.drop("state", axis=1, inplace=True)
    patient_features.drop("Region", axis=1, inplace=True)
    patient_features.drop("Division", axis=1, inplace=True)
    
    
#%% Deal with null median income values
# FOr patients without median income listed, estimate with weighted average

median_estimate =   12500/100 * (patient_features['Pct_atleast_10k'] - patient_features['Pct_atleast_15k'])  + \
                    20000/100 * (patient_features['Pct_atleast_15k'] - patient_features['Pct_atleast_35k'])  + \
                    42500/100 * (patient_features['Pct_atleast_35k'] - patient_features['Pct_atleast_50k']) + \
                    57500/100 * (patient_features['Pct_atleast_50k'] - patient_features['Pct_atleast_65k']) + \
                    60000/100 * (patient_features['Pct_atleast_65k'] - patient_features['Pct_atleast_75k']) + \
                    100000/100* patient_features['Pct_atleast_75k'] + \
                    5000/100 * (100 - patient_features['Pct_atleast_10k'])
                    
median_estimate_error = median_estimate - patient_features['median_income']
plt.hist(median_estimate_error)

#How many median incomes are null?
np.sum(pd.isnull(patient_features['median_income']))    
                    
plt.hist(patient_features['median_income'])

patient_features['median_income'].fillna(median_estimate, inplace = True)    

#How many median incomes are null?
assert np.sum(pd.isnull(patient_features['median_income'])) == 0
#shoudl be 0


patient_features = patient_features.fillna('0')
#can remove procedure code, since it is now a 1 hot
patient_features= patient_features.drop(['Proc_Cd'], axis=1)

patient_features.drop_duplicates(['Patid', 'age_in_days'], keep='last', inplace=True)

patient_features.set_index( ['Patid', 'surgery_date'], inplace=True)


surgery_proc_ccs3.set_index( ['Patid', 'surgery_date'], inplace=True)

patient_features = surgery_proc_ccs3.join(patient_features)


#create a new dataframe with patid, surgery date, and procedure code
#patid_surg_name_date =  patient_features.filter(['Patid','Proc_Cd','surgery_date'], axis=1)

    
#%% Add mme info for periop and cumulative history up to a year back
# Extract only certain columns 
mme_0_30 = mme_0_30[["Patid", "surgery_date", "MME_opioid", "Days_Sup_opioid"]]
mme_0_180 = mme_0_180[["Patid", "surgery_date", "MME_opioid", "Days_Sup_opioid"]]
mme_0_365 = mme_0_365[["Patid", "surgery_date", "MME_opioid", "Days_Sup_opioid"]]

#Rename the columns to include the history timeframe
keep_same = {"Patid", "surgery_date"}
mme_periop.columns = ['{}{}'.format(c, '' if c in keep_same else '_periop') for c in mme_periop.columns]
mme_0_30.columns = ['{}{}'.format(c, '' if c in keep_same else '_0_30') for c in mme_0_30.columns]
mme_0_180.columns = ['{}{}'.format(c, '' if c in keep_same else '_0_180') for c in mme_0_180.columns]
mme_0_365.columns = ['{}{}'.format(c, '' if c in keep_same else '_0_365') for c in mme_0_365.columns]

#ptdf = patient_features[['num_med_claims']]


#Join with the features data frame
#patient_features.set_index( ['Patid', 'surgery_date'], inplace=True)
mme_periop = mme_periop.set_index(['Patid', 'surgery_date'])
mme_periop = mme_periop.drop([ 'MME_missing_periop'], axis=1)

mme_0_30 = mme_0_30.set_index(['Patid', 'surgery_date'])
mme_0_180= mme_0_180.set_index(['Patid', 'surgery_date'])
mme_0_365 = mme_0_365.set_index(['Patid', 'surgery_date'])

patient_features = patient_features.join(mme_periop).fillna('0')
patient_features = patient_features.join(mme_0_30).fillna('0')
patient_features = patient_features.join(mme_0_180).fillna('0')
patient_features = patient_features.join(mme_0_365).fillna('0')


#%% Look at how many people have pain. 
chronic_pain = chronic_pain[['Patid', 'surgery_date','pain_combined']]
chronic_pain = chronic_pain.set_index(['Patid', 'surgery_date'])
patient_features = patient_features.join(chronic_pain).fillna('0')

#%% Pivot and fix drug features
ahfs_names['class_and_name'] = ahfs_names['AHFSCLSS'] + "_" + ahfs_names['DESCRIPTION']

ahfs_mapping = pd.Series(ahfs_names.class_and_name.values,index=ahfs_names.AHFSCLSS).to_dict()

drug_history = all_drug_counts.pivot_table(
        values='Total_days_supply', 
        index=['Patid', 'surgery_date'], 
        columns='Ahfsclss', 
        aggfunc=np.sum)


drug_history.rename(columns = ahfs_mapping, inplace = True)

drug_history.drop(['000000_UNKNOWN', 'UNK_UNKNOWN'], axis=1, inplace = True)


#Do a left join
patient_features = patient_features.join(drug_history).fillna('0')




#%% Add patient diag->ccs data

patient_diag_ccs_1hot = convert_diag_to_CCS(diag_history)
patient_features = patient_features.join(patient_diag_ccs_1hot).fillna('0')

#%% Add patient proc->ccs data
patient_proc_ccs_1hot = convert_procedure_to_CCS("ICD", proc_history)
patient_cpt_proc_ccs_1hot = convert_procedure_to_CCS("CPT", cpt_proc_history)

patient_proc_ccs_1hot.reset_index(inplace= True)
patient_cpt_proc_ccs_1hot.reset_index(inplace= True)
patient_proc_ccs_1hot2 = pd.concat([patient_proc_ccs_1hot, patient_cpt_proc_ccs_1hot],  axis=0).fillna(0)
patient_proc_ccs_1hot3 = patient_proc_ccs_1hot2.groupby(['Patid', 'surgery_date']).max()



#keep_same = {"Patid", "surgery_date"}
patient_proc_ccs_1hot3.columns = ['{}{}'.format(c, '_ccs_proc') for c in patient_proc_ccs_1hot3.columns]


#join with the features frame
patient_features = patient_features.join(patient_proc_ccs_1hot3).fillna('0')



#%% Turn into floats

cols=[i for i in patient_features.columns if i not in ["Proc_Cd"]]
for col in cols:
    patient_features[col] = patient_features[col].astype(float)

#%% remove top .1 % of outliers for num med claims and inpatient visits
med_claims_outlier = patient_features['num_med_claims'].quantile(0.999)
#med_claims_no_outlier = patient_features[patient_features["num_med_claims"] < med_claims_outlier]['num_med_claims']
#med_claims_no_outlier.hist(bins=80)
patient_features= patient_features[patient_features["num_med_claims"] < med_claims_outlier]


inp_visits_outlier = patient_features['inp_visits'].quantile(0.99)
#inp_visits_no_outlier = patient_features[patient_features["inp_visits"] < inp_visits_outlier]['inp_visits']
#inp_visits_no_outlier.hist(bins=20)
patient_features= patient_features[patient_features["inp_visits"] < inp_visits_outlier]


#%% Look at histograms of patient features and outcome variables 

print_plots = False

if print_plots:

    patient_features['Gender'].hist()
    patient_features['age_in_days'].hist(bins=20)
    patient_features['num_prescriptions'].hist(bins=80)
    patient_features['num_med_claims'].hist(bins=50)
    
    patient_features['outp_visits'].hist(bins=20)
    patient_features['ED_visits'].hist(bins=10)
    patient_features['inp_visits'].hist(bins=50)
    patient_features['median_income'].hist(bins=50)

    
    patient_features['opioid_90_180d_after'].hist()

    patient_features['Pct_divorced'].hist(bins=50)



led_num_claims = np.log(patient_features.num_med_claims[patient_features.num_med_claims >0])

patient_features.num_prescriptions[patient_features.num_prescriptions >0] = np.log(patient_features.num_prescriptions[patient_features.num_prescriptions >0])


np.sqrt(patient_features['num_prescriptions']).hist(bins=50 )

patient_features['num_prescriptions'].hist(bins=50 )


patient_features[['num_prescriptions']] = np.log(patient_features[['num_prescriptions']].replace(0, np.nan))
patient_features['num_prescriptions'].fillna(0, inplace=True)

patient_features[['num_med_claims_log']] = np.log(patient_features[['num_med_claims']].replace(0, np.nan))
patient_features['num_med_claims_log'].fillna(0, inplace=True)
patient_features['num_med_claims_log'].hist(bins=50)

patient_features[['outp_visits_log']] = np.log(patient_features[['outp_visits']].replace(0, np.nan))
patient_features['outp_visits_log'].fillna(0, inplace=True)
patient_features['outp_visits_log'].hist(bins=20)

patient_features[['ED_visits_log']] = np.log(patient_features[['ED_visits']].replace(0, np.nan))
patient_features['ED_visits_log'].fillna(0, inplace=True)
patient_features['ED_visits_log'].hist(bins=50)

patient_features[['median_income_log']] = np.log(patient_features[['median_income']].replace(0, np.nan))
patient_features['median_income_log'].fillna(0, inplace=True)
patient_features['median_income_log'].hist(bins=50)

#%% Remove any columns that are all 0. 
patient_features.columns[(patient_features == 0).all()]
patient_features = patient_features.loc[:, (patient_features != 0).any(axis=0)]
column_names = list(patient_features.columns)

#%% Plots
#rows_sums_true = patient_features.sum(axis=0)

rows_sums_boolean = patient_features.astype(bool).sum(axis=0)

rows_sums_proc_ccs = rows_sums_boolean.iloc[570:] #plot ccs procedure code history
rows_sums_proc_ccs.plot('bar')


rows_sums_surgcode_ccs = rows_sums_boolean.iloc[:100] #plotccs surgery codes
rows_sums_surgcode_ccs.plot('bar')

rows_sums_drug = rows_sums_boolean.iloc[161:332] #plot drug history
rows_sums_drug.plot('bar')

rows_sums_diag_ccs = rows_sums_boolean.iloc[333:572] #plot ccs diag history
rows_sums_diag_ccs.plot('bar')

rows_sums_boolean.to_csv("boolean_row_sums.csv")

rows_sums_boolean_privacy = rows_sums_boolean[rows_sums_boolean < 10] = 0

rows_sums_boolean.to_csv("boolean_row_sums_privacy.csv")





#%% Try looking at summary
summary = patient_features.describe()
summary = summary.T
summary.to_csv("patient_Feature_summary.csv")

#%% Event occurrence
number_of_events = np.sum(patient_features['opioid_90_180d_after'])
number_of_events_pct = np.sum(patient_features['opioid_90_180d_after'])/np.shape(patient_features['opioid_90_180d_after'])[0]


print("Number of patient surgeries with event is" ,  number_of_events )
print("Number of patient surgeries in total is" ,  np.shape(patient_features['opioid_90_180d_after'])[0] )
print("Percent of people with event is" ,  number_of_events_pct*100 )

#%% Do some data normalization 

#First, turn all the columns with percentage data into actual percent (between 0 and 1)
cols_with_pct = patient_features.columns[pd.Series(patient_features.columns).str.startswith('Pct')]
for col in cols_with_pct:
    patient_features[col] = patient_features[col]/100

#Next, take log transform of the remaining continuous variables
#%% SPlit training and test, make sure that a patient doesnt appear in both train and test
#TODO: redo the splitting. split by group, and split stratified by outcome

patient_features = patient_features.reset_index()
train_features, test_features = StratifiedGroupShuffleSplit(patient_features)



patient_features_event = patient_features.loc[patient_features['opioid_90_180d_after'] == 1]
patient_features_no_event = patient_features.loc[patient_features['opioid_90_180d_after'] == 0]

#See how manysurgeries patients have. split by whether they have the event or not
num_surg_patients_event = patient_features_event.Patid.value_counts().value_counts()
num_surg_patients_no_event = patient_features_no_event.Patid.value_counts().value_counts()


patient_features.Patid.value_counts().value_counts()

#%% Save the file
if save_data:
    train_features.to_csv('../Patient_data/' + dataset_type + 'train_features_1pct.csv', index=False)
    test_features.to_csv('../Patient_data/' + dataset_type + 'test_features_1pct.csv', index=False)


#%% Look at distributions of drugs
if print_plots:

    drugs = drug_history.drop(['Patid', 'surgery_date'], axis=1)
    drugs = drug_history.fillna('0')
    drugs= drugs.astype('float64')
    drug_means = drugs.mean(axis = 0) 
    drug_stddevs = drugs.std(axis = 0) 
    
    
    drug_means_plot = drug_means.plot(style='.') #plot the means (num)
    drug_means_plot_hist = drug_means.hist() #plot the distribution of the means
    drug_stddevs_plot = drug_stddevs.hist() #plot the distribution of the SD
    
    #For each drub, how many people have gotten a prescription 
    num_ppl_with_drugs = drugs.astype(bool).sum(axis=0)
    num_ppl_with_drugs.plot(style='.')
    num_ppl_with_drugs.hist()
    
    
    #try a heatmap
    plt.pcolor(drugs)
    plt.show()

