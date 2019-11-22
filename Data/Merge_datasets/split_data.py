# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:12:53 2019

@author: trishaj
"""
# Source: https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit

#%%
import numpy as np
import pandas as pd

print_stats = True


def StratifiedGroupShuffleSplit(patient_features):

    # create empty train and test datasets
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()
    
    hparam_mse_wgt = 0.1 # must be between 0 and 1
    assert(0 <= hparam_mse_wgt <= 1)
    train_proportion = 0.8 # must be between 0 and 1
    assert(0 <= train_proportion <= 1)
    test_proportion = 1 - train_proportion # must be between 0 and 1
    assert(train_proportion+test_proportion == 1)
    
    
    subject_grouped_df_main = patient_features.groupby(['Patid'], sort=False, as_index=False)
    category_grouped_df_main = patient_features.groupby('opioid_90_180d_after').count()[['Patid']]/len(patient_features)*100


    def calc_mse_loss(df):
        grouped_df = df.groupby('opioid_90_180d_after').count()[['Patid']]/len(df)*100
        df_temp = category_grouped_df_main.join(grouped_df, on = 'opioid_90_180d_after', how = 'left', lsuffix = '_main')
        df_temp.fillna(0, inplace=True)
        df_temp['diff'] = (df_temp['Patid_main'] - df_temp['Patid'])**2
        mse_loss = np.mean(df_temp['diff'])
        return mse_loss
    i = 0
    for _, group in subject_grouped_df_main:
        
        #initialize the train_features and tes_features datasets with data from the first two groups
        if (i < 2):
            if (i == 0): #add to train
                train_features = train_features.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            else: #add to test
                test_features = test_features.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
    
        mse_loss_diff_train = calc_mse_loss(train_features) - calc_mse_loss(train_features.append(pd.DataFrame(group), ignore_index=True))
        #mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(df_val.append(pd.DataFrame(group), ignore_index=True))
        mse_loss_diff_test = calc_mse_loss(test_features) - calc_mse_loss(test_features.append(pd.DataFrame(group), ignore_index=True))
    
        total_records = len(train_features) +  len(test_features)
        #assert(total_records == patient_features.shape[0])
    
        len_diff_train = (train_proportion - (len(train_features)/total_records))
        #len_diff_val = (val_test_proportion - (len(df_val)/total_records))
        len_diff_test = (test_proportion - (len(test_features)/total_records)) 
    
        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        #len_loss_diff_val = len_diff_val * abs(len_diff_val)
        len_loss_diff_test = len_diff_test * abs(len_diff_test)
    
        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1-hparam_mse_wgt) * len_loss_diff_train)
        #loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1-hparam_mse_wgt) * len_loss_diff_val)
        loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1-hparam_mse_wgt) * len_loss_diff_test)
    
        if (max(loss_train,loss_test) == loss_train):
            train_features = train_features.append(pd.DataFrame(group), ignore_index=True)
        else:
            test_features = test_features.append(pd.DataFrame(group), ignore_index=True)
    
        #print ("Group " + str(i) + ". loss_train: " + str(loss_train) + " | " + "loss_test: " + str(loss_test) + " | ")
        i += 1
    
    
    if print_stats:
        number_of_events = np.sum(patient_features['opioid_90_180d_after'])
        number_of_events_pct = np.sum(patient_features['opioid_90_180d_after'])/np.shape(patient_features['opioid_90_180d_after'])[0]
        print("Number of patient surgeries with event is" ,  number_of_events )
        print("Number of patient surgeries in total is" ,  np.shape(patient_features['opioid_90_180d_after'])[0] )
        print("Percent of people with event is" ,  number_of_events_pct*100 )
        
        
        number_of_events_train = np.sum(train_features['opioid_90_180d_after'])
        number_of_events_pct_train = np.sum(train_features['opioid_90_180d_after'])/np.shape(train_features['opioid_90_180d_after'])[0]
        print("Number of patient surgeries in training set with event is" ,  number_of_events_train )
        print("Number of patient surgeries in training set in total is" ,  np.shape(train_features['opioid_90_180d_after'])[0] )
        print("Percent of people in training set with event is" ,  number_of_events_pct_train*100 )
        
        number_of_events_test = np.sum(test_features['opioid_90_180d_after'])
        number_of_events_pct_test = np.sum(test_features['opioid_90_180d_after'])/np.shape(test_features['opioid_90_180d_after'])[0]
        print("Number of patient surgeries in test set with event is" ,  number_of_events_test )
        print("Number of patient surgeries in test set in total is" ,  np.shape(test_features['opioid_90_180d_after'])[0] )
        print("Percent of people in test set with event is" ,  number_of_events_pct_test*100 )
            

    #Make sure there is no overlap in the patids of train and test 
    train_patids = set(train_features.Patid)
    test_patids = set(test_features.Patid)
    assert(train_patids.intersection(test_patids ) == set())
    
    return train_features, test_features




#%%
    
#patient_features = pd.read_csv("train_features.csv")
#train_features, test_features = StratifiedGroupShuffleSplit(patient_features)
#
#
#patient_features_event = patient_features.loc[patient_features['opioid_90_180d_after'] == 1]
#patient_features_no_event = patient_features.loc[patient_features['opioid_90_180d_after'] == 0]



