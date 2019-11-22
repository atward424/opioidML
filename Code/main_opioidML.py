#%%

from medical_ML3 import Experiment, split_cohort
import pandas as pd
import os
#%%
def train_val(RESULT_DIR, alldata, to_exclude, test_ind_col, models, label, oversample_rate = 1,
                  imputer = 'iterative', add_missing_flags = True):
    print('\n\n' + 'STARTING EXPERIMENT FOR ' + RESULT_DIR + '\n\n')
    expt = Experiment(alldata, label = label, 
                      to_exclude = to_exclude, 
                      test_ind_col = test_ind_col, drop = 'all', 
                      result_dir = RESULT_DIR)
    cv = 5
    score_name = 'AUC'
    expt.predict_models_from_groups(0, models, cv=cv, score_name=score_name, mode='classification',
                                                    oversample_rate = oversample_rate, 
                                                   imputer = imputer, add_missing_flags = add_missing_flags)
    expt.save_and_plot_results(models, 
                               cv = 5, test = False)
    
def train_val_test(RESULT_DIR, alldata, to_exclude, test_ind_col, models, ascvd_est, label, oversample_rate = 1,
                  imputer = 'iterative', add_missing_flags = True):
    print('\n\n' + 'STARTING EXPERIMENT FOR ' + RESULT_DIR + '\n\n')
    expt = Experiment(alldata, label = label, 
                      to_exclude = to_exclude, 
                      test_ind_col = test_ind_col, drop = 'all', 
                      result_dir = RESULT_DIR)

    for model in models:
        expt.classification_ascvd(model, oversample_rate = oversample_rate, imputer = imputer, add_missing_flags = add_missing_flags)
    
#    test_on_new_cohort(RESULT_DIR, expt, alldata, to_exclude = to_exclude,
#                       test_ind_col = test_ind_col,
#                       models = models, ascvd_est = ascvd_est)
    expt.predict_on_test(models, out_dir = RESULT_DIR)#, test_file = '../Data/cohort/test_' + datafile)
    to_exclude['pce_invalid_vars'] = True
    pce_train_est2, pce_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
    expt.save_and_plot_results(models, 
                               cv = 5, test = True)

    for test_res_dir in test_others.keys():
        test_on_new_cohort(RESULT_DIR + '/' + test_res_dir, expt, alldata, 
                           test_others[test_res_dir], 
                           test_ind_col, models, 
                           ascvd_est)
    
    
def test_on_new_cohort(R2, expt, alldata, to_exclude, test_ind_col, models, 
                       ascvd_est):
    if not os.path.isdir(R2): os.mkdir(R2)
    _, test_data = split_cohort(alldata, to_exclude, test_ind_col, drop = 'all')
    expt.test_data = test_data
    expt.predict_on_test(models, test_file = None,
                        out_dir = R2)
    to_exclude['pce_invalid_vars'] = True
    ascvd_train_est2, ascvd_test_est2 = split_cohort(ascvd_est, to_exclude, test_ind_col, drop = 'all')
    expt.save_and_plot_test_results(models + ['PCE'], 
                               cv = 5, pce_file = ascvd_train_est2, 
                         test_pce_file = ascvd_test_est2,
                              out_dir = R2)
#%%    
if __name__ == '__main__':
    datafile = 'train_features_1pct.csv'
    alldata = pd.read_csv('../Data/Patient_data/ZIP/' + datafile)
#    pce_train_est = ascvd_est[(ascvd_est.pce_cohort == 1) &
#                            (ascvd_est.test_ind == 0)]
#    pce_test_est = ascvd_est[(ascvd_est.pce_cohort == 1) &
#                            (ascvd_est.test_ind == 1)]
    test_ind_col = 'test_ind'
    label = 'opioid_90_180d_after'
#     expt = Experiment('../Data/cohort/' + datafile, to_exclude, test_ind_col, drop = 'all')
#    import pdb; pdb.set_trace()
    
    models = [
           'logreg'
             ,
             'lasso2'
             ,
             'rf'
             ,
             'gbm'
#             ,
#             'xgb'
              ]
    
        
    imputer = 'iterative'
    add_missing_flags = True
    imputer = 'simple'
    add_missing_flags = False
    train_val(RESULT_DIR = '../Results/test_1122new2', alldata = alldata, 
                   to_exclude = None,
                   test_ind_col = None, models = models, label = label,
                  imputer = imputer, add_missing_flags = add_missing_flags)

#     datafile = 'allvars.csv'
#     alldata = pd.read_csv('../Data/cohort/' + datafile)
#     for res_dir in expts.keys():
        
#         imputer = 'iterative'
#         add_missing_flags = True
#         if expts[res_dir]['pce_invalid_vars']:
#             imputer = 'simple'
#             add_missing_flags = False
#         train_val_test(RESULT_DIR = '../Results/allvars_' + res_dir + '_0913', alldata = alldata, 
#                        to_exclude = expts[res_dir],
#                        test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label,
#                       imputer = imputer, add_missing_flags = add_missing_flags)
        
#    train_val_test(RESULT_DIR = '../Results/test_new2', alldata = alldata, 
#                   to_exclude = {'pce_cohort': False,
#                            'pce_invalid_vars': False,
#                            'cvd_bl': True,
#                            'antilpd': True,
#                            'oldyoung': True} 
##                             'agebl': 80}
#                   ,
#                   test_ind_col = test_ind_col, models = models, ascvd_est = ascvd_est, label = label)
    
    